"""VISAR correction

Code to correct image deformation for the KEPLER (and others) streak cameras
used for the VISAR device at the HED instrument at European XFEL.
"""

import json
import logging
from bisect import insort, bisect_left
from functools import cache, cached_property, wraps
from inspect import getmembers, ismethod
from math import ceil
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import toml
import xarray as xr
#from extra.components import XrayPulses
from extra_data import KeyData, by_id, open_run
from extra_data.exceptions import SourceNameError
from scipy.interpolate import griddata
from scipy.ndimage import zoom
from scipy.stats import median_abs_deviation

from .shock import find_shocks, show_roi
from ..utils import ppu_trigger, dipole_trigger, dipole_ppu_open, save_tiff, sample_name

__all__ = ["VISAR"]


log = logging.getLogger(__name__)


VISAR_DEVICES = {
    "KEPLER1": {
        "arm": "COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_1",
        "trigger": "HED_EXP_VISAR/TSYS/ARM_1_TRIG",
        "detector": ("HED_SYDOR_TEST/CAM/KEPLER_1:daqOutput", "data.image.pixels"),
        "ctrl": "HED_SYDOR_TEST/CTRL/CONTROL_UNIT_1",
    },
    "KEPLER2": {
        "arm": "COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_2",
        "trigger": "HED_EXP_VISAR/TSYS/ARM_2_TRIG",
        "detector": ("HED_SYDOR_TEST/CAM/KEPLER_2:daqOutput", "data.image.pixels"),
        "ctrl": "HED_SYDOR_TEST/CTRL/CONTROL_UNIT_2",
    },
    "VISAR_1w": {
        "arm": "COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_3",
        "trigger": "HED_EXP_VISAR/TSYS/ARM_3_TRIG",
        "detector": ("HED_EXP_VISAR/EXP/ARM_3_STREAK:daqOutput", "data.image.pixels"),
        "ctrl": "HED_EXP_VISAR/EXP/ARM_3_STREAK",
    },
    "SOP": {
        "trigger": "HED_EXP_VISAR/TSYS/SOP_TRIG",
        "detector": ("HED_EXP_VISAR/EXP/SOP_STREAK:daqOutput", "data.image.pixels"),
        "ctrl": "HED_EXP_VISAR/EXP/SOP_STREAK",
    },
}


def remap(image, source, target):
    return cv2.remap(
        image, source, target, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )


def resize(image, row_factor=2, column_factor=2):
    return cv2.resize(
        image,
        (image.shape[0] * row_factor, image.shape[1] * column_factor),
        interpolation=cv2.INTER_CUBIC,
    )


def format_train_ids(data):
    """Format train ID data

    This function takes a 1D array, it then formats the output to show the first
    element and formats the subsequent elements to only show the changing digits
    """
    if isinstance(data, np.ndarray) and data.shape == ():
        return f'[{data.tolist()}]'
    if len(data) == 0:
        return "[]"

    max_diff = (data - data[0]).max()
    ndigits = len(str(max_diff))
    out = f"[{data[0]}"
    for value in data[1:]:
        out += f" ..{str(value)[-ndigits:]}"
    out += "]"
    return out


def find_max(data: xr.DataArray, n: int, dim='dim_0') -> np.ndarray:
    """
    Find indices of rows containing the n largest maximum values along the second axis.
    Returns row indices in their original order, not sorted by value.

    Args:
        data: 2D xarray DataArray
        n: Number of rows to return

    Returns:
        numpy array of row indices in original order
    """
    # Get the maximum values along axis 1
    max_vals = data.max(dim=data.dims[1])

    # Get indices of n largest values
    top_n_idx = np.argpartition(max_vals, -n)[-n:]
    # Sort indices by their position, not by value
    return np.sort(top_n_idx).tolist()


def largest_group(arr):
    """
    Find the largest contiguous group in a sorted array of integers.

    Parameters:
        arr (numpy.ndarray): 1D sorted array of integers

    Returns:
        array containing the contiguous sequence
    """
    if len(arr) == 0:
        return arr[0:0]

    current_start = 0
    current_length = 1
    start = 0
    length = 1

    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1:
            current_length += 1
        else:
            if current_length > length:
                length = current_length
                start = current_start
            current_start = i
            current_length = 1

    # Check one last time in case the longest sequence was at the end
    if current_length > length:
        length = current_length
        start = current_start

    return arr[start : start + length]


def find_closest(shot_ids, frames, offset=0):
    """
    For each number in `shot_ids`, find the matching number in `frames`.
    If a matching number is missing, find the closest larger number.

    Parameters:
      ids: List[int], sorted list of integers
      frames: pd.Series, sorted series of integers

    Returns:
      List[int]: A list of numbers from `frames` corresponding to `shot_ids`
    """
    results = []

    for num in shot_ids:
        # Find the position to insert num in the sorted frames list
        pos = bisect_left(frames, num) + offset
        if pos < len(frames):  # If within bounds
            results.append(frames[pos])  # Add the found number or the closest larger number
        else:
            raise ValueError(f"No larger value found for {num} in frames")

    return results


def _cache(name, py_type=False):
    """Decorator to cache the result of a method in the (xarray) dataset.

    Parameters:
        name (str): The name under which the result will be cached in
    the dataset.
        py_type (bool): If True, the cached result will be converted
    to a Python type.

    The decorator checks if the result for the given name is already cached in
    the dataset. If not, it calls the decorated function to compute the result,
    caches it, and then returns it. If py_type is True, the result is converted
    to a python type before returning.

    note: the class must have a dataset object of type xarray.Dataset defined.
    """

    def decorator(func):
        @wraps(func)
        def inner(self, *args, **kwargs):
            if name not in self.dataset:
                data = func(self, *args, **kwargs)
                self.dataset[name] = data

            res = self.dataset[name]
            if py_type:
                return res.data.tolist()
            return res

        inner._is_cached = True

        return inner

    return decorator


class SaveFriend:
    """ functions to compute cached data and save Dataset to hdf5
    
    The main class must define a (xr.Dataset) dataset and a
    (extra_data.DataCollection) run object.
    """

    def _quantities(self, ignore=[]):
        """Return a list of all cache-able methods"""
        methods = getmembers(self, predicate=ismethod)
        return [
            (name, method)
            for name, method in methods
            if getattr(method, "_is_cached", False) and name not in ignore
        ]

    @cached_property
    def run_number(self):
        return self.run.run_metadata().get("runNumber", "?")

    @cached_property
    def proposal_number(self):
        return self.run.run_metadata().get("proposalNumber", "?")

    def compute(self, profile=False):
        try:
            if self.coords.trainId.size == 0:
                # nothing to compute
                return
        except AttributeError:
            if self.dataset.trainId.size == 0:
                return


        for name, quantity in self._quantities():
            t0 = perf_counter()
            quantity()
            if profile:
                log.info(f"{self.name}.{name}: {round((perf_counter()-t0)*1000, 3)}ms")

    def to_h5(self, path, group):
        self.compute()
        try:
            self.dataset.to_netcdf(
                path=path, mode="a", group=group, format="NETCDF4", engine="h5netcdf"
            )
        except Exception as ex:
            print(self.proposal_number, self.run_number)
            print(self.name)
            print(self.dataset)
            raise


class DIPOLE(SaveFriend):
    def __init__(self, visar, run, name="DiPOLE"):
        self.name = name

        # try:
        #     reference = int(visar.coords.where(visar.coords.type == 'reference', drop=True).trainId.data.tolist()[0])
        # except:
        #     reference = None
        # train_ids = dipole_trigger(run).tolist()
        # if isinstance(train_ids, int):
        #     train_ids = [train_ids]
        # if reference is not None:
        #     insort(train_ids, reference)
        # types = ['shot'] * len(train_ids)
        # if reference is not None:
        #     types[train_ids.index(reference)] = 'reference'

        # self.run = run.select_trains(by_id[train_ids])
        # self.dataset = xr.Dataset(coords={
        #         "trainId": np.array(train_ids, dtype=np.uint64),
        #         "type": ("trainId", types),})
        self.run = run
        if visar is None:
            trigger = dipole_trigger(run)
            coords = xr.Dataset(
                coords={
                    "trainId": trigger,
                    "type": ("trainId", ["shot"] * len(trigger)),
                }
            )
            self.dataset = xr.Dataset(coords=coords)
        else:
            self.dataset = xr.Dataset(coords=visar.coords)

    def info(self):
        print(self.format())

    def format(self, compact=False):
        """Format information about the VISAR component."""
        run_str = f'p{self.proposal_number:06}, r{self.run_number:04}'
        info_str = f"{self.name} properties for {run_str}:\n"

        if compact:
            return f"{self.name}, {run_str}"

        data = []
        for name, quantity in self._quantities():
            if name.startswith("_"):
                continue
            if name == 'trace':
                continue

            value = quantity()
            value = value[value.type == 'shot']
            name = value.name
            units = value.attrs.get("units", "")
            # only show a single value if it's not changing
            if len(np.unique(value)) == 1:
                value = f"{value.data[0]:.6g}"
            else:
                with np.printoptions(precision=3):
                    value = str(value.data)
            data.append((f"{name}:", value, units))

        span = max(len(e[0]) for e in data) + 1
        info_str += "\n".join(
            [f"  {name:<{span}}{value} {units}" for name, value, units in sorted(data)]
        )
        return info_str

    @_cache(name="Delay")
    def delay(self):
        delay = self.run["APP_DIPOLE/MDL/DIPOLE_TIMING", "actualPosition"]
        data = delay.xarray()
        data.attrs["units"] = delay.units
        return data

    @_cache(name="Energy")
    def energy(self):
        energy = self.run["APP_DIPOLE/MDL/DIPOLE_DIAGNOSTIC", "energy2W"]
        data = energy.xarray()
        data.attrs["units"] = energy.units
        return data

    @_cache(name="Trace")
    def trace(self, threshold_sigma: float = 3.0, margin: int = 5):
        """
        Find DiPOLE power signals

        Parameters:
            threshold_sigma (float): The number of standard deviations above the median
                to use as the threshold for detecting significant signals in the traces.
                Default is 3.0.
            margin (int): The additional margin in nanoseconds around the detected dipole
                signal. This is used to extend the trace. Default is 5.

        Returns:
            xr.DataArray: A DataArray containing the extracted power signals for each
            train ID, with time coordinates adjusted according to the specified margin.
            The dimensions are ["trainId", "time [ns]"], and the data is in Watts (W).
        """
        run = open_run(self.proposal_number, self.run_number)
        try:
            run['HED_PLAYGROUND/SCOPE/TEXTRONIX_TEST']
            run["HED_PLAYGROUND/SCOPE/TEXTRONIX_TEST:output"]
        except SourceNameError:
            return xr.DataArray([])

        sample_rate = run['HED_PLAYGROUND/SCOPE/TEXTRONIX_TEST', 'samplerate'].as_single_value()  # Hz
        dt = 1e9 / sample_rate  # [ns]
        margin = int(margin / dt)  # margin in # sample

        scope = run["HED_PLAYGROUND/SCOPE/TEXTRONIX_TEST:output", "ch1.corrected"]
        traces = scope.xarray()

        # train ID of the dipole trace is not reliable
        shot_ids = dipole_trigger(run)
        n_shots = shot_ids.size
        indices = find_max(traces, n_shots)
        traces = traces[indices]

        energy = self.energy()
        energy = energy[energy.type == "shot"].data

        noise_std = median_abs_deviation(traces, axis=1, scale="normal")
        threshold = traces.median(axis=1) + threshold_sigma * noise_std

        power_traces = []
        durations = []
        for trace, thresh, nrj in zip(traces, threshold, energy):
            indices = largest_group(np.where(trace > thresh)[0])
            if indices.size == 0:
                continue
            start, stop = indices[0], min(trace.size, indices[-1] + 1)

            dipole_duration = (stop - start) * dt * 1e-9  # [s]
            durations.append(dipole_duration / 1e-9)
            scaling = nrj / (trace[start:stop].sum() * dipole_duration)
            power = (
                trace[max(0, start - margin) : min(stop + margin, trace.size)] * scaling
            )
            power_traces.append(power)

        if len(power_traces) == 0:
            return xr.DataArray([])

        longest_trace = max(power_traces, key=len).size
        time_coord = (np.arange(longest_trace) - margin) * dt  # [ns]

        out = np.full((traces.shape[0], longest_trace), np.nan)
        for idx, power in enumerate(power_traces):
            out[idx, :power.size] = power

        return xr.DataArray(
            out,
            coords={"time [ns]": time_coord, "trainId": shot_ids},
            dims=["trainId", "time [ns]"],
            name="Power",
            attrs={"units": "W", "durations": durations},
        )


# class DIPOLE1(DIPOLE):
#     def __init__(self, run):
#         # shots = dipole_trigger(run, offset=-20).tolist()
#         # self.run = run.select_trains(by_id[shots])
#         # self.dataset = xr.Dataset(coords={'trainId': self.run.train_ids})

#         # shots = dipole_trigger(run).tolist()
#         self.name = 'DiPOLE'
#         traces = run["HED_PLAYGROUND/SCOPE/TEXTRONIX_TEST:output", 'ch1.corrected'].xarray()
#         tmax = traces.max(dim='dim_0').argmax().data.tolist()
#         tid = traces[int(tmax)].trainId.data.tolist()
#         self.run_trace = run.select_trains(by_id[[int(tid)]])

#         self.run = run.select_trains(by_id[dipole_trigger(run).tolist()])
#         self.dataset = xr.Dataset(coords={'trainId': self.run.train_ids})


class CalibrationData:
    def __init__(self, visar, file_path=None):
        self.visar = visar

        if file_path is not None:
            config = toml.load(file_path)
            self.config = config['global']
            self.config.update(config[self.visar.name])
        else:
            self.config = {}

    def __getitem__(self, key):
        return self.config[key]

    @property
    def dx(self):
        """Length per pixel in µm"""
        return self["dx"]

    @property
    def dipole_zero(self):
        """Dipole position at 0 ns delay, 0 ns sweep delay"""
        pixel_offset = self["pixDipole_0ns"][f"{self.visar.sweep_time()}ns"]
        return self.timepoint(pixel_offset)

    @property
    def fel_zero(self):
        """Xray position at 0 ns delay, 0 ns sweep delay"""
        return self.timepoint(self["pixXray"])

    @property
    def reference_trigger_delay(self):
        return self["positionTrigger_ref"][f"{self.visar.sweep_time()}ns"]

    @cached_property
    def timepoint(self):
        """Compute Time from pixel position in ns"""
        constants = self["timeAxisPolynomial"][f"{self.visar.sweep_time()}ns"]
        # Pad with leading 0 because there is no intercept for the time axis
        return np.poly1d(np.array([0, *constants])[::-1])

    @cache
    def map(self) -> tuple[np.ndarray, np.ndarray]:
        """Return input and output transformation maps"""
        try:
            tr_map_file = self["transformationMaps"][f"{self.visar.sweep_time()}ns"]
        except KeyError:
            return None  # no maps for this detector

        file_path = Path(self["dirTransformationMaps"]) / tr_map_file
        coords = np.loadtxt(file_path, delimiter=",")
        target = coords[..., 2:]
        source = coords[..., :2]

        y, x = self.visar.detector.entry_shape
        grid_1, grid_2 = np.mgrid[:y, :x]
        grid_z = griddata(target, source, (grid_1, grid_2), method="linear")
        map_1 = grid_z[..., 1].astype(np.float32)
        map_2 = grid_z[..., 0].astype(np.float32)

        return map_1, map_2

    @cache
    def dataset(self):
        data = xr.Dataset(
            data_vars={
                "dx": xr.DataArray(self.dx, attrs={"units": "um"}),
                "Drive pixel t0": xr.DataArray(self.dipole_zero, attrs={"units": "ns"}),
                "FEL zero": xr.DataArray(self.fel_zero, attrs={"units": "ns"}),
                "Reference trigger delay": xr.DataArray(
                    self.reference_trigger_delay, attrs={"units": "ns"}
                ),
            }
        )
        if self.map() is not None:
            data["Dewarp source"] = (["dim_0", "dim_1"], self.map()[0])
            data["Dewarp target"] = (["dim_0", "dim_1"], self.map()[1])
        return data

    def to_h5(self, file_path, group):
        self.dataset().to_netcdf(
            path=file_path,
            mode="a",
            group=group,
            format="NETCDF4",
            engine="h5netcdf",
        )


class _StreakCamera(SaveFriend):
    def __init__(self, run, name, config_file=None):
        self.run = run
        self.name = name
        self.visar = VISAR_DEVICES[name]

        self.sel = sel = run.select_trains(self.train_ids)
        #self.xray = XrayPulses(sel)

        if "arm" in self.visar:
            self.arm = sel[self.visar["arm"]]
        self.trigger = sel[self.visar["trigger"]]
        self.detector = sel[self.visar["detector"]]
        self.ctrl = sel[self.visar["ctrl"]]

        self.dipole = DIPOLE(self, sel)
        self.cal = CalibrationData(self, config_file)

        self.dataset = xr.Dataset(
            coords=self.coords,
            attrs={
                "calibration": json.dumps(self.cal.config),
                "run_number": self.run_number,
                "proposal_number": self.proposal_number,
            },
        )

    def __repr__(self):
        return f"<{type(self).__name__} {self.name}>"

    def _data(self, kd: KeyData) -> xr.DataArray:
        data = kd.xarray()
        data.attrs["units"] = kd.units
        return data

    def info(self):
        """Print information about the VISAR component"""
        print(f'{self.format()}\n\n{self.dipole.format()}')

    def format(self, compact=False):
        """Format information about the VISAR component."""
        run_str = f'p{self.proposal_number:06}, r{self.run_number:04}'
        info_str = f"{self.name} properties for {run_str}:\n"

        if compact:
            return f"{self.name}, {run_str}"

        data = []
        for name, quantity in self._quantities():
            if name.startswith("_"):
                continue
            # skip image data
            if name == "image":
                continue
            if name == "sweep_time":
                # special case
                data.append(("Sweep time:", f"{quantity():.6g}", "ns"))
                continue

            value = quantity()
            # only display information for shots, skip reference
            value = value[value.type == 'shot']
            name = value.name
            units = value.attrs.get("units", "")

            if name == 'Shocks' and value.size > 0:
                # display the breakout times only
                value = value[:, 0]

            # only show a single value if it's not changing
            if len(np.unique(value)) == 1:
                value = f"{value.data[0]:.6g}"
            else:
                with np.printoptions(precision=3):
                    value = str(value.data)

            data.append((f"{name}:", value, units))

        span = max(len(e[0]) for e in data) + 1
        info_str += "\n".join(
            [f"  {name:<{span}}{value} {units}" for name, value, units in sorted(data)]
        )

        def _train_ids(_type):
            return self.coords.where(self.coords.type == _type, drop=True).trainId.data

        info_str += f'\n\n  Train ID (shots): {format_train_ids(_train_ids("shot"))}'
        info_str += f'\n  Train ID (ref.): {format_train_ids(_train_ids("reference"))}'
        info_str += f"\n\n Sample ID: {', '.join(sample_name(self.run, int(tid)) for tid in _train_ids('shot'))}"
        return info_str

    @_cache(name="Difference X-drive")
    def fel_delay(self):
        fel_delay = (
            self.cal.fel_zero
            - self.cal.dipole_zero
            - self.dipole.delay()
            # - self.sweep_delay()
        )

#        # add fel duration if number of pulses > 1
#        multi_pulses = (self.xray.pulse_counts() > 1).index
#        fel_delay.loc[multi_pulses] += self.xray.train_durations().loc[multi_pulses] * 1e9
        return xr.DataArray(fel_delay, dims=["trainId"], attrs={"units": "ns"})

    @_cache(name="Sweep delay")
    def sweep_delay(self):
        for key in ["actualDelay", "actualPosition"]:
            if key in self.trigger:
                break
        else:
            raise KeyError(f"sweep delay property not found in {self.trigger}")

        data = self.trigger[key].xarray()
        data.attrs["units"] = self.trigger[key].units
        return data - self.cal.reference_trigger_delay

    @_cache(name="Sweep time", py_type=True)
    def sweep_time(self):
        """Sweep window in nanosecond

        We assume the sweep time does not change over a run
        """
        sw, units = self.ctrl.run_value("timeRange").split(" ")
        return xr.DataArray(int(sw), attrs={"units": units})

    @property
    def train_ids(self):
        # return self.coords.trainId.values
        return by_id[self.coords.trainId.values]

    @cached_property
    def coords(self):
        """Get train IDs of data with open PPU.

        The first trainID will be the first found train without signal
        (reference signal) followed by all trains with detector data and PPU
        opened.
        """
        # train ID with data in the run
        tids = (
            self.run[self.visar["detector"]].drop_empty_trains().train_id_coordinates()
        )

        # first we try to get the train IDs from the master trigger device
        shot_ids = dipole_trigger(self.run)
        if len(shot_ids) > 0:
            # most streak cameras have a delay between the trigger and the frame
            # so we look for the trainId of the the available frame if it does
            # not falls on the shot_id 
            try:
                if self.name == 'VISAR_1w' and self.run_number > 346:
                    offset = 1
                else:
                    offset = 0
                shot_ids = find_closest(shot_ids, tids.tolist(), offset=offset)
            except ValueError:
                # no frame found in this run
                shot_ids = []
            ref_ids = np.setdiff1d(tids, ppu_trigger(self.run) + dipole_trigger(self.run))
        else:
            if self.proposal_number == 6659:
                # only rely on trigger
                shot_ids = []
                ref_ids = np.array([])
            else:
                # else we try to get the train IDs from correlation between detector
                # frames and dipole open shutter
                ppu_open = dipole_ppu_open(self.run)

                # train ID with data and ppu open
                shot_ids = np.intersect1d(ppu_open, tids).tolist()

                # train IDs with data and ppu closed
                ref_ids = np.setdiff1d(tids, ppu_open.data.tolist() + ppu_trigger(self.run))

        train_ids = shot_ids
        types = ["shot"] * len(train_ids)

        if ref_ids.size > 0:
            insort(train_ids, ref_ids[0])
            types.insert(train_ids.index(ref_ids[0]), "reference")

        coords = xr.Dataset(
            coords={
                "trainId": np.array(train_ids, dtype=np.uint64),
                "type": ("trainId", types),
            }
        )
        return coords

    @_cache(name="image")
    def image(self):
        """Get corrected images

        Returns corrected data for the trains with PPU opened. If *reference*
        is True, returns the first frame with PPU closed instead.
        """
        data = self.detector.ndarray()
        # SOP and 1w VISAR is 2x2 binned
        # need to upscale it full size to use the time calibration
        data = np.array([resize(frame) for frame in data])
        data = np.rot90(data, 1, axes=(1, 2))
        data = np.array([np.flipud(frame) for frame in data])
        return xr.DataArray(data, dims=["trainId", "dim_0", "dim_1"])

    @_cache(name="Time axis")
    def _time_axis(self):
        axis = self.cal.timepoint(np.arange(self.image().shape[-1]))
        offset = self.cal.dipole_zero + self.dipole.delay() - self.sweep_delay()
        # offset by xray train duration
        if "APP_DIPOLE/MDL/DIPOLE_TIMING" in self.sel.control_sources:
            timing = self.sel["APP_DIPOLE/MDL/DIPOLE_TIMING"]
            extra_bunches = timing["addNbBunches"].ndarray().max()
            dt = {"4.5 MHz": 2.215385e-07, "2.25 MHz": 4.430769e-07}[timing["accRepRate"].ndarray()[-1].decode()]
            fel_offset = extra_bunches * dt * 1e9
        else:
            fel_offset = 0

        data = (
            np.repeat(axis[None, ...], self.coords.trainId.size, axis=0)
            - offset.data[:, None] - fel_offset
        )
        return xr.DataArray(data, dims=["trainId", "Time"], attrs={"units": "ns"})

    @_cache(name="Space axis")
    def _space_axis(self):
        axis = np.arange(self.image().shape[-2]) * self.cal.dx
        axis = axis - axis.mean()

        data = np.repeat(axis[None, ...], self.coords.trainId.size, axis=0)
        return xr.DataArray(data, dims=["trainId", "Space"], attrs={"units": "um"})

    def plot(self, train_id, ax=None, fig=None, file_path=None):
        self.compute()
        ds = self.dataset.sel(trainId=train_id)
        data = ds.image
        type_ = ds.type.data.tolist()

        time_axis = ds["Time axis"]
        space_axis = ds["Space axis"]

        # shot or reference index in the run
        frames = self.dataset.where(self.dataset.type == type_, drop=True)
        frame_index = frames.trainId.data.tolist().index(train_id)

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(9, 5))

        tid_str = f"{type_} ({frame_index + 1}/{frames.trainId.size}), tid:{train_id}"
        sample_str = f"sample: {sample_name(self.run, train_id)}"
        ax.set_title(f"{self.format(compact=True)}, {tid_str}\n{sample_str}")
        ax.set_ylabel(f'Distance [{space_axis.attrs.get("units", "?")}]')
        ax.set_xlabel(f'Time [{time_axis.attrs.get("units", "?")}]')

        extent = [time_axis[0], time_axis[-1], space_axis[0], space_axis[-1]]
        im = ax.imshow(
            data, extent=extent, cmap="jet", vmin=0, vmax=data.mean() + 3 * data.std()
        )

        ys, xs = np.where(data > 0)
        ax.set_xlim(xmin=time_axis[xs.min()], xmax=time_axis[xs.max()])
        ax.set_ylim(ymin=-space_axis[ys.max()], ymax=-space_axis[ys.min()])
        ax.set_aspect("auto")

        # Vertical line indicating the FEL laser
        fel_delay = ds["Difference X-drive"].data.tolist()
        ax.vlines(
            fel_delay,
            ymin=space_axis[0],
            ymax=space_axis[-1],
            linestyles="-",
            lw=2,
            color="purple",
            alpha=1,
        )

        # Vertical line indicating the shock breakout
        try:
            shocks = self.shocks().sel(trainId=train_id)
        except AttributeError:
            shocks = np.array([])

        if shocks.size > 0:
            ax.vlines(
                shocks[0],
                ymin=space_axis[0],
                ymax=space_axis[-1],
                linestyles="-.",
                lw=2,
                color="yellow",
                alpha=1,
            )

        # add second xaxis on top to label fel and shock breakouts
        ax2 = ax.twiny()
        ax2.xaxis.set_ticks_position("top")

        ticks = [f'FEL: {round(fel_delay, 3)}ns']
        if shocks.size > 0:
            value = round(float(shocks[0].data), 3)
            ticks += [f'Breakout: {value}ns\n']
            ticks += [f"Shock {i}\n" for i, _ in enumerate(shocks[1:], start=2)]
        ax2.set_xticks([fel_delay, *shocks])
        ax2.set_xticklabels(ticks)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_ylim(ax.get_ylim())
        ax2.set_aspect("auto")

        # add delta t fel-breakout information if shock is found
        if shocks.size > 0:
            from matplotlib.offsetbox import AnchoredOffsetbox, TextArea
            try:
                delta = round(shocks.data[0] - fel_delay, 2)
            except ValueError:
                delta = 'nan '
            box1 = TextArea(
                f"$\delta t$ fel-breakout: {delta}ns",
                textprops=dict(color="k")
            )
            # box = HPacker(children=[box1], align="center", pad=0, sep=5)
            anchored_box = AnchoredOffsetbox(
                loc='lower right',
                child=box1, pad=0.2,
                frameon=True,
                bbox_to_anchor=(1., 1.02),
                bbox_transform=ax.transAxes,
                borderpad=0.,
            )
            ax2.add_artist(anchored_box)

        from matplotlib.ticker import MultipleLocator

        if ys.min() < -400 or ys.max() > 400:
            major, minor = 400, 100
        else:
            major, minor = 200, 50

        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(major))
        ax.yaxis.set_minor_locator(MultipleLocator(minor))
        ax.grid(which="major", color="k", linestyle="--", linewidth=2, alpha=0.5)
        ax.grid(which="minor", color="k", linestyle=":", linewidth=1, alpha=1)

        try:
            fig.colorbar(im, ax=ax2)
        except Exception:
            pass
        fig.tight_layout()

        if file_path is not None:
            fig.savefig(file_path, bbox_inches="tight", format="png")

        return ax

    def to_h5(self, output=".", filename="VISAR_p{proposal:06}_r{run:04}.h5"):
        """Save the corrected VISAR data to an HDF5 file.

        This method saves the VISAR object, including, calibration data, and
        dipole data, to an HDF5 file, in the netCDF format.

        Parameters:
            output (str): The directory where the HDF5 file will be saved.
            filename (str): The filename format for the saved HDF5 file,
                            which can include placeholders for proposal and run
                            numbers.
        """
        fname = filename.format(proposal=self.proposal_number, run=self.run_number)
        fpath = Path(output) / fname

        super().to_h5(fpath, self.name)
        self.cal.to_h5(fpath, f"{self.name}/calibration")
        # self.dipole.to_h5(fpath, f"{self.name}/DiPOLE")

        fpath.chmod(0o777)

    def to_png(
        self,
        output=".",
        filename="p{proposal:06}_r{run:04}_{name}.png",
        plots_per_row=1,
    ):
        """Save the VISAR plots to PNG files.

        This method generates and saves plots for each train ID in the VISAR
        dataset to PNG files. The output filename can include placeholders for
        proposal and run numbers, as well as the name of the VISAR component.

        Parameters:
            output (str): The directory where the PNG files will be saved.
            filename (str): The filename format for the saved PNG files,
                            which can include placeholders for proposal, run numbers, and the name.
            plots_per_row (int): The number of plots to display per row in the saved figure.
        """
        fname = filename.format(name=self.name, proposal=self.proposal_number, run=self.run_number)
        fpath = Path(output) / fname

        self.compute()
        n_images = len(self.train_ids.value)

        import matplotlib.pyplot as plt

        rows = ceil(n_images / plots_per_row)
        cols = min(n_images, plots_per_row)

        fig, axes = plt.subplots(rows, cols, figsize=(9 * cols, 5 * rows))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        # Flatten axes for easy indexing, even if there's only one row or column
        axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

        for tid, ax in zip(self.train_ids.value, axes):
            self.plot(tid, ax, fig)

        # Turn off any unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis("off")

        fig.savefig(fpath, bbox_inches="tight", format="png")
        fpath.chmod(0o777)

    def save(self, output):
        output = Path(output)
        self.compute()
        if self.coords.trainId.size == 0:
            # nothing to compute
            with open(output/ f"{self.name}_p{self.proposal_number:06}_r{self.run_number:04}_NO_SHOT.txt", "w") as f:
                f.write("NO SHOT IN FOR THIS RUN")
            return

        # save info at text
        with open(output / f"{self.name}_p{self.proposal_number:06}_r{self.run_number:04}_INFO.txt", "w") as f:
            f.write(self.format())
            f.write("\n\n")
            f.write(self.dipole.format())
            f.write("\n")

        for train_id in self.coords.trainId:
            _tid = train_id.data.tolist()
            _type = train_id.type.data.tolist()

            def _save(data, path):
                data = data.squeeze()
                data = np.nan_to_num(data)
                if self.name.startswith('KEPLER'):
                    data = zoom(data, 1/4)
                save_tiff(data, path)

            # save raw frame as tiff
            raw = self.detector[by_id[[_tid]]].ndarray().squeeze()
            _save(raw, output / f"{self.name}_RAW_{_tid}_{_type}.tiff")
            # save dewarped frame as tiff
            dewarped = self.image().sel(trainId=_tid).data.squeeze()
            _save(dewarped, output / f"{self.name}_DEWARPED_{_tid}_{_type}.tiff")
            # save plot
            self.plot(_tid, file_path=output / f"{self.name}_{_tid}_{_type}.png")
            # save hdf5
            self.to_h5(output)


class _VISAR(_StreakCamera):
    @_cache(name="Zero delay")
    def zero_delay_position(self):
        return self._data(self.arm["zeroDelayPosition"])

    @_cache(name="Etalon thickness")
    def etalon_thickness(self):
        return self._data(self.arm["etalonThickness"])

    @_cache(name="Motor displacement")
    def motor_displacement(self):
        return self._data(self.arm["motorDisplacement"])

    @_cache(name="Sensitivity")
    def sensitivity(self):
        return self._data(self.arm["sensitivity"])

    @_cache(name="Temporal delay")
    def temporal_delay(self):
        return self._data(self.arm["temporalDelay"])

    @_cache(name="Shocks")
    def shocks(self):
        # TODO handle all parameters
        # TODO parallelize?
        # TODO it seems to be more accurate when working on cropped image, need to test
        # train_ids = self.coords.where(self.coords.type == "shot", drop=True).trainId.values.tolist()
        train_ids = self.train_ids.value

        if isinstance(train_ids, int):
            train_ids = [train_ids]

        shocks = [self._shocks(tid) for tid in train_ids]
        if len(shocks) == 0:
            return xr.DataArray(
                np.full((len(train_ids), 0), np.nan),
                dims=['trainId', 'Shock time'],
                coords={'trainId': train_ids}
            )

        # Determine the maximum length
        max_length = max(len(s) for s in shocks)
        shocks = [s + [np.nan] * (max_length - len(s)) for s in shocks]
        return xr.DataArray(
            shocks,
            dims=['trainId', 'Shock time'],
            coords={'trainId': train_ids},
            attrs={
                'units': 'ns',
                'roi_ref': self.cal['shock']['roi_ref'],
                'roi_phase': self.cal['shock']['roi_phase'],
            }
        )

    def _shocks(self, train_id, debug=False):
        ref = self.image().where(self.image().type == 'reference', drop=True)
        data = self.image().sel(trainId=train_id)
        time = self._time_axis().sel(trainId=train_id)

        if ref.size ==0 or data.size == 0:
            return []

        ref = ref.data.squeeze()
        data = data.data.squeeze()
        time = time.data.squeeze()

        ref = cv2.resize(ref, (1024, 1024))[::-1]
        data = cv2.resize(data, (1024, 1024))[::-1]
        time = np.linspace(time[0], time[-1], 1024)

        # # Debug
        # if debug:
        #     show_roi(ref, roi_ref, 'ref')
        #     show_roi(data, roi_phase, 'phase')

        conf = self.cal['shock']
        try:
            shocks = find_shocks(
                ref, conf['roi_ref'],
                data, conf['roi_phase'],
                time,
                debug=debug
            )
            if shocks is None:
                return []
            return shocks.tolist()
        except Exception as ex:
            print(ex)
            return []


class _KEPLER(_VISAR):
    SWEEP_SPEED = {1: 50, 2: 20, 3: 10, 4: 5, 5: 1, 6: 100}

    @_cache(name="Sweep time", py_type=True)
    def sweep_time(self):
        """Sweep window in nanosecond

        We assume the sweep time does not change over a run.

        Raise ValueError if the sweep speed changes over the run.
        """
        ss = self.ctrl["sweepSpeed"].as_single_value()
        st = self.SWEEP_SPEED[int(ss)]
        return xr.DataArray(st, attrs={"units": "ns"})

    @_cache(name="image")
    def image(self):
        data = self.detector.ndarray()
        data = np.rot90(data, 1, axes=(1, 2))
        data = np.array([np.fliplr(remap(frame, *self.cal.map())) for frame in data])
        return xr.DataArray(data, dims=["trainId", "dim_0", "dim_1"])


class _VISAR_1w(_VISAR):
    pass


def VISAR(run, name="KEPLER1", config_file=None):
    """Create a VISAR object based on the specified device name.

    Parameters:
        run (DataCollection): The run data collection object containing the data
            for the VISAR device. name (str): The name of the VISAR device to
            instantiate. Must be one of "KEPLER1", "KEPLER2", "VISAR_1w", "SOP".
            Default is "KEPLER1".
        config_file (str, optional): Path to the
            configuration file for the VISAR device. If provided, it will be
            used to initialize the CalibrationData.

    Returns:
        _StreakCamera: An instance of the appropriate VISAR class based on the
            device name.

    Raises:
        ValueError: If the provided name is not one of the valid VISAR device
            names.
    """
    if name == "SOP":
        _V = _StreakCamera
    elif name == "VISAR_1w":
        _V = _VISAR_1w
    elif name in ("KEPLER1", "KEPLER2"):
        _V = _KEPLER
    else:
        raise ValueError(f'name must be one of {", ".join(VISAR_DEVICES)}')
    return _V(run, name=name, config_file=config_file)


if __name__ == "__main__":
    from extra_data import open_run

    r = open_run(6656, 22)
    config = "/gpfs/exfel/data/user/tmichela/tmp/visar_calibration_values_6656.toml"

    for v in VISAR_DEVICES:
        vis = VISAR(r, name=v, config_file=config)
        vis.info()

        for train_id in vis.train_ids.value:
            vis.plot(train_id)
        vis.to_h5()
