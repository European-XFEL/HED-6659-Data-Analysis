from enum import Enum
from functools import cache, cached_property, lru_cache
from pathlib import Path

import cv2
import h5py
import numpy as np
import toml
from extra_data import by_id, KeyData
from pint import UnitRegistry
from pint.UnitRegistry import Quantity as PintQuantity
from scipy.interpolate import griddata

ureg = UnitRegistry()
ureg.setup_matplotlib(True)
Quantity = ureg.Quantity

VISAR_DEVICES = {
    'KEPLER1': {
        'arm': 'COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_1',
        'trigger': 'HED_EXP_VISAR/TSYS/ARM_1_TRIG',
        'detector': 'HED_SYDOR_TEST/CAM/KEPLER_1:daqOutput',
        'ctrl': 'HED_SYDOR_TEST/CTRL/CONTROL_UNIT_1',
    },
    'KEPLER2': {
        'arm': 'COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_2',
        'trigger': 'HED_EXP_VISAR/TSYS/ARM_2_TRIG',
        'detector': 'HED_SYDOR_TEST/CAM/KEPLER_2:daqOutput',
        'ctrl': 'HED_SYDOR_TEST/CTRL/CONTROL_UNIT_2',
    },
    'VISAR_1w': {
        'arm': 'COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_3',
        'trigger': 'HED_EXP_VISAR/TSYS/ARM_3_TRIG',
        'detector': 'HED_EXP_VISAR/EXP/ARM_3_STREAK:daqOutput',
        'ctrl': 'HED_EXP_VISAR/EXP/ARM_3_STREAK',
    },
    'SOP': {
        'trigger': 'HED_EXP_VISAR/TSYS/SOP_TRIG',
        'detector': 'HED_EXP_VISAR/EXP/SOP_STREAK:daqOutput',
        'ctrl': 'HED_EXP_VISAR/EXP/SOP_STREAK',
    },
}


class DipolePPU(Enum):
    OPEN = np.uint32(34)
    CLOSED = np.uint32(4130)


def remap(image, source, target):
    return cv2.remap(image, source, target, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


def resize(image, row_factor=2, column_factor=2):
    return cv2.resize(
        image,
        (image.shape[1] * row_factor, image.shape[0] * column_factor),
        interpolation=cv2.INTER_CUBIC
    )


def rotate(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def save_quantity(h5group, name, quantity):
    """Save a Pint quantity to an HDF5 group.

    Args:
        h5group: h5py.Group or h5py.File object to save to
        name: str, name of the dataset
        quantity: pint.Quantity object to save
    """
    qty_group = h5group.create_group(name)
    qty_group.create_dataset('magnitude', data=quantity.magnitude)
    qty_group.attrs['units'] = str(quantity.units)


def load_quantity(h5group, name):
    """Load a Pint quantity from an HDF5 group.

    Args:
        h5group: h5py.Group or h5py.File object to load from
        name: str, name of the dataset

    Returns:
        pint.Quantity: The loaded quantity with units
    """
    qty_group = h5group[name]
    magnitude = qty_group['magnitude'][()]
    units = qty_group.attrs['units']
    return Quantity(magnitude, units)


def as_single_value(kd: KeyData) -> PintQuantity:
    value = kd.as_single_value()
    if value.is_integer():
        value = int(value)
    return Quantity(value, kd.units)


def as_quantity(kd: KeyData, train_ids: int | list[int] = None):
    if train_ids is None:
        return as_single_value(kd)

    if isinstance(train_ids, int):
        train_ids = [train_ids]
    sel = kd[by_id[train_ids]]

    try:
        value = sel.as_single_value()
    except ValueError:
        # value is changing, get all individual values
        value = sel[by_id[train_ids]].ndarray()
    return Quantity(value, kd.units)


class DIPOLE:
    def __init__(self, run, name='DIPOLE'):
        self.name = name
        self.run = run

    def info(self):
        print(self.format())

    def format(self, compact=False):
        """Format information about the VISAR component.
        """
        meta = self.run.run_metadata()
        run_str = f'p{meta.get("proposalNumber", "?"):06}, r{meta.get("runNumber", "?"):04}'
        # info_str = f'{self.name} properties for {run_str}:\n'

        if compact or True:
            return f'{self.name}, {run_str}'

        # quantities = []
        # quantities.append(('Delay:', self.delay()))
        # quantities.append(('Energy:', self.energy()))
        # # quantities.append(('Trace:', self.trace()))

        # span = len(max(quantities)[0]) + 1
        # info_str += '\n'.join([f'  {name:<{span}}{value:~.7g}' for name, value in quantities])
        # return info_str

    def ppu(self):
        return self.run[
            'HED_HPLAS_HET/SWITCH/DIPOLE_PPU_OPEN', 'hardwareStatusBitField'
        ].xarray().where(lambda x: x == DipolePPU.OPEN.value, drop=True)

    def delay(self, train_ids: int | list[int] = None) -> PintQuantity:
        return as_quantity(self.run['APP_DIPOLE/MDL/DIPOLE_TIMING', 'actualPosition'], train_ids)

    def energy(self, train_ids: int | list[int] = None) -> PintQuantity:
        return as_quantity(self.run['APP_DIPOLE/MDL/DIPOLE_DIAGNOSTIC', 'energy2W'], train_ids)

    def trace(self, train_ids: int | list[int] = None, dt: float = 0.2) -> PintQuantity:
        """
        dt: float [ns/sample]
        """
        # TODO fix that function
        if train_ids is None:
            train_ids = self.run.train_ids
        elif isinstance(train_ids, int):
            train_ids = [train_ids]

        traces = self.run['HED_PLAYGROUND/SCOPE/TEXTRONIX_TEST:output', 'ch1.corrected']
        traces = traces[by_id[train_ids]].ndarray()

        vmax = np.unique(traces[:, :40000], axis=1).max(axis=1)

        time_axis = []
        power_trace = []
        for trace, trace_max in zip(traces, vmax):
            idx = np.where(trace > trace_max)[0]
            time_axis.append((np.arange(idx[0] - 25, idx[-1] + 25) - idx[0]) * dt)

            dipole_duration = (idx[-1] - idx[0]) * dt * 1e-9
            energy = self.energy(train_ids)
            power_scaling = energy / (trace[idx[0]:idx[-1]].sum() * dipole_duration)
            power_trace.append(trace[idx[0]-25:idx[-1]+25] * power_scaling)
        return time_axis, power_trace


class CalibrationData:
    def __init__(self, visar, file_path=None):
        self.visar = visar

        if file_path is not None:
            config = toml.load(file_path)
            self.config = config[self.visar.name]
            self.config.update(config['global'])
        else:
            self.config = {}

    def __getitem__(self, key):
        return self.config[key]

    @cached_property
    def dx(self):
        """Length per pixel in µm
        """
        return Quantity(self['dx'], 'um')

    @cached_property
    def dipole_zero(self):
        """Dipole position at 0 ns delay, 0 ns sweep delay
        """
        pixel_offset = self['pixDipole_0ns'][f'{self.visar.sweep_time.m}ns']
        return self.timepoint(pixel_offset)

    @cached_property
    def fel_zero(self):
        """Xray position at 0 ns delay, 0 ns sweep delay
        """
        return self.timepoint(self['pixXray'])

    @cached_property
    def timepoint(self):
        """Compute Time from pixel position in ns
        """
        constants = self['timeAxisPolynomial'][f'{self.visar.sweep_time.m}ns']
        # Pad with leading 0 because there is no intercept for the time axis
        poly = np.poly1d(np.array([0, *constants])[::-1])
        def _timepoint(data):
            return Quantity(poly(data), 'ns')
        return _timepoint

    @cached_property
    def reference_trigger_delay(self):
        ref = self['positionTrigger_ref'][f'{self.visar.sweep_time.m}ns']
        return Quantity(ref, 'ns')

    @cache
    def map(self) -> tuple[np.ndarray, np.ndarray]:
        """Return input and output transformation maps
        """
        tr_map_file = self['transformationMaps'][f'{self.visar.sweep_time.m}ns']
        file_path = Path(self['dirTransformationMaps']) / tr_map_file
        coords = np.loadtxt(file_path, delimiter=',')
        target = coords[..., 2:]
        source = coords[..., :2]

        y, x = self.visar.pixels.entry_shape
        grid_1, grid_2 = np.mgrid[:y, :x]
        grid_z = griddata(target, source, (grid_1, grid_2), method='linear')
        map_1 = grid_z[..., 1].astype(np.float32)
        map_2 = grid_z[..., 0].astype(np.float32)

        return map_1, map_2


class _StreakCamera:
    def __init__(self, run, name='KEPLER1', dipole=None, config_file=None):
        self.name = name
        visar = VISAR_DEVICES[name]

        self.run = run
        if 'arm' in visar:
            self.arm = run[visar['arm']]
        self.trigger = run[visar['trigger']]
        self.detector = run[visar['detector']]
        self.ctrl = run[visar['ctrl']]

        self.dipole = dipole or DIPOLE(run)
        self.cal = CalibrationData(self, config_file)

    def __repr__(self):
        return f'<{type(self).__name__} {self.name}>'

    def _quantity(self, kd: KeyData, train_ids=None) -> PintQuantity:
        """Get values for shot train IDs
        
        If the quantity doesn't change over trains, this returns a single value,
        else returns one value per train.
        """
        return as_quantity(kd, train_ids or self.shots())

    def as_dict(self):
        quantities = {}
        for attr in sorted(dir(self)):
            if attr.startswith('_'):
                continue
            q = getattr(self, attr)
            if isinstance(q, Quantity):
                quantities[attr] = q
        quantities['train_ID_(shots)'] = self.shots()
        quantities['train_ID_(ref.)'] = self.shots(reference=True)
        return quantities

    def info(self):
        """Print information about the VISAR component
        """
        print(self.format())

    def format(self, compact=False):
        """Format information about the VISAR component.
        """
        meta = self.run.run_metadata()
        run_str = f'p{meta.get("proposalNumber", "?"):06}, r{meta.get("runNumber", "?"):04}'
        info_str = f'{self.name} properties for {run_str}:\n'

        if compact:
            return f'{self.name}, {run_str}'

        quantities = []
        for attr in sorted(dir(self)):
            if attr.startswith('_'):
                continue
            q = getattr(self, attr)
            if isinstance(q, Quantity):
                quantities.append((f'{attr.replace("_", " ").capitalize()}:', q))

        span = len(max(quantities)[0]) + 1
        info_str += '\n'.join([f'  {name:<{span}}{value:~.7g}' for name, value in quantities])
        info_str += f'\n\n  Train ID (shots): {self.shots()}'
        info_str += f'\n  Train ID (ref.): {self.shots(reference=True)}'
        return info_str

    @cached_property
    def zero_delay_position(self) -> PintQuantity:
        return self._quantity(self.arm['zeroDelayPosition'])

    @cache
    def sweep_delay(self, train_id=None) -> PintQuantity:
        for key in ['actualDelay', 'actualPosition']:
            if key in self.trigger:
                return self._quantity(self.trigger[key], train_id) - self.cal.reference_trigger_delay

    @cached_property
    def sweep_time(self):
        return Quantity(self.ctrl.run_value('timeRange'))

    @property
    def pixels(self):
        return self.detector['data.image.pixels']

    @lru_cache()
    def shots(self, reference=False):
        """Get train IDs of data with open PPU.

        If reference is True, return the first data with closed PPU instead.
        """
        # train ID with data in the run
        tids = self.pixels.drop_empty_trains().train_id_coordinates()
        if tids.size == 0:
            return  # there's not data in this run

        shot_ids = np.intersect1d(self.dipole.ppu().trainId, tids)

        if reference:
            # return the first data point with closed ppu
            for tid in tids:
                if tid not in shot_ids:
                    return np.array([tid], dtype=int)
            else:
                return  # no data with closed ppu

        if shot_ids.size == 0:
            return  # no data with open ppu in this run

        return shot_ids.astype(int)

    @cache
    def data(self, reference=False):
        """Get corrected data
        
        Returns the dewarped data for the trains with PPU opened. If *reference*
        is True, returns the first frame with PPU closed instead.
        """
        raise NotImplementedError
    
    @cache
    def fel_delay(self, train_id: int):
        return self.cal.fel_zero - self.cal.dipole_zero - self.dipole.delay(train_id) - self.sweep_delay(train_id)

    @cache
    def _time_axis(self, train_id: int):
        axis = self.cal.timepoint(np.arange(self.data().shape[-1]))
        offset = self.cal.dipole_zero + self.dipole.delay(train_id) # - self.sweep_delay(train_id)
        return axis - offset

    @cache
    def _space_axis(self):
        axis = np.arange(self.data().shape[-2]) * self.cal.dx
        return axis - axis.mean()

    def plot(self, train_id, ax=None):
        shot_index = self.shots().tolist().index(train_id)
        data = self.data()[shot_index]

        time_axis = self._time_axis(train_id)
        space_axis = self._space_axis()

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5))

        tid_str = f'SHOT ({shot_index + 1}/{len(self.shots())}), tid:{train_id}'
        ax.set_title(f'{self.format(compact=True)}, {tid_str}')
        ax.set_xlabel(f'Distance [{space_axis.u:~}]')
        ax.set_ylabel(f'Time [{time_axis.u:~}]')

        extent = [time_axis.m[0], time_axis.m[-1], space_axis.m[0], space_axis.m[-1]]
        im = ax.imshow(data, extent=extent, cmap='jet', vmin=0, vmax=data.mean()+3*data.std())
        ax.vlines(
            self.fel_delay(train_id),
            ymin=space_axis.m[0],
            ymax=space_axis.m[-1],
            linestyles='-',
            lw=2,
            color='purple',
            alpha=1,
        )

        ys, xs = np.where(data > 0)
        ax.set_xlim(xmin=time_axis.m[xs.min()], xmax=time_axis.m[xs.max()])
        ax.set_ylim(ymin=-space_axis.m[ys.max()], ymax=-space_axis.m[ys.min()])
        ax.set_aspect('auto')

        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(400))
        ax.yaxis.set_minor_locator(MultipleLocator(100))
        ax.grid(which='major', color='k', linestyle = '--', linewidth=2, alpha = 0.5)
        ax.grid(which='minor', color='k', linestyle=':', linewidth=1, alpha = 1)

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        return ax

    def save(self, output='.', filename='VISAR_p{proposal:06}_r{run:04}.h5'):
        """Save the data for this VISAR in HDF5

        output: str
            The output directory to write the file
        """
        meta = self.run.run_metadata()
        proposal = meta.get('proposalNumber', '')
        run = meta.get('runNumber', '')
        fpath = f'{output}/{filename.format(proposal=proposal, run=run)}'
        with h5py.File(fpath, 'a') as fh:
            if self.name not in fh:
                visar = fh.create_group(self.name)

                ref = visar.create_group('Reference')
                ref['Reference'] = self.data(reference=True)
                ref['train ID'] = self.shots(reference=True)

                shots = visar.create_group('Shots')
                shots['Corrected images'] = self.data()
                save_quantity(shots, "Time axis",
                    np.vstack([self._time_axis(tid) for tid in self.shots()]))
                save_quantity(shots, "Space axis",
                    np.tile(self._space_axis(), (self.shots().size, 1)))
                save_quantity(shots, "Drive pixel t0", self.cal.dipole_zero)
                save_quantity(shots, "Sensitivity", self.sensitivity)
                save_quantity(shots, "Etalon thickness", self.etalon_thickness)
                save_quantity(shots, "Etalon delay", self.temporal_delay)
                save_quantity(shots, "Sweep window", self.sweep_time)
                save_quantity(shots, "Sweep delay", self.sweep_delay())
                save_quantity(shots, "Difference X-drive", self.fel_delay)
                save_quantity(shots, "Train ID", self.shots())

            if self.dipole.name not in fh:
                dipole = fh.create_group(self.dipole.name)
                save_quantity(dipole, "Energy", self.dipole.energy(self.shots()))
                save_quantity(dipole, "Delay", self.dipole.delay(self.shot()))
                # save_quantity(dipole, "profile", self.dipole.profile(self.shots()))  # TODO
                # save_quantity(dipole, "profile time axis", self.dipole.profile_axis(self.shots()))  # TODO


class _VISAR(_StreakCamera):
    @cached_property
    def etalon_thickness(self) -> PintQuantity:
        return self._quantity(self.arm['etalonThickness'])

    @cached_property
    def motor_displacement(self) -> PintQuantity:
        return self._quantity(self.arm['motorDisplacement'])

    @cached_property
    def sensitivity(self) -> PintQuantity:
        return self._quantity(self.arm['sensitivity'])

    @cached_property
    def temporal_delay(self) -> PintQuantity:
        return self._quantity(self.arm['temporalDelay'])


class _KEPLER(_VISAR):
    SWEEP_SPEED = {1: 50, 2: 20, 3: 10, 4: 5, 5: 1, 6: 100}

    @cached_property
    def sweep_time(self):
        """Sweep window

        Raise ValueError if the sweep speed changes over the run.
        """
        ss = as_single_value(self.ctrl['sweepSpeed'])
        return Quantity(self.SWEEP_SPEED[ss], 'ns')

    @lru_cache()
    def data(self, reference=False):
        if (tid := self.shots(reference=reference)) is None:
            return
        frames = self.pixels[by_id[tid]].ndarray()

        corrected = np.array([
            np.fliplr(remap(rotate(frame), *self.cal.map()))
            for frame in frames
        ])
        return corrected


class _VISAR_1w(_VISAR):
    @lru_cache()
    def data(self, reference=False):
        # 1w VISAR is 2x2 binned
        # need to upscale it full size to use the time calibration
        if (tid := self.shots(reference=reference)) is None:
            return
        frames = self.pixels[by_id[tid]].ndarray()
        corrected = np.array([rotate(resize(frame)) for frame in frames])
        return corrected


def VISAR(run, name='KEPLER1', dipole=None, config_file=None):
    if name == 'SOP':
        _V = _StreakCamera
    elif name == 'VISAR_1w':
        _V = _VISAR_1w
    elif name in ('KEPLER1', 'KEPLER2'):
        _V = _KEPLER
    else:
        raise ValueError(f'name must be one of {", ".join(VISAR_DEVICES)}')
    return _V(run, name=name, dipole=dipole, config_file=config_file)


if __name__ == '__main__':
    from extra_data import open_run

    r = open_run(6656, 22)
    config = '/gpfs/exfel/data/user/tmichela/tmp/visar_calibration_values_6656.toml'

    dipole = DIPOLE(r)

    for v in VISAR_DEVICES:
        vis = VISAR(r, name=v, dipole=dipole, config_file=config)
        vis.info()

        for train_id in vis.shots():
            vis.plot(train_id)
        vis.save()
