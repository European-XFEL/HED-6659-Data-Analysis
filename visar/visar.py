from bisect import insort
from enum import Enum
from functools import cache, cached_property, lru_cache
from inspect import signature
from pathlib import Path

import cv2
import numpy as np
import toml
import xarray as xr
from extra_data import DataCollection, KeyData, by_id
from scipy.interpolate import griddata

VISAR_DEVICES = {
    'KEPLER1': {
        'arm': 'COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_1',
        'trigger': 'HED_EXP_VISAR/TSYS/ARM_1_TRIG',
        'detector': ('HED_SYDOR_TEST/CAM/KEPLER_1:daqOutput', 'data.image.pixels'),
        'ctrl': 'HED_SYDOR_TEST/CTRL/CONTROL_UNIT_1',
    },
    'KEPLER2': {
        'arm': 'COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_2',
        'trigger': 'HED_EXP_VISAR/TSYS/ARM_2_TRIG',
        'detector': ('HED_SYDOR_TEST/CAM/KEPLER_2:daqOutput', 'data.image.pixels'),
        'ctrl': 'HED_SYDOR_TEST/CTRL/CONTROL_UNIT_2',
    },
    'VISAR_1w': {
        'arm': 'COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_3',
        'trigger': 'HED_EXP_VISAR/TSYS/ARM_3_TRIG',
        'detector': ('HED_EXP_VISAR/EXP/ARM_3_STREAK:daqOutput', 'data.image.pixels'),
        'ctrl': 'HED_EXP_VISAR/EXP/ARM_3_STREAK',
    },
    'SOP': {
        'trigger': 'HED_EXP_VISAR/TSYS/SOP_TRIG',
        'detector': ('HED_EXP_VISAR/EXP/SOP_STREAK:daqOutput', 'data.image.pixels'),
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


def dipole_ppu_open(run: DataCollection):
    """ Get trainIds in run with dipole PPU open """
    return run[
        'HED_HPLAS_HET/SWITCH/DIPOLE_PPU_OPEN', 'hardwareStatusBitField'
    ].xarray().where(lambda x: x == DipolePPU.OPEN.value, drop=True).trainId


class DIPOLE:
    def __init__(self, visar, run, name='DIPOLE'):
        self.visar = visar
        self.name = name
        self.run = run

    def info(self):
        print(self.format())

    def format(self, compact=False):
        """Format information about the VISAR component.
        """
        meta = self.run.run_metadata()
        run_str = f'p{meta.get("proposalNumber", "?"):06}, r{meta.get("runNumber", "?"):04}'

        if compact:
            return f'{self.name}, {run_str}'
        
        info_str = (
            f'{self.name} properties for {run_str}:\n'
            # TODO
        )
        return info_str
        # quantities = []
        # quantities.append(('Delay:', self.delay()))
        # quantities.append(('Energy:', self.energy()))
        # # quantities.append(('Trace:', self.trace()))

        # span = len(max(quantities)[0]) + 1
        # info_str += '\n'.join([f'  {name:<{span}}{value:~.7g}' for name, value in quantities])
        # return info_str

    def save(self, file_path):
        self.dataset().to_netcdf(
            path=file_path, 
            mode='a',
            group=f'{self.visar.name}/dipole',
            format='NETCDF4',
            engine='h5netcdf'
        )

    @cache
    def dataset(self):
        return xr.Dataset(
            data_vars={
                "Delay": self.delay(),
                "Energy": self.energy(),
                #TODO add trace information
                # "profile", self.dipole.profile(self.shots()))  # TODO
                # "profile time axis", self.dipole.profile_axis(self.shots()))  # TODO
            }
        )

    def delay(self):
        delay = self.run['APP_DIPOLE/MDL/DIPOLE_TIMING', 'actualPosition']
        data = delay.xarray()
        data.attrs['units'] = delay.units
        return data

    def energy(self):
        energy = self.run['APP_DIPOLE/MDL/DIPOLE_DIAGNOSTIC', 'energy2W']
        data = energy.xarray()
        data.attrs['units'] = energy.units
        return data

    def trace(self, dt: float = 0.2):
        """
        dt: float [ns/sample]
        """
        # TODO fix that function

        traces = self.run['HED_PLAYGROUND/SCOPE/TEXTRONIX_TEST:output', 'ch1.corrected'].ndarray()

        vmax = np.unique(traces[:, :40000], axis=1).max(axis=1)

        time_axis = []
        power_trace = []
        for trace, trace_max in zip(traces, vmax):
            idx = np.where(trace > trace_max)[0]
            time_axis.append((np.arange(idx[0] - 25, idx[-1] + 25) - idx[0]) * dt)

            dipole_duration = (idx[-1] - idx[0]) * dt * 1e-9
            energy = self.energy()
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

    @property
    def dx(self):
        """Length per pixel in µm
        """
        return self['dx']

    @property
    def dipole_zero(self):
        """Dipole position at 0 ns delay, 0 ns sweep delay
        """
        pixel_offset = self['pixDipole_0ns'][f'{self.visar.sweep_time}ns']
        return self.timepoint(pixel_offset)

    @property
    def fel_zero(self):
        """Xray position at 0 ns delay, 0 ns sweep delay
        """
        return self.timepoint(self['pixXray'])

    @property
    def reference_trigger_delay(self):
        return self['positionTrigger_ref'][f'{self.visar.sweep_time}ns']

    @cached_property
    def timepoint(self):
        """Compute Time from pixel position in ns
        """
        constants = self['timeAxisPolynomial'][f'{self.visar.sweep_time}ns']
        # Pad with leading 0 because there is no intercept for the time axis
        return np.poly1d(np.array([0, *constants])[::-1])

    @cache
    def map(self) -> tuple[np.ndarray, np.ndarray]:
        """Return input and output transformation maps
        """
        tr_map_file = self['transformationMaps'][f'{self.visar.sweep_time}ns']
        file_path = Path(self['dirTransformationMaps']) / tr_map_file
        coords = np.loadtxt(file_path, delimiter=',')
        target = coords[..., 2:]
        source = coords[..., :2]

        y, x = self.visar.detector.entry_shape
        grid_1, grid_2 = np.mgrid[:y, :x]
        grid_z = griddata(target, source, (grid_1, grid_2), method='linear')
        map_1 = grid_z[..., 1].astype(np.float32)
        map_2 = grid_z[..., 0].astype(np.float32)

        return map_1, map_2
    
    @cache
    def dataset(self):
        return xr.Dataset(
            data_vars={
                "dx": xr.DataArray(self.dx, attrs={'units': 'um'}),
                "Drive pixel t0": xr.DataArray(self.dipole_zero, attrs={'units': 'ns'}),
                "FEL zero": xr.DataArray(self.fel_zero, attrs={'units': 'ns'}),
                "Reference trigger delay": xr.DataArray(self.reference_trigger_delay, attrs={'units': 'ns'}),
                "Dewarp source": (["dim_0", "dim_1"], self.map()[0]),
                "Dewarp target": (["dim_0", "dim_1"], self.map()[1]),
            }
        )

    def save(self, file_path):
        self.dataset().to_netcdf(
            path=file_path, 
            mode='a',
            group=f'{self.visar.name}/calibration',
            format='NETCDF4',
            engine='h5netcdf'
        )


class _StreakCamera:
    def __init__(self, run, name, config_file=None):
        self.run = run
        self.name = name
        self.visar = VISAR_DEVICES[name]

        sel = run.select_trains(self.train_ids)

        if 'arm' in self.visar:
            self.arm = sel[self.visar['arm']]
        self.trigger = sel[self.visar['trigger']]
        self.detector = sel[self.visar['detector']]
        self.ctrl = sel[self.visar['ctrl']]

        self.dipole = DIPOLE(self, sel)
        self.cal = CalibrationData(self, config_file)

        self.dataset = xr.Dataset(coords=self.coords)

    def process(self):
        import time

        t0 = time.perf_counter()
        self.sweep_time
        print(f'sweep time: {round((time.perf_counter()-t0)*1000, 3)}ms')

        for name in dir(self):
            if name.startswith('__'):
                continue
            
            t0 = time.perf_counter()
            attr = getattr(self, name)
            t1 = time.perf_counter()
            if not callable(attr):
                continue

            if '_name' in signature(attr).parameters:
                attr()
                t2 = time.perf_counter()
                print(f'{name}: getattr:{round((t1-t0)*1000, 3)}ms - execute:{round((t2-t1)*1000, 3)}ms')
            else:
                print(f'{name}: getattr:{round((t1-t0)*1000, 3)}ms')

        # calibration data
        self.cal.dataset()

        # add dipole data
        self.dipole.dataset()

    def __repr__(self):
        return f'<{type(self).__name__} {self.name}>'

    def _data(self, name: str, kd: KeyData) -> xr.DataArray:
        if name not in self.dataset:
            data = kd.xarray()
            data.attrs['units'] = kd.units
            self.dataset[name] = data
        return self.dataset[name]

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

        data = []
        for attr in dir(self):
            if attr.startswith('_'):
                continue
            if attr in ['image']:
                continue

            v = getattr(self, attr)
            if not callable(v):
                continue

            args =  signature(v).parameters
            if '_name' not in args:
                continue

            value = v()
            units = value.attrs.get('units', '')
            if len(np.unique(value)) == 1:
                value = f'{value.data[0]:.6g}'
            else:
                value = ', '.join(f'{v:.6g}' for v in value)

            data.append((f'{args["_name"].default}:', value, units))
        data.append(("Sweep time:", f'{self.sweep_time:.6g}', 'ns'))

        span = max(len(e[0]) for e in data) + 1
        # TODO format multiple values
        info_str += '\n'.join([f'  {name:<{span}}{value} {units}' for name, value, units in sorted(data)])
        info_str += f'\n\n  Train ID (shots): {self.coords.where(self.coords.type=="shot", drop=True).trainId.data}'
        info_str += f'\n  Train ID (ref.): {self.coords.where(self.coords.type=="reference", drop=True).trainId.data}'
        return info_str

    def fel_delay(self, _name='Difference X-drive'):
        if _name not in self.dataset:
            fel_delay = self.cal.fel_zero - self.cal.dipole_zero - self.dipole.delay() - self.sweep_delay()
            self.dataset[_name] = xr.DataArray(fel_delay, dims=['trainId'], attrs={'units': 'ns'})
        return self.dataset[_name]

    def sweep_delay(self, _name="Sweep delay"):
        if _name not in self.dataset:

            for key in ['actualDelay', 'actualPosition']:
                if key in self.trigger:
                    break
            else:
                raise KeyError(f'sweep delay property not found in {self.trigger}')
            
            data = self.trigger[key].xarray()
            data.attrs['units'] = self.trigger[key].units
            self.dataset[_name] = data - self.cal.reference_trigger_delay

        return self.dataset[_name]

    @property
    def sweep_time(self, _name="Sweep time"):
        """Sweep window in nanosecond

        We assume the sweep time does not change over a run
        """
        if _name not in self.dataset:
            sw, units = self.ctrl.run_value('timeRange').split(' ')
            data = xr.DataArray(int(sw), attrs={'units': units})
            self.dataset[_name] = data
        return self.dataset[_name].data.tolist()

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
        tids = self.run[self.visar['detector']].drop_empty_trains().train_id_coordinates()
        ppu_open = dipole_ppu_open(self.run)

        # train ID with data and ppu open
        shot_ids = np.intersect1d(ppu_open, tids)
        # train IDs with data and ppu closed
        ref_ids = np.setdiff1d(tids, ppu_open)

        train_ids = shot_ids.tolist()
        types = ['shot'] * len(train_ids)

        if ref_ids.size > 0:
            insort(train_ids, ref_ids[0])
            types.insert(train_ids.index(ref_ids[0]), 'reference')

        return xr.Dataset(
            coords={
                'trainId': np.array(train_ids, dtype=np.uint64),
                'type': ('trainId', types),
            })

    @cache
    def image(self, _name='image'):
        """Get corrected images

        Returns corrected data for the trains with PPU opened. If *reference*
        is True, returns the first frame with PPU closed instead.
        """
        if _name not in self.dataset:
            data = self.detector.ndarray()
            # SOP and 1w VISAR is 2x2 binned
            # need to upscale it full size to use the time calibration
            data = np.array([resize(frame) for frame in data])
            data = np.rot90(data, 1, axes=(1, 2))
            self.dataset[_name] = xr.DataArray(data, dims=['trainId', 'dim_0', 'dim_1'])
        return self.dataset[_name]

    def _time_axis(self, _name='Time axis'):
        if _name not in self.dataset:
            axis = self.cal.timepoint(np.arange(self.image().shape[-1]))
            offset = self.cal.dipole_zero + self.dipole.delay() - self.sweep_delay()

            data = np.repeat(axis[None, ...], self.coords.trainId.size, axis=0) - offset.data[:, None]
            self.dataset[_name] = xr.DataArray(data, dims=['trainId', 'Time'], attrs={'units': 'ns'})
        return self.dataset[_name]

    def _space_axis(self, _name="Space axis"):
        if _name not in self.dataset:
            axis = np.arange(self.image().shape[-2]) * self.cal.dx
            axis = axis - axis.mean()

            data = np.repeat(axis[None, ...], self.coords.trainId.size, axis=0)
            self.dataset[_name] = xr.DataArray(data, dims=['trainId', 'Space'], attrs={'units': 'um'})
        return self.dataset[_name]

    def plot(self, train_id, ax=None):
        ds = self.dataset.sel(trainId=train_id)
        data = ds.image
        type_ = ds.type.data.tolist()

        time_axis = ds['Time axis']
        space_axis = ds['Space axis']

        # shot or reference index in the run
        frames = self.dataset.where(self.dataset.type==type_, drop=True)
        frame_index = frames.trainId.data.tolist().index(train_id)

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5))

        tid_str = f'{type_} ({frame_index + 1}/{frames.trainId.size}), tid:{train_id}'
        ax.set_title(f'{self.format(compact=True)}, {tid_str}')
        ax.set_xlabel(f'Distance [{space_axis.attrs.get("units", "?")}]')
        ax.set_ylabel(f'Time [{time_axis.attrs.get("units", "?")}]')

        extent = [time_axis[0], time_axis[-1], space_axis[0], space_axis[-1]]
        im = ax.imshow(data, extent=extent, cmap='jet', vmin=0, vmax=data.mean()+3*data.std())
        ax.vlines(
            ds['Difference X-drive'],  # fel_delay
            ymin=space_axis[0],
            ymax=space_axis[-1],
            linestyles='-',
            lw=2,
            color='purple',
            alpha=1,
        )

        ys, xs = np.where(data > 0)
        ax.set_xlim(xmin=time_axis[xs.min()], xmax=time_axis[xs.max()])
        ax.set_ylim(ymin=-space_axis[ys.max()], ymax=-space_axis[ys.min()])
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
        self.process()
        meta = self.run.run_metadata()
        proposal = meta.get('proposalNumber', '')
        run = meta.get('runNumber', '')
        fpath = f'{output}/{filename.format(proposal=proposal, run=run)}'

        self.dataset.to_netcdf(path=fpath, mode='a', group=self.name, format='NETCDF4', engine='h5netcdf')
        self.cal.save(fpath)
        self.dipole.save(fpath)


class _VISAR(_StreakCamera):
    def zero_delay_position(self, _name="Zero delay"):
        return self._data(_name, self.arm['zeroDelayPosition'])

    def etalon_thickness(self, _name="Etalon thickness"):
        return self._data(_name, self.arm['etalonThickness'])

    def motor_displacement(self, _name="Motor displacement"):
        return self._data(_name, self.arm['motorDisplacement'])

    def sensitivity(self, _name="Sensitivity"):
        return self._data(_name, self.arm['sensitivity'])

    def temporal_delay(self, _name="Temporal delay"):
        return self._data(_name, self.arm['temporalDelay'])


class _KEPLER(_VISAR):
    SWEEP_SPEED = {1: 50, 2: 20, 3: 10, 4: 5, 5: 1, 6: 100}

    @property
    def sweep_time(self, _name='Sweep time'):
        """Sweep window in nanosecond

        We assume the sweep time does not change over a run.

        Raise ValueError if the sweep speed changes over the run.
        """
        if _name not in self.dataset:
            ss = self.ctrl['sweepSpeed'].as_single_value()
            st = self.SWEEP_SPEED[int(ss)]
            self.dataset[_name] = xr.DataArray(st, attrs={'units': 'ns'})
        return self.dataset[_name].data.tolist()

    @cache
    def image(self, _name='image'):
        if _name not in self.dataset:
            data = self.detector.ndarray()
            data = np.rot90(data, 1, axes=(1, 2))
            data = np.array([
                np.fliplr(remap(frame, *self.cal.map()))
                for frame in data
            ])
            self.dataset[_name] = xr.DataArray(data, dims=['trainId', 'dim_0', 'dim_1'])
        return self.dataset[_name]


class _VISAR_1w(_VISAR):
    pass


def VISAR(run, name='KEPLER1', config_file=None):
    if name == 'SOP':
        _V = _StreakCamera
    elif name == 'VISAR_1w':
        _V = _VISAR_1w
    elif name in ('KEPLER1', 'KEPLER2'):
        _V = _KEPLER
    else:
        raise ValueError(f'name must be one of {", ".join(VISAR_DEVICES)}')
    return _V(run, name=name, config_file=config_file)


if __name__ == '__main__':
    from extra_data import open_run

    r = open_run(6656, 22)
    config = '/gpfs/exfel/data/user/tmichela/tmp/visar_calibration_values_6656.toml'

    for v in VISAR_DEVICES:
        vis = VISAR(r, name=v, config_file=config)
        vis.info()

        for train_id in vis.shots():
            vis.plot(train_id)
        vis.save()
