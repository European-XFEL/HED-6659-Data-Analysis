from enum import Enum
from functools import cached_property, lru_cache
from pathlib import Path

import cv2
import numpy as np
import toml
from extra_data import by_id
from pint import UnitRegistry
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
    }
}


class DipolePPU(Enum):
    OPEN = np.uint32(34)
    CLOSED = np.uint(4130)


def remap(image, source, target):
    return cv2.remap(image, source, target, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


class CalibrationData:
    def __init__(self, visar, file_path=None):
        self.visar = visar

        if file_path is not None:
            config = toml.load(file_path)
            self.config = config[self.visar.name]
            self.config.update(config['global'])
        else:
            self.config = {}

    @property
    def dx(self):
        """Length per pixel in µm
        """
        return Quantity(self.config['dx'], 'um')

    @property
    def dipole_zero(self):
        """Dipole position at 0 ns delay, 0 ns sweep delay
        """
        pixel_offset = self.config['pixDipole_0ns'][f'{self.visar.sweep_time.m}ns']
        return pixel_offset * self.ds

    @property
    def fel_zero(self):
        """Xray position at 0 ns delay, 0 ns sweep delay
        """
        return self.config['pixXray'] * self.ds

    @property
    def ds(self):
        """Time per pixel in ns
        """
        sweep_time = self.config['sweepTime'][f'{self.visar.sweep_time.m}ns']
        return Quantity(sweep_time, 'ns')

    @property
    def reference_trigger_delay(self):
        ref = self.config['positionTrigger_ref'][f'{self.visar.sweep_time.m}ns']
        return Quantity(ref, 'ns')

    def map(self) -> tuple[np.ndarray, np.ndarray]:
        """Return input and output transformation maps
        """
        tr_map_file = self.config['transformationMaps'][f'{self.visar.sweep_time.m}ns']
        file_path = Path(self.config['dirTransformationMaps']) / tr_map_file
        coords = np.loadtxt(file_path, delimiter=',')
        target = coords[..., 2:]
        source = coords[..., :2]

        y, x = self.visar.data.entry_shape
        grid_1, grid_2 = np.mgrid[:y, :x]
        grid_z = griddata(target, source, (grid_1, grid_2), method='linear')
        map_1 = grid_z[..., 1].astype(np.float32)
        map_2 = grid_z[..., 0].astype(np.float32)

        return map_1, map_2


class VISAR:

    SWEEP_SPEED = {1: 50, 2: 20, 3: 10, 4: 5, 5: 1, 6: 100}

    def __init__(self, run, name='KEPLER1', cal_file=None):
        self.name = name
        visar = VISAR_DEVICES[name]

        self.run = run
        self.arm = run[visar['arm']]
        self.trigger = run[visar['trigger']]
        self.detector = run[visar['detector']]
        self.ctrl = run[visar['ctrl']]

        self.cal = CalibrationData(self, cal_file)

    def __repr__(self):
        return f'<{type(self).__name__} {self.name}>'

    def _as_single_value(self, kd):
        value = kd.as_single_value()
        if value.is_integer():
            value = int(value)
        return Quantity(value, kd.units)

    def as_dict(self):
        ...

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
        info_str += '\n'.join([f'  {name:<{span}}{value:#~.7g}' for name, value in quantities])
        info_str += f'\n\n  Train ID (shot): {self.shot()}'
        info_str += f'\n  Train ID (ref.): {self.shot(reference=True)}'
        return info_str

    @cached_property
    def etalon_thickness(self):
        return self._as_single_value(self.arm['etalonThickness'])

    @cached_property
    def motor_displacement(self):
        return self._as_single_value(self.arm['motorDisplacement'])

    @cached_property
    def sensitivity(self):
        return self._as_single_value(self.arm['sensitivity'])

    @cached_property
    def temporal_delay(self):
        return self._as_single_value(self.arm['temporalDelay'])

    @cached_property
    def zero_delay_position(self):
        return self._as_single_value(self.arm['zeroDelayPosition'])

    @cached_property
    def sweep_delay(self):
        for key in ['actualDelay', 'actualPosition']:
            if key in self.trigger:
                return self._as_single_value(self.trigger[key]) - self.cal.reference_trigger_delay

    @cached_property
    def sweep_time(self):
        ss = self._as_single_value(self.ctrl['sweepSpeed'])
        return Quantity(self.SWEEP_SPEED[ss], 'us')

    @cached_property
    def dipole_energy(self):
        energy = self.run['APP_DIPOLE/MDL/DIPOLE_DIAGNOSTIC', 'energy2W']
        return self._as_single_value(energy[by_id[[self.shot()]]])

    @cached_property
    def dipole_delay(self):
        delay = self.run['APP_DIPOLE/MDL/DIPOLE_TIMING', 'actualPosition']
        return self._as_single_value(delay[by_id[[self.shot()]]])

    @property
    def data(self):
        return self.detector['data.image.pixels']

    @lru_cache()
    def shot(self, reference=False):
        """Get train ID of data with open PPU.

        If reference is True, return the first data with closed PPU instead.
        """
        # train ID with data in the run
        tids = self.data.drop_empty_trains().train_id_coordinates()
        if tids.size == 0:
            return  # there's not data in this run

        ppu_open = self.run[
            'HED_HPLAS_HET/SWITCH/DIPOLE_PPU_OPEN', 'hardwareStatusBitField'
        ].xarray().where(lambda x: x == DipolePPU.OPEN.value, drop=True)

        shot_ids = np.intersect1d(ppu_open.trainId, tids)

        if reference:
            # return the first data point with closed ppu
            for tid in tids:
                if tid not in shot_ids:
                    return int(tid)
            else:
                return  # no data with closed ppu

        if shot_ids.size == 0:
            return  # no data with open ppu in this run

        # TODO could there be multiple shot in a run?
        return int(shot_ids[0])

    @lru_cache()
    def frame(self, reference=False):
        if (tid := self.shot(reference=reference)) is None:
            return
        frame = self.data[by_id[[tid]]].ndarray().squeeze()
        frame = np.rot90(frame, -1)

        source, target = self.cal.map()
        corrected = remap(frame, source, target)
        return np.flipud(corrected)

    def plot(self, reference=False, ax=None):
        data = self.frame(reference=reference)

        time_axis = np.arange(data.shape[0]) * self.cal.ds
        offset = self.cal.dipole_zero + self.dipole_delay - self.sweep_delay
        time_axis -= offset

        space_axis = np.arange(data.shape[1]) * self.cal.dx
        space_axis -= space_axis.mean()

        fig = None
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5))

        tid_str = f'{"REF." if reference else "SHOT"}, tid:{self.shot(reference=reference)}'
        ax.set_title(f'{self.format(compact=True)}, {tid_str}')
        ax.set_xlabel(f'Distance [{space_axis.u:~}]')
        ax.set_ylabel(f'Time [{time_axis.u:~}]')

        extent = [time_axis.m[0], time_axis.m[-1], space_axis.m[0], space_axis.m[-1]]
        im = ax.imshow(data, extent=extent, cmap='jet', vmin=0, vmax=data.mean()+3*data.std())
        ax.vlines(
            self.cal.fel_zero - self.cal.dipole_zero - self.dipole_delay,
            ymin=space_axis.m[0],
            ymax=space_axis.m[-1],
            linestyles='-',
            lw=2,
            color='purple',
            alpha=1,
        )

        ys, xs = np.where(data > 0)
        ax.set_xlim(xmin=time_axis.m[xs.min()], xmax=time_axis.m[xs.max()])
        ax.set_ylim(ymin=space_axis.m[ys.min()], ymax=space_axis.m[ys.max()])
        ax.set_aspect('auto')

        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(400))
        ax.yaxis.set_minor_locator(MultipleLocator(100))
        ax.grid(which='major', color='k', linestyle = '--', linewidth=2, alpha = 0.5)
        ax.grid(which='minor', color='k', linestyle=':', linewidth=1, alpha = 1)

        if fig is not None:
            fig.colorbar(im)
            fig.tight_layout()

        return ax
