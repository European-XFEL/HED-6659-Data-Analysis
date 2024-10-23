from functools import lru_cache
from pathlib import Path
from time import time

import cv2
import numpy as np
import toml
import xarray as xr
from scipy.interpolate import griddata

from metropc.context import View, ViewGroup
from metropc.viewdef import Parameter


def remove_border(data, value=0., ratio=1):
    """Find zero-rows/cols borders

    returns sliced array without zero borders
    """
    # find first/last non-zero rows and columns
    non_zero_rows = np.where(np.any(data != 0, axis=1))[0]
    non_zero_cols = np.where(np.any(data != 0, axis=0))[0]
    # Get the bounds
    row_start, row_end = non_zero_rows[0], non_zero_rows[-1] + 1
    col_start, col_end = non_zero_cols[0], non_zero_cols[-1] + 1
    return slice(row_start, row_end, ratio), slice(col_start, col_end, ratio)


@View.Scalar(name='dipoleEnergy')
def dipole_energy(energy: 'APP_DIPOLE/MDL/DIPOLE_DIAGNOSTIC.energy2W'):
    return energy


@View.Scalar(name='dipoleDelay')
def dipole_delay(delay: 'APP_DIPOLE/MDL/DIPOLE_TIMING.actualPosition'):
    return delay


@View.Scalar(name='dipoleOpen')
def dipole_ppu(status: 'HED_HPLAS_HET/SWITCH/DIPOLE_PPU_OPEN.hardwareStatusBitField'):
    if int(status) == 34:
        return True


class Settings:
    def __init__(self, file_path, visar):
        self.file_path = file_path
        config = toml.load(file_path)
        self.conf = config[visar]
        self.conf.update(config['global'])

    def __getitem__(self, key):
        return self.conf[key]

    @lru_cache()
    def map(self, sweep_time, shape):
        tr_map_file = self['transformationMaps'][f'{sweep_time}ns']
        file_path = Path(self['dirTransformationMaps']) / tr_map_file
        coords = np.loadtxt(file_path, delimiter=',')
        target = coords[..., 2:]
        source = coords[..., :2]

        y, x = shape
        grid_1, grid_2 = np.mgrid[:y, :x]
        grid_z = griddata(target, source, (grid_1, grid_2), method='linear')
        map_1 = grid_z[..., 1].astype(np.float32)
        map_2 = grid_z[..., 0].astype(np.float32)

        return map_1, map_2

    @lru_cache()
    def _delay(self, pixel_index, sweep_time):
        """Compute time delay
        """
        time_poly = np.hstack([[0], self['timeAxisPolynomial'][f'{sweep_time}ns']])
        return np.poly1d(time_poly[::-1])(pixel_index)

    @lru_cache()
    def sweep_delay(self, sweep_time):
        return self['positionTrigger_ref'][f'{sweep_time}ns']

    @lru_cache()
    def fel(self, sweep_time):
        return self._delay(self['pixXray'], sweep_time)

    @lru_cache()
    def dipole(self, sweep_time):
        return self._delay(self['pixDipole_0ns'][f'{sweep_time}ns'], sweep_time)


class VISARBase(ViewGroup):
    SWEEP_SPEED = {1: 50, 2: 20, 3: 10, 4: 5, 5: 1, 6: 100}

    calibration_file: Parameter = "/gpfs/exfel/data/user/tmichela/tmp/visar_calibration_values_6656.toml"
    rot90: Parameter = -1
    fliplr: Parameter = False
    flipud: Parameter = False
    downsample: Parameter = 4

    arm: Parameter = 'COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_2'
    control: Parameter = 'HED_SYDOR_TEST/CTRL/CONTROL_UNIT_2'
    trigger: Parameter = 'HED_EXP_VISAR/TSYS/ARM_2_TRIG'
    detector: Parameter = 'HED_SYDOR_TEST/CAM/KEPLER_2:output'

    def __init__(self, *args, prefix="KEPLER1", **kwargs):
        super().__init__(*args, prefix=f'{prefix}/', **kwargs)
        self._ref = Settings(self.calibration_file, prefix)
        self.t0 = time()

    @property
    def ref(self):
        if self.calibration_file != self._ref.file_path:
            self._ref = Settings(self.calibration_file, self.prefix.rstrip('/'))
        return self._ref

    @lru_cache()
    def delay(self, pixel_index, sweep_time):
        """Compute time delay
        """
        time_poly = np.hstack([[0], self._config['timeAxisPolynomial'][f'{sweep_time}ns']])
        return np.poly1d(time_poly[::-1])(pixel_index)

    @View.Image(name='{prefix}rawData', hidden=True)
    def raw_data(self, data: '{detector}:daqOutput[data.image.pixels]'):
        self.t0 = time()

        if self.rot90:
            data = np.rot90(data, self.rot90)
        if self.flipud:
            data = np.flipud(data)
        if self.fliplr:
            data = np.fliplr(data)

        return data
    
    @View.Image(name='{prefix}rawShot')
    def raw_data_shot(data: 'view#{prefix}rawData', hw_status: 'view#dipoleOpen'):
        return data

    @View.Scalar(name='{prefix}trainId')
    def train_id(_: '{prefix}rawShot', train_id: 'internal#train_id'):
        return train_id

    @View.Scalar(name='{prefix}sweepTime')
    def sweep_time(self, sweep_time: '{control}.sweepSpeed'):
        return self.SWEEP_SPEED[int(sweep_time)]

    @View.Scalar(name='{prefix}sweepDelay')
    def sweep_delay(self, position: '{trigger}.actualPosition', sweep_time: '{prefix}sweepTime'):
        reference_position = self.ref['positionTrigger_ref'][f'{sweep_time}ns']
        return position - reference_position

    @View.Scalar(name='{prefix}xrayDelay')
    def xray_delay(self,
                   sweep_time: '{prefix}sweepTime',
                   sweep_delay: '{prefix}sweepDelay',
                   dipole_delay: 'view#dipoleDelay'
    ):
        return self.ref.fel(sweep_time) - self.ref.dipole(sweep_time) - dipole_delay - sweep_delay

    @View.Scalar(name='{prefix}shotInfo')
    def info(
        train_id: '{prefix}trainId',
        dipole_delay: 'view#dipoleDelay',
        dipole_energy: 'view#dipoleEnergy',
        etalon_thickness: '{arm}.etalonThickness',
        motor_displacement: '{arm}.motorDisplacement',
        sensitivity: '{arm}.sensitivity',
        sweep_delay: '{prefix}sweepDelay',
        sweep_time: '{prefix}sweepTime',
        temporal_delay: '{arm}.temporalDelay',
        zero_delay_position: '{arm}.zeroDelayPosition',
    ):
        return f"""\
            <div style="text-align: left">
                <b>SHOT - TrainID:</b> {train_id} <br><br>
                <b>Dipole delay:</b>: {dipole_delay} ns <br>
                <b>Dipole energy:</b>: {dipole_energy:.3f} J <br>
                <br>
                <b>etalon thickness:</b>: {etalon_thickness:.3f} mm <br>
                <b>Motor displacement:</b>: {motor_displacement:.3f} mm <br>
                <b>Sensitivity:</b>: {sensitivity:.3f} m / s <br>
                <b>Sweep delay:</b>: {sweep_delay:.3f} ns <br>
                <b>Sweep time:</b>: {sweep_time} µs <br>
                <b>Temporal delay:</b>: {temporal_delay:.3f} ns <br>
                <b>Zero delay position:</b>: {zero_delay_position:.3f} mm <br>
            </div>
            """

    def _labelled(self, data, sweep_time, xray_delay, dipole_delay):
        slice_ = remove_border(data, ratio=self.downsample)

        def _time_axis():
            tx = self.ref._delay(range(data.shape[1]), sweep_time)
            offset = self.ref.dipole(sweep_time) + dipole_delay
            return (tx - offset)[slice_[1]]

        def _space_axis():
            sx = np.arange(data.shape[0]) * self.ref['dx']
            return (sx - sx.mean())[slice_[0]]
    
        return xr.DataArray(
            data[slice_],
            coords={'Time [ns]': _time_axis(),
                    'Distance [um]': _space_axis()},
            dims=['Distance [um]', 'Time [ns]'],
            attrs={'vlines': {xray_delay: f'FEL delay ({round(xray_delay, 3)} ns)'}},
        )


class VISAR(VISARBase):
    @View.Image(name='{prefix}correctedShot')
    def corrected_data_shot(self,
                            data: '{prefix}rawShot',
                            sweep_time: '{prefix}sweepTime',
                            xray_delay: '{prefix}xrayDelay',
                            dipole_delay: 'view#dipoleDelay'):
        source, target = self.ref.map(sweep_time, data.shape)

        data = cv2.remap(data, source, target, cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        res = self._labelled(data, sweep_time, xray_delay, dipole_delay)
        print('>>', self.__class__.__name__, round(time() - self.t0, 3))
        return res


class VISAR_1w(VISARBase):
    fliplr: Parameter = True

    @View.Image(name='{prefix}correctedShot')
    def corrected_data_shot(self,
                            data: '{prefix}rawShot',
                            sweep_time: '{prefix}sweepTime',
                            xray_delay: '{prefix}xrayDelay',
                            dipole_delay: 'view#dipoleDelay'):
        data = cv2.resize(data, (data.shape[1] * 2, data.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        res = self._labelled(data, sweep_time, xray_delay, dipole_delay)
        print('>>', self.__class__.__name__, round(time() - self.t0, 3))
        return res


kepler1 = VISAR(
    arm='COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_1',
    control='HED_SYDOR_TEST/CTRL/CONTROL_UNIT_1',
    trigger='HED_EXP_VISAR/TSYS/ARM_1_TRIG',
    detector='HED_SYDOR_TEST/CAM/KEPLER_1',
    prefix="KEPLER1"
)

kepler2 = VISAR(
    arm='COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_2',
    control='HED_SYDOR_TEST/CTRL/CONTROL_UNIT_2',
    trigger='HED_EXP_VISAR/TSYS/ARM_2_TRIG',
    detector='HED_SYDOR_TEST/CAM/KEPLER_2',
    prefix="KEPLER2"
)

visar1w = VISAR_1w(
    arm='COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_3',
    trigger='HED_EXP_VISAR/TSYS/ARM_3_TRIG',
    detector='HED_EXP_VISAR/EXP/ARM_3_STREAK',
    ctrl='HED_EXP_VISAR/EXP/ARM_3_STREAK',
    prefix='VISAR_1w',
)
