from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import toml
from scipy.interpolate import griddata

from metropc.context import ctx, View, ViewGroup, ViewPrototype
from metropc.viewdef import Parameter


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


class VISARCalibration(ViewGroup):
    SWEEP_SPEED = {1: 50, 2: 20, 3: 10, 4: 5, 5: 1, 6: 100}

    # kepler: Parameter = ""
    calibration_file: Parameter = "/gpfs/exfel/exp/HED/202405/p006656/scratch/amore-dev/visar_calibration_values.toml"

    arm: Parameter = 'COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_2'
    control: Parameter = 'HED_SYDOR_TEST/CTRL/CONTROL_UNIT_2'
    trigger: Parameter = 'HED_EXP_VISAR/TSYS/ARM_2_TRIG'
    detector: Parameter = 'HED_SYDOR_TEST/CAM/KEPLER_2:output'

    def __init__(self, *args, prefix="KEPLER1", **kwargs):
        super().__init__(*args, prefix=prefix, **kwargs)
        cal = toml.load(self.calibration_file)
        self._config = cal[prefix]
        self._config.update(cal['global'])

    @lru_cache()
    def map(self, sweep_time, shape):
        tr_map_file = self._config['transformationMaps'][f'{sweep_time}ns']
        file_path = Path(self._config['dirTransformationMaps']) / tr_map_file
        coords = np.loadtxt(file_path, delimiter=',')
        target = coords[..., 2:]
        source = coords[..., :2]

        y, x = shape
        grid_1, grid_2 = np.mgrid[:y, :x]
        grid_z = griddata(target, source, (grid_1, grid_2), method='linear')
        map_1 = grid_z[..., 1].astype(np.float32)
        map_2 = grid_z[..., 0].astype(np.float32)

        return map_1, map_2

    @View.Image(name='{prefix}rawData')
    def raw_data(data: '{detector}:daqOutput[data.image.pixels]'):
        import time
        print(time.time())
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
        reference_position = self._config['positionTrigger_ref'][f'{sweep_time}ns']
        return position - reference_position

    @View.Image(name='{prefix}correctedShot')
    def corrected_data_shot(self, data: '{prefix}rawShot', sweep_time: '{prefix}sweepTime'):
        source, target = self.map(sweep_time, data.shape)

        image = np.rot90(data, -1)
        dat = cv2.remap(image, source, target, cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        # dat = np.fliplr(dat)
        # data = np.flipud(dat)

        # TODO add vline
        import time
        print('>>', time.time())
        return dat

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


kepler1 = VISARCalibration(
    arm='COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_1',
    control='HED_SYDOR_TEST/CTRL/CONTROL_UNIT_1',
    trigger='HED_EXP_VISAR/TSYS/ARM_1_TRIG',
    detector='HED_SYDOR_TEST/CAM/KEPLER_1',
    prefix="KEPLER1"
)

kepler2 = VISARCalibration(
    arm='COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_2',
    control='HED_SYDOR_TEST/CTRL/CONTROL_UNIT_2',
    trigger='HED_EXP_VISAR/TSYS/ARM_2_TRIG',
    detector='HED_SYDOR_TEST/CAM/KEPLER_2',
    prefix="KEPLER2"
)
