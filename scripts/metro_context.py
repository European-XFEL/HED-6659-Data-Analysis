from functools import lru_cache
from pathlib import Path
from socket import gethostname
from time import time

import cv2
import numpy as np
import toml
import xarray as xr
from metropc.context import View, ViewGroup, parameters
from metropc.viewdef import Parameter
from scipy.interpolate import griddata

from shock import find_shocks


parameters(
    calibration_file = "/gpfs/exfel/exp/HED/202405/p006746/usr/Shared/visar_calibration/visar_calibration_values_p6746.toml"
)


@View.Scalar(name='triggerFEL')
def ppu_trigger(
    sequence_start: f'HED_PLAYGROUND/MDL/MASTER_TIMER_PPU.sequenceStart',
    tid: 'internal#train_id'
):
    if sequence_start == tid:
        return sequence_start


@View.Scalar(name='triggerDipole')
def dpu_trigger(
    sequence_start: f'HED_PLAYGROUND/MDL/MASTER_TIMER_DIPOLE.sequenceStart',
    tid: 'internal#train_id'
):
    if sequence_start == tid:
        return sequence_start


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


# @View.Scalar(name='dipoleOpen')
# def dipole_ppu(
#     status: 'HED_HPLAS_HET/SWITCH/DIPOLE_PPU_OPEN.hardwareStatusBitField'=0,
#     shutter: 'HED_HPLAS_HET/SHUTTER/DIPOLE_PPU.isOpened'=None
# ):
#     if int(status) == 34 or shutter == 1:
#         return True
# TODO user the shutter information to see if the dipole is open
# if we see one DIPOLE shot from the master device we log that
# get the next detector frame 
# ignore following frames unless there's a second train ID on master while the shutter is open


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


class StreakCamera(ViewGroup):
    SWEEP_SPEED = {1: 50, 2: 20, 3: 10, 4: 5, 5: 1, 6: 100}

    rot90: Parameter = 1
    fliplr: Parameter = False
    flipud: Parameter = False
    downsample: Parameter = 4

    # control: Parameter = 'HED_SYDOR_TEST/CTRL/CONTROL_UNIT_2'
    # trigger: Parameter = 'HED_EXP_VISAR/TSYS/ARM_2_TRIG'
    # detector: Parameter = 'HED_SYDOR_TEST/CAM/KEPLER_2:output'

    def __init__(self, *args, prefix="KEPLER1", trigger_offset=0, a=1, **kwargs):
        super().__init__(*args, prefix=f'{prefix}/', **kwargs)
        self._ref = Settings(calibration_file, prefix)
        self.t0 = time()

        # initialize image views
        self._labelled_image(kind='reference')
        # self._labelled_image(kind='preShot')
        self._labelled_image(kind='shot')

        # offset between the Dipole train ID and the camera train ID
        self.offset = trigger_offset
        self._ref_cache = None

    @property
    def ref(self):
        if calibration_file != self._ref.file_path:
            self._ref = Settings(calibration_file, self.prefix.rstrip('/'))
        return self._ref

    @lru_cache()
    def delay(self, pixel_index, sweep_time):
        """Compute time delay
        """
        time_poly = np.hstack([[0], self._config['timeAxisPolynomial'][f'{sweep_time}ns']])
        return np.poly1d(time_poly[::-1])(pixel_index)

    @View.Scalar(name='{prefix}trigger')
    def train_id(
        self,
        tid: 'internal#train_id',
        sequence_start: 'HED_PLAYGROUND/MDL/MASTER_TIMER_DIPOLE.sequenceStart'=None,
        shutter: 'HED_HPLAS_HET/SWITCH/DIPOLE_PPU_OPEN.hardwareStatusBitField'=None,
    ):
        if shutter == 34:
            print('shutter', tid)
            return tid
        if sequence_start + self.offset == tid:
            print('sequence', tid)
            return tid

    @View.Image(name='{prefix}rawData')
    def raw_data(self, data: '{detector}', tid: 'internal#train_id'):
        self.t0 = time()
        if prefix == 'KEPLER1/':
            print('data', tid)

        if self.rot90:
            data = np.rot90(data, self.rot90)
        if self.flipud:
            data = np.flipud(data)
        if self.fliplr:
            data = np.fliplr(data)

        return data

    @View.Scalar(name='{prefix}trainId')
    def _train_id(_: '{prefix}shot', train_id: 'internal#train_id'):
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

    @View.Scalar(name='{prefix}info')
    def info(
        train_id: '{prefix}trainId',
        dipole_delay: 'view#dipoleDelay',
        dipole_energy: 'view#dipoleEnergy',
        sweep_delay: '{prefix}sweepDelay',
        sweep_time: '{prefix}sweepTime',
    ):
        return f"""\
            <div style="text-align: left">
                <b>SHOT - TrainID:</b> {train_id} <br><br>
                <b>Dipole delay:</b> {round(dipole_delay, 3)} ns <br>
                <b>Dipole energy:</b> {dipole_energy:.3f} J <br>
                <br>
                <b>Sweep delay:</b> {sweep_delay:.3f} ns <br>
                <b>Sweep time:</b> {sweep_time} µs <br>
            </div>
            """

    def _correct(self, data, sweep_time):
        data = cv2.resize(data, (data.shape[1] * 2, data.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        data = np.flipud(np.fliplr(data))
        # return self._labelled(data, sweep_time, xray_delay, dipole_delay)
        return data

    @View.Compute(name='{prefix}shotData', hidden=True)
    def _shot(self,
             data: '{prefix}rawData',
             sweep_time: '{prefix}sweepTime',
            #  _: "view#triggerDipole"):
            _: "{prefix}trigger",
    ):
        print(prefix, data.shape)
        return self._correct(data, sweep_time)

    @View.Compute(name='{prefix}preShotData', hidden=True)
    def _preshot(self,
                data: '{prefix}rawData',
                sweep_time: '{prefix}sweepTime',
                _: "view#triggerFEL"):
        return self._correct(data, sweep_time,)

    @View.Compute(name='{prefix}referenceData', hidden=True)
    def _reference(self,
                data: '{prefix}rawData',
                sweep_time: '{prefix}sweepTime',
                fel: "view#triggerFEL"=None,
                # dipole: "view#triggerDipole"=None,
                dipole: "{prefix}trigger"=None,
    ):
        if fel is None and dipole is None:
            ref = self._correct(data, sweep_time)
            self._ref_cache = ref
            return ref

    @ViewPrototype.Image(name='{prefix}{kind}')
    def _labelled_image(
        self,
        data: 'view#{prefix}{kind}Data',
        sweep_time: '{prefix}sweepTime',
        xray_delay: '{prefix}xrayDelay',
        dipole_delay: 'view#dipoleDelay',
    ):
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

    # @View.Compute(name='{prefix}shocks')
    # def shocks(
    #     self,
    #     data: 'view#{prefix}shotData',
    #     sweep_time: '{prefix}sweepTime',
    #     dipole_delay: 'view#dipoleDelay',
    # ):
    #     print(prefix, 'T')
    #     if self._ref_cache is None:
    #         return
    #     if prefix != 'KEPLER1/':
    #         return

    #     slice_ = remove_border(data, ratio=1)

    #     def _time_axis():
    #         tx = self.ref._delay(range(data.shape[1]), sweep_time)
    #         offset = self.ref.dipole(sweep_time) + dipole_delay
    #         return (tx - offset)[slice_[1]]

    #     data = cv2.resize(data[slice_], (1024, 1024))
    #     ref = cv2.resize(self._ref_cache[slice_], (1024, 1024))

    #     phaseroi = [170, 720,  300, 150]
    #     refroi   = [170, 100,  300, 300]
    #     W         =  20    #Width of backward-looking window (pixels)
    #     F         =   2    #Width of forward-looking window  (pixels)
    #     p_thresh  =   5e-4 #Threshold for shock detection algorithm (p-value)

    #     ds        = 100    #Number of downsampled points (downsampling applied prior
    #                     #to shock search)
    #     ta = _time_axis()
    #     ta = np.linspace(ta[0], ta[-1], 1024)

    #     res = find_shocks(ref, refroi, data, phaseroi, ta, -1, W, F, p_thresh, ds)
    #     print(res)
    #     return res


class VISARBase(StreakCamera):
    arm: Parameter = 'COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_2'

    @View.Scalar(name='{prefix}info')
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
            <div style="text-align: left; font-family: monospace;">
                <table style="width: 100%;">
                    <tr><td><b>SHOT - TrainID:</b></td><td style="text-align: right;">{train_id}</td></tr>
                    <tr><td colspan="2"><br></td></tr>
                    <tr><td><b>Dipole delay:</b></td><td style="text-align: right;">{round(dipole_delay, 3)} ns</td></tr>
                    <tr><td><b>Dipole energy:</b></td><td style="text-align: right;">{dipole_energy:.3f} J</td></tr>
                    <tr><td colspan="2"><br></td></tr>
                    <tr><td><b>Etalon thickness:</b></td><td style="text-align: right;">{etalon_thickness:.3f} mm</td></tr>
                    <tr><td><b>Motor displacement:</b></td><td style="text-align: right;">{motor_displacement:.3f} mm</td></tr>
                    <tr><td><b>Sensitivity:</b></td><td style="text-align: right;">{sensitivity:.3f} m/s</td></tr>
                    <tr><td><b>Sweep delay:</b></td><td style="text-align: right;">{sweep_delay:.3f} ns</td></tr>
                    <tr><td><b>Sweep time:</b></td><td style="text-align: right;">{sweep_time} µs</td></tr>
                    <tr><td><b>Temporal delay:</b></td><td style="text-align: right;">{temporal_delay:.3f} ns</td></tr>
                    <tr><td><b>Zero delay position:</b></td><td style="text-align: right;">{zero_delay_position:.3f} mm</td></tr>
                </table>
            </div>
            """


class VISAR(VISARBase):
    def _correct(self, data, sweep_time):
        source, target = self.ref.map(sweep_time, data.shape)
        data = cv2.remap(data, source, target, cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        data = np.flipud(np.fliplr(data))
        return data


class VISAR_1w(VISARBase):
    fliplr: Parameter = True


# ---

if gethostname().startswith('max-exfl'):
    channel = 'daqOutput' #'output'
    source = ''
else:
    channel = 'output'
    source = ''
    # TODO when DAQ is fixed
    # channel = daqOutput
    # source = 'HED_DAQ_DATA/DA/4:output'


kepler1 = VISAR(
    arm='COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_1',
    control='HED_SYDOR_TEST/CTRL/CONTROL_UNIT_1',
    trigger='HED_EXP_VISAR/TSYS/ARM_1_TRIG',
    detector=f'HED_SYDOR_TEST/CAM/KEPLER_1:{channel}[data.image.pixels]{source}',
    prefix="KEPLER1",
    trigger_offset=0,
)

kepler2 = VISAR(
    arm='COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_2',
    control='HED_SYDOR_TEST/CTRL/CONTROL_UNIT_2',
    trigger='HED_EXP_VISAR/TSYS/ARM_2_TRIG',
    detector=f'HED_SYDOR_TEST/CAM/KEPLER_2:{channel}[data.image.pixels]{source}',
    prefix="KEPLER2",
    trigger_offset=3,
)

visar1w = VISAR_1w(
    arm='COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_3',
    trigger='HED_EXP_VISAR/TSYS/ARM_3_TRIG',
    detector=f'HED_EXP_VISAR/EXP/ARM_3_STREAK:{channel}[data.image.pixels]{source}',
    control='HED_EXP_VISAR/EXP/ARM_3_STREAK',
    prefix='VISAR_1w',
    trigger_offset=1,
)

SOP = StreakCamera(
    trigger="HED_EXP_VISAR/TSYS/SOP_TRIG",
    detector=f"HED_EXP_VISAR/EXP/SOP_STREAK:{channel}[data.image.pixels]{source}",
    control="HED_EXP_VISAR/EXP/SOP_STREAK",
    prefix="SOP",
    trigger_offset=1,
)
