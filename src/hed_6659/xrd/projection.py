from hexrd.instrument import HEDMInstrument
import numpy as np
from PIL import Image
import h5py
from hexrd.projections.polar import PolarView
from scipy.interpolate import RegularGridInterpolator
from hexrd.rotations import mapAngle


class Projection:

    def __init__(self,
                 instr_path=None,
                 projection_instr_path=None,
                 two_theta_min=6.0,
                 two_theta_max=72.0,
                 eta_min=0.,
                 eta_max=360.,
                 pixel_size=(0.025, 0.025),
                 cache_coordinate_map=True):

        self.instr_path            = instr_path
        self.projection_instr_path = projection_instr_path
        self._tth_min               = two_theta_min
        self._tth_max               = two_theta_max
        self._eta_min               = eta_min
        self._eta_max               = eta_max
        self._pixel_size            = pixel_size
        self._cache_coordinate_map  = cache_coordinate_map

        self.initialize_polar_view()

    def initialize_instrument(self):
        if self.instr_path is not None:
            ins = h5py.File(self.instr_path, 'r')
            self.instr = HEDMInstrument(instrument_config=ins)
            ins.close()
        else:
            self.instr = None
            msg = (f'instrument file not specified.\n'
                   f'projection will not be done')
            print(msg)

    def initialize_projection_instrument(self):
        if self.projection_instr_path is not None:
            ins = h5py.File(self.projection_instr_path, 'r')
            self.projection_instr = HEDMInstrument(instrument_config=ins)
            ins.close()
        else:
            self.projection_instr = None
            msg = (f'projection instrument file not specified.\n'
                   f'projection will not be done')
            print(msg)

    def initialize_polar_view(self):
        self.pv = PolarView(
                  (self.tth_min, self.tth_max),
                  self.instr,
                  eta_min=self.eta_min,
                  eta_max=self.eta_max,
                  pixel_size=self.pixel_size,
                  cache_coordinate_map=self.cache_coordinate_map)

    def initialize_interpolation_object(self):

        kwargs = {
            'points': (self.eta_grid, self.tth_grid),
            'values': self.pvarray,
            'method': 'linear',
            'bounds_error': False,
            'fill_value': np.nan,
        }

        self.interp_obj = RegularGridInterpolator(**kwargs)

    def project_intensity_detector(self,
                                   det):
        tth, eta = np.degrees(det.pixel_angles())
        eta = mapAngle(eta, (0, 360.0), units='degrees')
        xi = (eta, tth)
        return self.interp_obj(xi)

    def project_intensities_to_raw(self):
        self.projected_image = dict.fromkeys(self.projection_instr.detectors)
        for det_name, det in self.projection_instr.detectors.items():
            self.projected_image[det_name] = self.project_intensity_detector(det)

    def make_projected_image(self, img_dict):
        '''main routine to warp the cake image
        back to an equivalent detector image
        '''
        pvarray = self.pv.warp_image(img_dict,
                                     pad_with_nans=True,
                                     do_interpolation=True)
        pvarray2 = pvarray.data
        pvarray2[pvarray.mask] = np.nan
        self.pvarray = pvarray2

        self.initialize_interpolation_object()
        self.project_intensities_to_raw()

    def write_projected_image(self,
                              fname,
                              fmt='tiff'):
        if hasattr(self, 'projected_image'):
            im = Image.fromarray(self.projected_image['Varex'])
            if '.' in fname:
                msg = (f'file name contains an extension.'
                       f' it will be ignored')
            im.save(f'{fname}.{fmt}')

    @property
    def instr_path(self):
        return self._instr_path

    @instr_path.setter
    def instr_path(self, path):
        if not isinstance(path, str):
            raise ValueError(f'path must be a string.')
        self._instr_path = path
        self.initialize_instrument()

    @property
    def projection_instr_path(self):
        return self._projection_instr_path

    @projection_instr_path.setter
    def projection_instr_path(self, path):
        if not isinstance(path, str):
            raise ValueError(f'path must be a string.')
        self._projection_instr_path = path
        self.initialize_projection_instrument()

    @property
    def tth_max(self):
        return self._tth_max

    @tth_max.setter
    def tth_max(self, v):
        if hasattr(self, 'pv'):
            self._tth_max = v
            self.initialize_polar_view()

    @property
    def tth_min(self):
        return self._tth_min

    @tth_min.setter
    def tth_min(self, v):
        if hasattr(self, 'pv'):
            self._tth_min = v
            self.initialize_polar_view()

    @property
    def eta_min(self):
        return self._eta_min

    @eta_min.setter
    def eta_min(self, v):
        if hasattr(self, 'pv'):
            self._eta_min = v
            self.initialize_polar_view()
    
    @property
    def eta_max(self):
        return self._eta_max

    @eta_max.setter
    def eta_max(self, v):
        if hasattr(self, 'pv'):
            self._eta_max = v
            self.initialize_polar_view()

    @property
    def pixel_size(self):
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, v):
        if hasattr(self, 'pv'):
            if len(v) == 2:
                self._pixel_size = v
            else:
                raise ValueError(f'list/tuple/array must be length 2')
            self.initialize_polar_view()

    @property
    def cache_coordinate_map(self):
        return self._cache_coordinate_map

    @cache_coordinate_map.setter
    def cache_coordinate_map(self, v):
        if hasattr(self, 'pv'):
            if isinstance(v, bool):
                self._cache_coordinate_map = v
            else:
                raise ValueError(f'value must be True/False')
            self.initialize_polar_view()

    @property
    def eta_grid(self):
        if hasattr(self, 'pv'):
            return np.degrees(self.pv.angular_grid[0][:,0])
        else:
            raise ValueError(f'polar view object not initialized')

    @property
    def tth_grid(self):
        if hasattr(self, 'pv'):
            return np.degrees(self.pv.angular_grid[1][0,:])
        else:
            raise ValueError(f'polar view object not initialized')

