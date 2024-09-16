# Functions for the two calibration transformation scripts
# created: 2024 09
# creator: thomas stevens

import numpy as np
import time

class PyFAIToHeXRD:
    def __init__(self, poni):
        self.distance = poni.get_dist()
        self.poni1, self.poni2 = poni.get_poni1(), poni.get_poni2()
        self.theta1, self.theta2, self.theta3 = poni.get_rot1(), poni.get_rot2(), poni.get_rot3()
        self.pixel1, self.pixel2 = poni.get_pixel1(), poni.get_pixel2()
        self.wavelength = poni.get_wavelength()
        self.shape = poni.get_shape()
        self.rotation_matrix_pyfai = self.pyfai_rotation_matrix()
        print(self.rotation_matrix_pyfai)

        self.location = None
        self.rotation = None
        self.energy = None

    def pyfai_rotation_matrix(self):
        """ The rotation order is 1 -> 2 -> 3.
        Rotation 3 is right-handed, Rotations 1 and 2 are left-handed.
        The heXRD lab coordinates map to the pyFAI coordinates as follows:
        Z = -X_3
        Y = X_1
        X = X_2
        """
        cosrot1 = np.cos(self.theta1)
        sinrot1 = np.sin(self.theta1)
        cosrot2 = np.cos(self.theta2)
        sinrot2 = np.sin(self.theta2)
        cosrot3 = np.cos(self.theta3)
        sinrot3 = np.sin(self.theta3)

        rot3_mat = np.array([[cosrot3, -sinrot3, 0],
                             [sinrot3, cosrot3, 0],
                             [0, 0, 1]])

        rot2_mat = np.array([[cosrot2, 0, -sinrot2],
                             [0, 1, 0],
                             [sinrot2, 0, cosrot2]])

        rot1_mat = np.array([[1, 0, 0],
                             [0, cosrot1, sinrot1],
                             [0, -sinrot1, cosrot1]])

        rotation_matrix = np.dot(np.dot(rot3_mat, rot2_mat), rot1_mat)
        # order is 3, 2, 1 because matrices
        return rotation_matrix

    def pyFAI_to_heXRD_params(self):
        corner_to_centre = np.array([self.pixel1 * self.shape[0] / 2,
                                     self.pixel2 * self.shape[1] / 2,
                                     0])
        pyfai_to_hexrd_mat = np.array([[0, 1, 0],
                                       [1, 0, 0],
                                       [0, 0, -1]])

        detector_corner_un_rotated = np.array([-self.poni1, -self.poni2, self.distance])
        detector_centre_un_rotated = detector_corner_un_rotated + corner_to_centre
        self.location = np.around(
            np.dot(pyfai_to_hexrd_mat, np.dot(self.rotation_matrix_pyfai, detector_centre_un_rotated)) * 1000, 8)

        rotation_matrix_hexrd = np.dot(np.dot(pyfai_to_hexrd_mat, self.rotation_matrix_pyfai), pyfai_to_hexrd_mat)
        theta_y = np.arcsin(-rotation_matrix_hexrd[2, 0])
        cos_theta_y = np.cos(theta_y)
        theta_x = np.arcsin(np.around(rotation_matrix_hexrd[2, 1] / cos_theta_y, 8))
        theta_z = np.arcsin(np.around(rotation_matrix_hexrd[1, 0] / cos_theta_y, 8))
        self.rotation = np.array([theta_x, theta_y, theta_z])

    def write_detector(self):
        dictionary = {
            'buffer': 'null',
            'detector_type': 'planar',
            'pixels': {'columns': self.shape[1],
                       'rows': self.shape[0],
                       'size': [self.pixel2 * 1000, self.pixel1 * 1000], },
            'transform': {'tilt': self.rotation.tolist(),
                          'translation': self.location.tolist()},
        }
        return dictionary

    def get_energy(self):
        self.energy = 1.23984193e-9 / self.wavelength  # $$E = \frac{hc}{\lambda} = \frac{1.23984193 \times 10^{-6}}{\lambda} \: \frac{eV \: m}{m} = \frac{1.23984193 \times 10^{-9}}{\lambda} \: \frac{keV \: m}{m}$$
        return self.energy


class HeXRDToPyFAI:
    def __init__(self, detector_name, data):
        self.rotation = data['detectors'][detector_name]['transform']['tilt']
        self.location = data['detectors'][detector_name]['transform']['translation']
        self.columns = data['detectors'][detector_name]['pixels']['columns']
        self.rows = data['detectors'][detector_name]['pixels']['rows']
        self.pixel_size = data['detectors'][detector_name]['pixels']['size']
        self.energy = data['beam']['energy']
        self.rotation_matrix_hexrd = self.hexrd_rotation_matrix()

        self.distance = None
        self.poni1 = None
        self.poni2 = None
        self.theta1 = None
        self.theta2 = None
        self.theta3 = None
        self.pixel1 = self.pixel_size[1] / 1000  # Axis 1 in pyFAI is vertical which is axis 2 in heXRD
        self.pixel2 = self.pixel_size[0] / 1000  # heXRD is in mm and pyFAI is in m
        self.wavelength = 1.23984193e-9 / self.energy  # $$E = \frac{hc}{\lambda} = \frac{1.23984193 \times 10^{-6}}{\lambda} \: \frac{eV \: m}{m} = \frac{1.23984193 \times 10^{-9}}{\lambda} \: \frac{keV \: m}{m}$$
        self.shape = [self.rows, self.columns]

    def hexrd_rotation_matrix(self):
        """ The rotation order is X -> Y -> Z.
        All rotations are right-handed.
        The heXRD lab coordinates map to the pyFAI coordinates as follows:
        Z = -X_3
        Y = X_1
        X = X_2
        """
        cosrotx = np.cos(self.rotation[0])
        sinrotx = np.sin(self.rotation[0])
        cosroty = np.cos(self.rotation[1])
        sinroty = np.sin(self.rotation[1])
        cosrotz = np.cos(self.rotation[2])
        sinrotz = np.sin(self.rotation[2])

        rotz = np.array([[cosrotz, -sinrotz, 0],
                         [sinrotz, cosrotz, 0],
                         [0, 0, 1]])

        roty = np.array([[cosroty, 0, sinroty],
                         [0, 1, 0],
                         [-sinroty, 0, cosroty]])

        rotx = np.array([[1, 0, 0],
                         [0, cosrotx, -sinrotx],
                         [0, sinrotx, cosrotx]])

        rotation_matrix = np.dot(np.dot(rotz, roty), rotx)
        # order is Z, Y, X because matrices
        return rotation_matrix

    def hexrd_to_pyfai_params(self):
        hexrd_to_pyfai_mat = np.array([[0, 1, 0],
                                       [1, 0, 0],
                                       [0, 0, -1]])
        rotation_matrix_pyfai = np.dot(np.dot(hexrd_to_pyfai_mat, self.rotation_matrix_hexrd), hexrd_to_pyfai_mat)
        self.theta2 = np.arcsin(rotation_matrix_pyfai[2, 0])
        cos_theta2 = np.cos(self.theta2)
        self.theta1 = np.arcsin(-np.around(rotation_matrix_pyfai[2, 1] / cos_theta2, 8))
        self.theta3 = np.arcsin(np.around(rotation_matrix_pyfai[1, 0] / cos_theta2, 8))

        inverse_rot = np.linalg.inv(rotation_matrix_pyfai)
        detector_centre = np.dot(hexrd_to_pyfai_mat, np.array(self.location)) / 1000
        detector_centre_un_rotated = np.dot(inverse_rot, detector_centre)

        centre_to_corner = np.array([-self.pixel1 * self.shape[0] / 2,
                                     -self.pixel2 * self.shape[0] / 2,
                                     0])
        detector_corner_un_rotated = detector_centre_un_rotated + centre_to_corner
        self.poni1 = -detector_corner_un_rotated[0]
        self.poni2 = -detector_corner_un_rotated[1]
        self.distance = detector_corner_un_rotated[2]

    def write_poni(self, fn, version=2.1, orientation=3):
        txt = ["# Nota: C-Order, 1 refers to the Y axis, 2 to the X axis",
               f"# Calibration converted from heXRD format at {time.ctime()}",
               f"poni_version: {version}",
               f"Detector: Detector",
               f'Detector_config: '
               f'{{"pixel1": {self.pixel1}, "pixel2": {self.pixel2}, "max_shape": {self.shape}, "orientation": {orientation}}}',
               f"Distance: {self.distance}",
               f"Poni1: {self.poni1}",
               f"Poni2: {self.poni2}",
               f"Rot1: {self.theta1}",
               f"Rot2: {self.theta2}",
               f"Rot3: {self.theta3}",
               f"Wavelength: {self.wavelength}", ""]
        fn.write("\n".join(txt))