import os
from . import unittest_data_path

from xrd.integrate import integrate_from_projected_image

data_path = os.path.join(unittest_data_path, 'xrd')

def test_integrate_from_projected_image():
    projected_image_path = os.path.join(data_path, 'dummy.tif')
    projected_poni_path = os.path.join(data_path, 'dummy.poni')
    
    x, y = integrate_from_projected_image(projected_image_path, projected_poni_path)

    print(x)
