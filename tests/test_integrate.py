import os
import h5py
from . import unittest_data_path

from xrd.integrate import integrate_single

data_path = os.path.join(unittest_data_path, 'xrd')

def test_integrate_single(tmp_path):
    projected_image_path = os.path.join(data_path, 'dummy.tif')
    projected_poni_path = os.path.join(data_path, 'dummy.poni')
    
    integrate_single(projected_image_path, projected_poni_path, tmp_path / 'output.h5')

    assert (tmp_path / 'output.h5').exists()

    with h5py.File(tmp_path / 'output.h5', 'r') as f:
        assert 'pattern' in f
        assert 'cake' in f

        assert 'x' in f['pattern']
        assert 'y' in f['pattern']
        assert 'intensity' in f['cake']
        assert 'tth' in f['cake']
        assert 'chi' in f['cake']
        
        assert f['pattern/x'].shape == (1000,)
        assert f['pattern/y'].shape == (1000,)
        assert f['cake/intensity'].shape == (360, 1000)
        assert f['cake/tth'].shape == (1000,)
        assert f['cake/chi'].shape == (360,)
