import os
import h5py
from . import unittest_data_path

from xrd.integrate import (
    integrate_single,
    get_azimuthal_integrator,
    integrate_multiple,
    get_multi_geometry,
)

data_path = os.path.join(unittest_data_path, "xrd")


def test_integrate_single(tmp_path):
    projected_image_path = os.path.join(data_path, "dummy.tif")
    projected_poni_path = os.path.join(data_path, "dummy.poni")

    integrate_single(projected_image_path, projected_poni_path, tmp_path / "output.h5")

    assert (tmp_path / "output.h5").exists()

    with h5py.File(tmp_path / "output.h5", "r") as f:
        assert "pattern" in f
        assert "cake" in f

        assert "x" in f["pattern"]
        assert "y" in f["pattern"]
        assert "intensity" in f["cake"]
        assert "tth" in f["cake"]
        assert "chi" in f["cake"]

        assert f["pattern/x"].shape == (1000,)
        assert f["pattern/y"].shape == (1000,)
        assert f["cake/intensity"].shape == (360, 1000)
        assert f["cake/tth"].shape == (1000,)
        assert f["cake/chi"].shape == (360,)


def test_ai_caching():
    ai1 = get_azimuthal_integrator(os.path.join(data_path, "dummy.poni"), "1D", 1000)
    ai2 = get_azimuthal_integrator(os.path.join(data_path, "dummy.poni"), "1D", 1000)
    assert ai1 is ai2, "Should return the same object for the same poni_path and type"

    ai2 = get_azimuthal_integrator(os.path.join(data_path, "dummy.poni"), "2D", 1000)
    assert ai1 is not ai2, "Should return different objects for different types"

    ai2 = get_azimuthal_integrator(os.path.join(data_path, "dummy.poni"), "1D", 2000)
    assert ai1 is not ai2, "Should return different objects for different num_points"


def test_integrate_multiple(tmp_path):
    projected_image_path = os.path.join(data_path, "dummy.tif")
    projected_image_paths = (projected_image_path, projected_image_path)
    projected_poni_path = os.path.join(data_path, "dummy.poni")
    projected_poni_paths = (projected_poni_path, projected_poni_path)

    integrate_multiple(
        projected_image_paths, projected_poni_paths, tmp_path / "output.h5", 1000
    )

    assert (tmp_path / "output.h5").exists()

    with h5py.File(tmp_path / "output.h5", "r") as f:
        assert "pattern" in f
        assert "cake" in f

        assert "x" in f["pattern"]
        assert "y" in f["pattern"]
        assert "intensity" in f["cake"]
        assert "tth" in f["cake"]
        assert "chi" in f["cake"]

        assert f["pattern/x"].shape == (1000,)
        assert f["pattern/y"].shape == (1000,)
        assert f["cake/intensity"].shape == (360, 1000)
        assert f["cake/tth"].shape == (1000,)
        assert f["cake/chi"].shape == (360,)


def test_mg_caching():
    mg1 = get_multi_geometry(
        (os.path.join(data_path, "dummy.poni"),), "1D", 1000, radial_range=(0, 180)
    )
    mg2 = get_multi_geometry(
        (os.path.join(data_path, "dummy.poni"),), "1D", 1000, radial_range=(0, 360)
    )
    assert mg1 is not mg2, "Should return different objects for different radial ranges"

    mg3 = get_multi_geometry((os.path.join(data_path, "dummy.poni"),), "1D", 1000)
    mg4 = get_multi_geometry((os.path.join(data_path, "dummy.poni"),), "1D", 2000)
    assert mg3 is not mg4, "Should return different objects for different num_points"

    mg5 = get_multi_geometry((os.path.join(data_path, "dummy.poni"),), "1D", 1000)
    mg6 = get_multi_geometry((os.path.join(data_path, "dummy.poni"),), "2D", 1000)
    assert (
        mg5 is not mg6
    ), "Should return different objects for different integration types"
