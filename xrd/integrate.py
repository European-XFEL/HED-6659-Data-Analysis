from functools import lru_cache
from pyFAI import AzimuthalIntegrator
from pyFAI.multi_geometry import MultiGeometry
import fabio
import h5py


@lru_cache(maxsize=10)
def get_azimuthal_integrator(poni_path, integration_type, num_points):
    """
    Get an AzimuthalIntegrator object from the cache or create a new one if it doesn't exist. Use the cache to avoid
    loading the same PONI file multiple times. (the initialization of the AzimuthalIntegrator object is expensive)

    :param poni_path: path to the PONI file which contains the calibration parameters
    :param integration_type: type of integration to perform (1D or 2D)
    :param num_points: number of points to use for the integration
    :return: AzimuthalIntegrator object
    """
    ai = AzimuthalIntegrator()
    ai.load(poni_path)
    return ai


def integrate_single(image_path, poni_path, h5_output_path, num_points=1000):
    """
    Integrate a single image using a 1D and 2D integration and save the results to an HDF5 file.

    :param image_path: path to the image to integrate
    :param poni_path: path to the PONI file which contains the calibration parameters
    :param h5_output_path: path to the output HDF5 file
    """
    ai_1d = get_azimuthal_integrator(poni_path, "1D", num_points)
    ai_2d = get_azimuthal_integrator(poni_path, "2D", num_points)

    img = fabio.open(image_path).data
    x, y = ai_1d.integrate1d(img, num_points, unit="2th_deg")
    int, tth, chi = ai_2d.integrate2d(img, num_points, 360, unit="2th_deg")

    # save xy and 2D integration results to HDF5 in individual groups (pattern, and cake)
    with h5py.File(h5_output_path, "w") as f:
        f.create_group("pattern")
        f.create_dataset("pattern/x", data=x)
        f.create_dataset("pattern/y", data=y)

        f.create_group("cake")
        f.create_dataset("cake/intensity", data=int)
        f.create_dataset("cake/tth", data=tth)
        f.create_dataset("cake/chi", data=chi)


@lru_cache(maxsize=10)
def get_multi_geometry(
    poni_paths,
    integration_type,
    num_points,
    radial_range=(0, 180),
    azimuth_range=(-180, 180),
):
    """
    Get a MultiGeometry object from the cache or create a new one if it doesn't exist. Use the cache to avoid
    loading the same PONI file multiple times. (the initialization of the MultiGeometry object is expensive)

    It is important that the poni_paths are given as tuple rather than a list to ensure that the cache works correctly.

    :param poni_paths: tuple of paths to the PONI files which contain the calibration parameters
    :param integration_type: type of integration to perform ("1D" or "2D")
    :param num_points: number of points in two theta to use for the integration
    :param radial_range: radial range in two theta degrees for the 1D and 2d integration
    :param azimuth_range: azimuthal range in degrees for the 2D integration
    :return: MultiGeometry object
    """
    return MultiGeometry(
        poni_paths,
        unit="2th_deg",
        radial_range=radial_range,
        azimuth_range=azimuth_range,
    )


def integrate_multiple(
    image_paths,
    poni_paths,
    h5_output_path,
    num_points,
    radial_range=(0, 90),
    azimuth_range=(-180, 180),
):
    """
    Integrate multiple images combining them into a pattern and Cake and save the results to an HDF5 file.

    :param image_paths: list of paths to the images to integrate
    :param poni_paths: list of paths to the PONI files which contain the calibration parameters
    :param h5_output_path: path to the output HDF5 file
    :param num_points: number of points in two theta to use for the integration
    :param radial_range: radial range in two theta degrees for the 1D and 2d integration
    :param azimuth_range: azimuthal range in degrees for the 2D integration
    """
    imgs = [fabio.open(image_path).data for image_path in image_paths]
    mg_1d = get_multi_geometry(tuple(poni_paths), "1D", radial_range, azimuth_range)
    mg_2d = get_multi_geometry(tuple(poni_paths), "2D", radial_range, azimuth_range)

    x, y = mg_1d.integrate1d(imgs, num_points)
    int, tth, chi = mg_2d.integrate2d(imgs, num_points, 360)

    with h5py.File(h5_output_path, "w") as f:
        f.create_group("pattern")
        f.create_dataset("pattern/x", data=x)
        f.create_dataset("pattern/y", data=y)

        f.create_group("cake")
        f.create_dataset("cake/intensity", data=int)
        f.create_dataset("cake/tth", data=tth)
        f.create_dataset("cake/chi", data=chi)
