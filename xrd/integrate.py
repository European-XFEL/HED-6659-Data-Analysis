
from pyFAI import AzimuthalIntegrator
import fabio
import h5py


def integrate_single(image_path, poni_path, output_path):
    """
    Integrate a single image using a 1D and 2D integration and save the results to an HDF5 file.

    :param image_path: path to the image to integrate
    :param poni_path: path to the PONI file which contains the calibration parameters
    :param output_path: path to the output HDF5 file
    """
    ai_1d = AzimuthalIntegrator()
    ai_1d.load(poni_path)

    ai_2d = AzimuthalIntegrator()
    ai_2d.load(poni_path)

    img = fabio.open(image_path).data
    x, y = ai_1d.integrate1d(img, 1000, unit="2th_deg")
    int, tth, chi = ai_2d.integrate2d(img, 1000, 360, unit="2th_deg")

    # save xy and 2D integration results to HDF5 in individual groups (pattern, and cake)
    with h5py.File(output_path, "w") as f:
        f.create_group("pattern")
        f.create_dataset("pattern/x", data=x)
        f.create_dataset("pattern/y", data=y)

        f.create_group("cake")
        f.create_dataset("cake/intensity", data=int)
        f.create_dataset("cake/tth", data=tth)
        f.create_dataset("cake/chi", data=chi)