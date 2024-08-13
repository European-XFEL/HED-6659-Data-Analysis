
from pyFAI import AzimuthalIntegrator
import fabio


def integrate_from_projected_image(projected_image_path, projected_poni_path):
    ai = AzimuthalIntegrator()
    ai.load(projected_poni_path)
    img = fabio.open(projected_image_path).data
    return ai.integrate1d(img, 1000, unit="2th_deg")