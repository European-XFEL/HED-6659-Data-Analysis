import logging
from dataclasses import dataclass
from functools import cache
from PIL import Image
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyFAI
from extra_data import by_id, open_run
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .projection import Projection as ImageProjection
from ..utils import ppu_trigger, dipole_ppu_open, dipole_trigger, fel_trigger, sample_name

log = logging.getLogger('__name__')

_VAREX_DEVICES = {
    'VAREX1': 'HED_EXP_VAREX/CAM/1:daqOutput',
    'VAREX2': 'HED_EXP_VAREX/CAM/2:daqOutput',
}


def save_tiff(array, output):
    Image.fromarray(np.nan_to_num(array)).save(output)


def trim_array(data, value=0., ratio=1):
    """
    Trim the border regions of an array filled with the specified value

    Parameters:
        data (ndarray): The input array of arbitrary dimension.
        value (float): The value to consider as the border. Default is 0.
        ratio (int): The step size for slicing the retained data. Default is 1.

    Returns:
        tuple: A tuple of slice objects, one for each dimension.
    """
    # Prepare slices for all dimensions
    slices = []
    for axis in range(data.ndim):
        if data.ndim == 1:
            # Handle 1D array directly
            non_zero_indices = np.where(data != value)[0]
        else:
            # Find indices along the current axis for higher dimensions
            non_zero_indices = np.where(np.any(data != value, axis=tuple(i for i in range(data.ndim) if i != axis)))[0]

        # Ensure bounds exist, or fallback to include entire dimension
        if len(non_zero_indices) > 0:
            start, end = non_zero_indices[0], non_zero_indices[-1] + 1
        else:
            start, end = 0, data.shape[axis]

        slices.append(slice(start, end, ratio))

    return tuple(slices)


@dataclass
class XRDProcessor:
    proposal_number: int
    run_number: int
    # TODO take Projection and AI object as imput so the don't have to be reinstantiated for each run
    source: str
    target: str
    poni: str
    output: Path

    def __post_init__(self):
        self.run = open_run(self.proposal_number, self.run_number)
        self.varex1 = VAREX(self.run, 'VAREX1')
        self.varex2 = VAREX(self.run, 'VAREX2')
        self.p = ImageProjection(
            instr_path=self.source,
            projection_instr_path=self.target
        )
        self.ai = pyFAI.load(self.poni)
        self.output = Path(self.output)

    def _info(self, train_id, kind):
        return (
            f'p{self.proposal_number:06}, r{self.run_number:04}, '
            f'{kind}, trainId: {train_id}, Sample: {sample_name(self.run, train_id)}'
        )

    @staticmethod
    def _preprocess(varex, train_id, k=0):
        data = varex.frame(train_id).data.squeeze()
        if data.size == 0:
            log.info(f'{varex.data.source} has no data for tid={train_id}')
            return None
        return np.rot90(np.fliplr(data), k=k)

    def project(self, train_id: int):
        data = {
            'Varex1': self._preprocess(self.varex1, train_id=train_id, k=1),
            'Varex2': self._preprocess(self.varex2, train_id=train_id, k=3),
        }

        try:
            self.p.make_projected_image(data)
            # self.p.write_projected_image('test')
            return self.p.projected_image['Varex']
        except Exception:
            log.warning('Projection failed', exc_info=True)
            return None

    def integrate1d(self, image, npt=1500):
        return self.ai.integrate1d(
            image, npt,
            unit="q_A^-1",
            error_model="poisson",
            correctSolidAngle=False,
            polarization_factor=0
        )

    def integrate2d(self, image, npt_rad=1500, npt_azim=2400):
        return self.ai.integrate2d(
            image, npt_rad,
            npt_azim=npt_azim,
            unit="q_A^-1",
            correctSolidAngle=False,
            polarization_factor=0
        )

    def plot(self, train_id, kind, integ, cake):
        slice_q = trim_array(integ.intensity)[0]
        slice_a = trim_array(np.mean(cake.intensity, axis=1))[0]

        fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=(3, 1), figsize=(18, 12))
        fig.add_gridspec(3, hspace=0)

        # cake
        extent=[cake.radial[0], cake.radial[-1], cake.azimuthal[0], cake.azimuthal[-1]]
        im1 = ax[0].imshow(cake.intensity, extent=extent, vmin=0, vmax=np.max(integ.intensity))
        ax[0].axis('auto')
        ax[0].set_ylim(cake.azimuthal[slice_a.start], cake.azimuthal[slice_a.stop])
        ax[0].set_xlim(cake.radial[slice_q.start], cake.radial[slice_q.stop])
        ax[0].grid(color='w', linestyle='--', linewidth=0.5, alpha=0.5, which='major')
        ax[0].set_xticks(np.linspace(1, 10, 10))
        ax[0].set_ylabel(r'Azimuthal Angle, $\phi$ [degrees]')

        for tick in ax[0].xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)

        ax[1].plot(integ.radial, integ.intensity)
        ax[1].grid(color='k', linestyle='--', linewidth=0.5, alpha=0.5, which='major')
        ax[1].set_xlabel(r'Scattering Vector, Q [$\AA^{-1}$]')
        ax[1].set_ylabel('Azimuthally-Averaged, \n Dark-Sub. Intensity [Counts]')

        for axis in ax:
            axis.label_outer()
            divider = make_axes_locatable(axis)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            if axis == ax[0]:
                cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
                cbar.set_label('Dark-Subtracted Intensity [counts]', rotation=270, labelpad=15)
            else:
                cax.yaxis.set_ticks([])
                cax.xaxis.set_ticks([])
                for spine in cax.spines.values():
                    spine.set_edgecolor('white')

        fig.suptitle(self._info(train_id, kind), y=0.895) 
        plt.subplots_adjust(wspace=0, hspace=0.005)
        return fig

    def save(self, train_id, kind, image=None, integration=None, figure=None):
        # kind = shot/preshot
        # save both varex
        save_tiff(self.varex1.frame(train_id), self.output / f'VAREX_1_tid_{train_id}_{kind}.tiff')
        save_tiff(self.varex2.frame(train_id), self.output / f'VAREX_2_tid_{train_id}_{kind}.tiff')

        # save poni
        self.ai.write(self.output / f'{train_id}.poni')

        if image is None:
            return  # projection failed, we return early

        # save assembled image as tiff
        save_tiff(image, self.output / f'VAREX_tid_{train_id}_{kind}.tiff')

        # 1d projection as txt
        with open(self.output / f'azimuthal_integration_tid_{train_id}_{kind}.txt', 'w') as f:
            f.write(f'sample ID: {sample_name(self.run, train_id)}\n')
            f.write('q [1/Å], Azimuthally-Averaged Intensity [counts], Azimuthal Sigma [counts]\n')
            for q, I, sigma in zip(integration.radial, integration.intensity, integration.sigma):
                f.write(f'{q:.6e}, {I:.6e}, {sigma:6e}\n')

        # png
        figure.savefig(self.output / f'cake_tid_{train_id}_{kind}.png', dpi=200)

    def process_train(self, train_id: int, kind: str):
        projected = self.project(train_id)
        if projected is not None:
            # projected can be None if either data loading or projection failed
            integ = self.integrate1d(projected)
            cake = self.integrate2d(projected)
            fig = self.plot(train_id, kind, integ, cake)

        self.save(train_id, kind, projected, integ, fig)

    def process(self):
        for train_id in fel_trigger(self.run):
            # preshot
            self.process_train(train_id, kind='preshot')

        for train_id in dipole_trigger(self.run):
            # shot
            self.process_train(train_id, kind='shot')


class VAREX:

    def __init__(self, run, name):
        self.run = run
        self.data = run[_VAREX_DEVICES[name], 'data.image.pixels']

    @cache
    def dark(self, n_frames=30):
        """compute dark frame from max n_frames in the run
        
        The dark is computed from frame excluding when the dipole shutter is
        open or the PPU is open
        """
        tid_ppu = ppu_trigger(self.run)
        tid_dipole = dipole_ppu_open(self.run)
        train_ids = set(self.run.train_ids) - set(tid_dipole.data) - set(tid_ppu)
        dark_frames = sorted(train_ids)[:n_frames]
        return self.data[by_id[dark_frames]].xarray().mean('trainId')

    def shot(self):
        tid = dipole_trigger(self.run)
        if len(tid) == 0:
            return
        return self.data[by_id[tid]].xarray() - self.dark()

    def preshot(self):
        tid = fel_trigger(self.run)
        if len(tid) == 0:
            return
        return self.data[by_id[tid]].xarray() - self.dark()

    def frame(self, train_ids: int):
        """Get background substracted frame for train ID"""
        return self.data[by_id[[train_ids]]].xarray().squeeze() - self.dark()
