#!/gpfs/exfel/sw/software/mambaforge/22.11/envs/hexrd/bin/python
import json

from threading import Thread
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import logging
import toml
import pyFAI
import numpy as np

from extra_data import open_run
from hed_6659.visar import VISAR, VISAR_DEVICES
from hed_6659.xrd.varex import XRDProcessor
from hed_6659.xrd.projection import Projection


log = logging.getLogger(__name__)


class DataSaver:
    def __init__(self, root_path):
        """
        Initialize the DataSaver with a root path.

        :param root_path: Root directory where data will be saved.
        """
        self.root_path = Path(root_path).resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)

    def _get_group_range(self, run_number, group_size=50):
        """
        Calculate the group range for a given run number.

        :param run_number: The run number to calculate the group for.
        :param group_size: The size of the group (default is 50).
        :return: A tuple (start, end) for the group range.
        """
        start = ((run_number - 1) // group_size) * group_size + 1
        end = start + group_size - 1
        return start, end

    def ensure_directory(self, run_number):
        """
        Ensure that the directory structure exists for a given run number.

        :param run_number: The run number to ensure directories for.
        :return: The path to the specific run directory.
        """
        # Determine the group range
        start, end = self._get_group_range(run_number)

        # Generate the group directory name
        group_dir_name = f"run_{start:04d}_{end:04d}"

        # Create the group directory path
        group_dir_path = self.root_path / group_dir_name
        group_dir_path.mkdir(parents=True, exist_ok=True)

        # Create the specific run directory inside the group
        run_dir_name = f"run_{run_number:04d}"
        run_dir_path = group_dir_path / run_dir_name
        run_dir_path.mkdir(parents=True, exist_ok=True)

        return run_dir_path


def _process_visar(proposal, run, visar, config):
    try:
        r = open_run(proposal, run)
        v = VISAR(r, visar, config_file=config)
        v.compute()
    except Exception:
        log.warning(f'VISAR {visar} processing failed', exc_info=True)
        return None, visar, False
    return v, visar, True


def _process_varex(proposal, run, output, config):
    try:
        config = toml.load(config)['VAREX']
        poni = pyFAI.load(config['poni'])
        varex_proj = Projection(
            instr_path=config['calibration'],
            projection_instr_path=config['projection']
        )

        xrd = XRDProcessor(
            proposal, run,
            (output / 'VAREX').resolve(),
            poni,
            varex_proj,
        )
        xrd.process()
    except Exception:
        log.warning('XRD processing failed', exc_info=True)
        return None, 'VAREX', False
    return None, 'VAREX', True


def executor(proposal, run, config, output):
    ds = DataSaver(output)
    output = ds.ensure_directory(run)
    (output / 'VISAR').mkdir(exist_ok=True)
    (output / 'VAREX').mkdir(exist_ok=True)

    # delete h5 file
    for file in output.glob("*.h5"):
        try:
            file.unlink()  # Delete the file
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

    with ProcessPoolExecutor() as executor:
        futures = []

        for visar in VISAR_DEVICES:
            futures.append(executor.submit(_process_visar, proposal, run, visar, config))
        # futures.append(executor.submit(_process_varex, proposal, run, output, config))

        for future in futures:
            component, name, success = future.result()
            print(name, success)
            try:
                if component is not None:
                    print(component.proposal_number, component.run_number)
                    component.save((output / 'VISAR').resolve())
            except Exception:
                log.warning(f'{name} saving failed', exc_info=True)
                success = False

            if success:
                log.info(f'{name} processed successfully')
            else:
                log.info(f'{name} processing failed')


if __name__ == '__main__':
    import sys
    proposal, run = int(sys.argv[1]), int(sys.argv[2])
    conf_file = sys.argv[3]
    output = sys.argv[4]

    log.info(f"New data: {proposal}, {run}")
    executor(
        proposal,
        run,
        conf_file,
        output,
    )
