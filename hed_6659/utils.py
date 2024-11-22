from enum import Enum
from functools import cache

import numpy as np
from extra_data import DataCollection, by_id
from extra_data.exceptions import PropertyNameError, SourceNameError


@cache
def ppu_trigger(run: DataCollection, offset: int=0) -> DataCollection:
    """Select train IDs based on PPU information from a DataCollection."""
    ppu = run["HED_XTD6_PPU/MDL/PPU_TRIGGER"]
    seq_start = ppu["trainTrigger.sequenceStart"].ndarray()
    # The trains picked are the unique values of trainTrigger.sequenceStart
    # minus the first (previous trigger before this run).
    start_train_ids = np.unique(seq_start)[1:] + offset

    # Number of trains picked per sequence
    n_trains = int(ppu["trainTrigger.numberOfTrains"].as_single_value())

    trains = []
    for train_id in start_train_ids:
        trains.extend(list(range(train_id, train_id + n_trains)))
    # return run.select_trains(by_id[trains])
    return trains


@cache
def dipole_trigger(run: DataCollection, offset: int=0) -> DataCollection:
    """Select train IDs based on Dipole trigger information from a DataCollection."""
    seq_start = run["HED_PLAYGROUND/MDL/MASTER_TIMER_DIPOLE", "sequenceStart.value"].ndarray()
    start_train_ids = np.unique(seq_start)[1:] + offset

    # return run.select_trains(by_id[start_train_ids])
    return start_train_ids


@cache
def fel_trigger(run: DataCollection, offset: int=0) -> DataCollection:
    """Select train IDs based on Dipole trigger information from a DataCollection."""
    seq_start = run["HED_PLAYGROUND/MDL/MASTER_TIMER_PPU", "sequenceStart.value"].ndarray()
    start_train_ids = np.unique(seq_start)[1:] + offset

    # return run.select_trains(by_id[start_train_ids])
    return start_train_ids


class DipolePPU(Enum):
    OPEN = np.uint32(34)
    CLOSED = np.uint32(4130)


@cache
def dipole_ppu_open(run: DataCollection):
    """Get trainIds in run with dipole PPU open"""
    try:
        return (
            run["HED_HPLAS_HET/SWITCH/DIPOLE_PPU_OPEN", "hardwareStatusBitField"]
            .xarray()
            .where(lambda x: x == DipolePPU.OPEN.value, drop=True)
            .trainId
        )
    except (SourceNameError, PropertyNameError):
        return (
            run["HED_HPLAS_HET/SHUTTER/DIPOLE_PPU", "isOpened"]
            .xarray()
            .where(lambda x: x == 1, drop=True)
            .trainId
        )


@cache
def sample_name(run, train_id):
    try:
        return (
            run["COMP_HED_IA2_DLC/MDL/DlcSampleMover", "sampleName"][by_id[[train_id]]]
            .ndarray()[0]
            .decode()
        )
    except (SourceNameError, PropertyNameError):
        return "-"