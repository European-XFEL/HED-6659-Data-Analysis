#!/gpfs/exfel/sw/software/mambaforge/22.11/envs/hexrd/bin/python
#/gpfs/exfel/exp/HED/202405/p006659/usr/Software/tmichela/venv_online/bin/python

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from extra_data import open_run
from kafka import KafkaConsumer

from hed_6659.visar import VISAR, VISAR_DEVICES
from hed_6659.xrd.varex import XRDProcessor


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# TODO class to handle generating output directories
# TODO instantiate poni and projection objections upfront
# TODO how to handle calibration file updates?

PROPOSAL_NUMBER = 6746

ROOT_PATH = Path('/gpfs/exfel/exp/HED/202405/p006659/usr/Software/tmichela')
VISAR_CONFIG = ROOT_PATH / 'visar_calibration_values_p6746.toml'
PONI = ROOT_PATH / 'fake_instrument.poni'  #pyFAI.load('')
HEXRD_SOURCE = ROOT_PATH / 'calibration.hexrd'
HEXRD_TARGET = ROOT_PATH / 'fake_instrument.hexrd'
# HEXRD_PROJECTION = Projection(HEXRD_SOURCE, HEXRD_TARGET)

OUTPUT = Path('/scratch/tmichela')


kafka_brokers = ['exflwgs06:9091']
kafka_topics = ['test.euxfel.hed.daq']
kafka_events = ['daq_run_complete']

# conns = KafkaConsumer(
#     *kafka_topics,
#     bootstrap_servers=kafka_brokers
# )


def _process_visar(proposal, run, visar):
    try:
        r = open_run(proposal, run)
        v = VISAR(r, visar, config_file=VISAR_CONFIG)
    except Exception:
        log.warning(f'VISAR {visar} processing failed', exc_info=True)
        return None, visar, False
    return v, visar, True


def _process_varex(proposal, run):
    try:
        xrd = XRDProcessor(proposal, run, str(HEXRD_SOURCE), str(HEXRD_TARGET), str(PONI), OUTPUT / "VAREX")
        xrd.process()
    except Exception:
        log.warning('XRD processing failed', exc_info=True)
        return None, 'VAREX', False
    return None, 'VAREX', True


def process(proposal, run):
    with ProcessPoolExecutor() as executor:
        futures = []

        for visar in VISAR_DEVICES:
            futures.append(executor.submit(_process_visar, proposal, run, visar))
        futures.append(executor.submit(_process_varex, proposal, run))

        for future in futures:
            component, name, success = future.result()
            try:
                if component is not None:
                    component.save(OUTPUT / "VISAR")
            except Exception:
                log.warning(f'{name} saving failed', exc_info=True)
                success = False

            if success:
                log.info(f'{name} processed successfully')
            else:
                log.info(f'{name} processing failed')


# for record in conns:
#     message = json.loads(record.value)
#     event = message.get('event')

#     if event in kafka_events:
#         proposal = int(message['proposal'])
#         run = int(message['run'])

#         if proposal != PROPOSAL_NUMBER:
#             # ignore messages if they are not for your proposal
#             continue

#         # start your processing here
#         log.info(f'new run: p{proposal} - r{run}')
#         process(proposal, run)


if __name__ == '__main__':
    import sys
    proposal, run, output = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

    global OUTPUT
    OUTPUT = Path(output) / f'r{run:04}'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "VISAR").mkdir()
    (OUTPUT / "VAREX").mkdir()

    process(proposal, run)
