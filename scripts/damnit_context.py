import numpy as np
from damnit_ctx import Cell, Variable

from visar import VISAR

##########################
###   MyMDC parameters ###
##########################

@Variable(title='Run Type')
def run_type(run, runtype:'mymdc#run_type'):
    """Run type from MyMDC
    """
    return runtype
    
@Variable(title='Sample')
def sample(run, samp:'mymdc#sample_name'):
    """Sample name from MyMDC
    """
    return samp

@Variable(title='Sample pos', data='raw')
def sample_pos(run):
    """Sample name from FS4 positioner middle layer
    """
    return run.get_run_value('HED_PLAYGROUND/MDL/SamplePositionList_2','actualSampleName')

##########################
### Control parameters ###
##########################

@Variable(title="Trains", summary='size')
def trains(run):
    """trainIds contained in the run
    """
    return np.array(run.train_ids)

@Variable(title='Pulses', summary='max')
def pulses(run):
    """Number of pulses
    """
    return run.alias['pulses'].xarray()


### VISAR

VISAR_CAL_FILE = './visar_calibration_values.toml'
KEPLER1 = None
KEPLER2 = None

@Variable(title='KEPLER 1')
def kepler1(run):
    global KEPLER1
    KEPLER1 = VISAR(run, 'KEPLER1', VISAR_CAL_FILE)
    return KEPLER1.shot()

@Variable(title='KEPLER 1 PLOT')
def kepler1_plot(run, _: 'var#kepler1'):
    return KEPLER1.plot()

@Variable(title='KEPLER 2')
def kepler2(run):
    global KEPLER2
    KEPLER2 = VISAR(run, 'KEPLER2', VISAR_CAL_FILE)
    return KEPLER2.shot()

@Variable(title='KEPLER 2 PLOT')
def kepler2_plot(run, _: 'var#kepler2'):
    return KEPLER2.plot()
