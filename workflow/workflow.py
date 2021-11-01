import time
import json
from pathlib import Path
import re
import pyFAI.detectors as dets
import numpy as np


# from dataproc.operations.hitp import (load_image, create_scan, save_Itth, save_qchi, 
#                                 integrate_1d_mg, fit_peak, save_dict, save_curve_fit, 
#                                 bkgd_sub, summarize_params)
from dataproc.operations.utils import single_select, folder_select
from scipy.ndimage.filters import gaussian_filter as gf

template = ''
configPath = "workflows/alanConfig"

# Configuration setup
# Grab configs
print(configPath)
with open(configPath) as jp:
    cfg = json.load(jp)
cfg['fitInfo']['blockBounds'] = boundaries




def fit_curves(y, **kwargs):
    boundaries = hitp.bayesian_block_finder(x, gf(y, 1.5))
    #boundaries = [b for b in boundaries if b >= boundaries_min and b <= boundaries_max]
    print(boundaries)
    cfg['fitInfo']['blockBounds'] = boundaries
    suby = workflow(y, boundaries, **kwargs)
    return suby

def workflow(y, boundaries, downsample_int = 10, noise_estimate = None, **kwargs):
    """
    kwargs are passed to hitp.fit_peak
    """

    # Fill out experimental information
    expInfo = {}


    expInfo['blockBounds'] = cfg['fitInfo']['blockBounds']

    print('Experimental Info used: \n')
    print(expInfo)

    # Pull out Fit info
    fitInfo = cfg['fitInfo']

    # Start processing loop =======================================================
    run_enable = True
    
    # restrict range?
    subx, suby = np.arange(len(y)) + 1, y
    #pdb.set_trace()
    # Background subtract/move to zero
    suby = suby - np.min(suby)
    subx, suby = hitp.bkgd_sub(subx, suby, downsample_int)

    # segment rangeinto two...
    xList = []
    yList = []
    noiseList = []
    bnds = expInfo['blockBounds']
    for leftBnd in range(len(bnds) - 1): # indexes
        selector = np.where((subx >= bnds[leftBnd]) & (subx < bnds[leftBnd + 1]))
        xList.append(subx[selector])
        yList.append(suby[selector])
        if noise_estimate is not None:
            noiseList.append(noise_estimate[selector] + 1e-9) 
        else:
            noiseList.append(None)
    for i, (xbit, ybit, noisebit) in enumerate(zip(xList, yList, noiseList)):
        # Restrict range and fit peaks
        curveParams, derivedParams = hitp.fit_peak(xbit, ybit,
                            peakShape=fitInfo['peakShape'],
                            fitMode=fitInfo['fitMode'],
                            numCurves=fitInfo['numCurves'],
                            noise_estimate = noisebit,
                                             **kwargs)
        print(f'    ----Saving data for block between {np.min(xbit):.2f} - {np.max(xbit):.2f}')
        # output/saving of blocks
        hitp.save_dict(curveParams, cfg['exportPath'], template + f'_block{i}_curve')
        hitp.save_dict(derivedParams, cfg['exportPath'], template + f'_block{i}_derived')
        hitp.save_curve_fit(xbit, ybit, curveParams, cfg['exportPath'], 
                        template + f'_block{i}', peakShape=fitInfo['peakShape'])
    return suby
