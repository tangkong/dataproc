import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyFAI.detectors as dets
import pyFAI

from pathlib import Path

import glob

from dataproc.operations import hitp

plt.rcParams["figure.figsize"]=(10, 8)

import time
import json
from pathlib import Path
import re
import pyFAI.detectors as dets
import numpy as np


from dataproc.operations.utils import single_select, folder_select
from scipy.ndimage.filters import gaussian_filter as gf
import pyFAI
template = ''
configPath = "/home/b_spec/roberttk/dataproc/dataproc/workflows/hitpConfig"

# Configuration setup
# Grab configs
print(configPath)
with open(configPath) as jp:
    cfg = json.load(jp)


def workflow(y, boundaries, downsample_int = 10, noise_estimate = None, background = None,
             **kwargs):
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
    
    if background is None:
        # Background subtract/move to zero
        suby = suby - np.min(suby)
        subx, suby = hitp.bkgd_sub(subx, suby, downsample_int)
    else:
        suby = y - background
        if suby.min() < 0:
            print('negative values in background-subtracted pattern. taking absolute value.')
            suby = suby - suby.min()

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

def fit_curves(y, **kwargs):
    boundaries = hitp.bayesian_block_finder(x, gf(y, 1.5))
    #boundaries = [b for b in boundaries if b >= boundaries_min and b <= boundaries_max]
    print(boundaries)
    cfg['fitInfo']['blockBounds'] = boundaries
    suby = workflow(y, boundaries, **kwargs)
    return suby

def curvefit_2d(patterns, background = None, noise_estimate = None, **kwargs):
    def _background(i):
        if background is not None:
            return background[i]
        return None
    def _noise_estimate(i):
        if noise_estimate is not None:
            return noise_estimate[i]
        return None
    return np.vstack([fit_curves(y, background = _background(i),
                                 noise_estimate = _noise_estimate(i), **kwargs)
                      for i, y in enumerate(patterns)])

df = pd.read_csv("YijinXRD.dat", sep = '\t')
qq = df.iloc[:, 0]

patterns = df.iloc[:, 2:]
patterns = patterns.values.T
for i in range(len(patterns)):
    patterns[i] = patterns[i] - i * 1000
    
x = np.arange(len(patterns[0])) + 1
y = patterns[0]
    
boundaries = hitp.bayesian_block_finder(x, y)

from dataproc.operations import source_separation as sep

# cutoff gives the standard deviation of the low-pass Gaussian kernel standard deviation,
#     as a fraction of the number of samples
# Threshold is a percentile threshold for classifiying points as 'background' or 'peak' in
#     the context of background interpolation
# smooth_q should be set to roughly the diffraction peak FWHM / 2.355
background = sep.get_background(patterns, threshold = 25, smooth_q = 1.7, method = 'simple')
slow_q, fast_q, slow_T, fast_T = sep.separate_signal(patterns, cutoff = .25,  threshold = 25, smooth_q = 1.7)


# Fit one pattern
i = 33
y_fsub_stop = fit_curves(patterns[i], background = background[i],
                         noise_estimate = fast_T[i], stdratio_threshold = 2)
