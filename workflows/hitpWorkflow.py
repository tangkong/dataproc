import time
import json
from pathlib import Path
import re
import pyFAI
import pyFAI.detectors as dets
import numpy as np

from scipy.ndimage.filters import gaussian_filter as gf

from dataproc.operations.hitp import (load_image, create_scan, save_Itth, save_qchi, 
                                integrate_1d_mg, fit_peak, save_dict, save_curve_fit, 
                                bkgd_sub, summarize_params, bayesian_block_finder)
from dataproc.operations.utils import single_select, folder_select

from dataproc.operations import source_separation as sep

# get config
config_path = './hitpConfig'
print(config_path)
with open(config_path) as jp:
    cfg = json.load(jp)


def fit_blocks(y, boundaries, downsample_int=10, noise_estimate=None, 
                background=None, template='', **kwargs):
    """
    kwargs are passed to hitp.fit_peak
    """

    # Pull out Fit info
    fitInfo = cfg['fitInfo']
    
    # restrict range?
    subx, suby = np.arange(len(y)) + 1, y
    
    if background is None:
        # Background subtract/move to zero
        suby = suby - np.min(suby)
        subx, suby = bkgd_sub(subx, suby, downsample_int)
    else:
        suby = y - background
        if suby.min() < 0:
            print('negative values in background-subtracted pattern. taking absolute value.')
            suby = suby - suby.min()

    # segment range into two...
    xList = []
    yList = []
    noiseList = []
    bnds = boundaries
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
        curveParams, derivedParams = fit_peak(xbit, ybit,
                            peakShape=fitInfo['peakShape'],
                            fitMode=fitInfo['fitMode'],
                            numCurves=fitInfo['numCurves'],
                            noise_estimate = noisebit,
                                             **kwargs)
        print(f'    ----Saving data for block between {np.min(xbit):.2f} - {np.max(xbit):.2f}')
        # output/saving of blocks
        save_dict(curveParams, cfg['exportPath'], template + f'_block{i}_curve')
        save_dict(derivedParams, cfg['exportPath'], template + f'_block{i}_derived')
        save_curve_fit(xbit, ybit, curveParams, cfg['exportPath'], 
                        template + f'_block{i}', peakShape=fitInfo['peakShape'])
    return suby

def fit_curves(y, **kwargs):
    boundaries = bayesian_block_finder(y, gf(y, 1.5))
    print(boundaries)
    cfg['fitInfo']['blockBounds'] = boundaries
    suby = fit_blocks(y, boundaries, **kwargs)
    return suby

# kick things off from here
def fullWaferWorkflow(hdr):
    """workflow processes HiTP image data

    Uses ohoidn's fourier background subtraction and noise estimation to 
    Requires spatially related data.  Currently set up for 2D wafer data

    :param hdrs: a single databroker header, containing 2D data
    :type hdrs: iterable, generator
    :param cfg: configuration, defaults to hitpCOnfig
    :type cfg: dictionary, optional
    :return: Configuration settings 
    """
    # get patterns in (xpos, ypos, pattern_length) array
    # access patterns could be databroker...
    patterns = db_to_pattern_array(hdr)

    # or reading files from bluedata from prompt export
    # patterns = files_to_pattern_array()
    
    # get background
    bkgd = sep.get_background(patterns, threshold=25, 
                               smooth_q=1.7, method='simple')
    slow_q, fast_q, slow_T, fast_T = sep.separate_signal(patterns, 
                                cutoff = .25,  threshold = 25, smooth_q = 1.7)

    # Fit one pattern
    i = 33
    y_fsub_stop = fit_curves(patterns[i], background=bkgd[i],
                            noise_estimate=fast_T[i], stdratio_threshold=2)

    # save data from pattern 

    return y_fsub_stop   
    
def db_to_pattern_array(hdr, cfg=cfg, img_key='pilatus1M_image'):
    # returns dataframe with keys like: [s_stage.px, s_stage.px, pilatus1M_image]
    df = hdr.table(fill=True)

    # iterate through images, integrate, generate 1D patterns, add to df
    # Don't modify something you're iterating over, in general.  Apply instead
    det = getattr(dets, cfg['expInfo']['detector'])()
    def _integrate(row, cfg=cfg, det=det):
        img = row[cfg['dbInfo']['imgkey']][0]
        einfo = cfg['expInfo']
        p = pyFAI.AzimuthalIntegrator(wavelength=einfo['wavelength'])
        p.setFit2D(einfo['dist'], einfo['center1'], einfo['center2'],  
                    einfo['tilt'], einfo['rot'], det.pixel1, det.pixel2)
        return p.integrate1d(img, 1000)

    df['qI_data'] = df.apply(_integrate, axis=1)
    
    # format into [x, y, len(pattern)] array.  empty spots are 0
    # assume uniform spacing
    pattern_length = len(df['qI_data'][1])
    xlocs = df['s_stage_px'].unique() #values may not be round, use actual locs
    xlocs.sort()
    ylocs = df['s_stage_py'].unique()
    ylocs.sort()
    patterns = np.zeros((len(xlocs), len(ylocs), pattern_length))

    for i, xl in enumerate(xlocs): # indices
        for j, yl in enumerate(ylocs):
            sub_df = df[(df['s_stage_px']==xl) & (df['s_stage_py']==yl)]
            if len(sub_df) > 0:
                patterns[i, j, :] = sub_df['qI_data'].item() # TODO verify

    return patterns