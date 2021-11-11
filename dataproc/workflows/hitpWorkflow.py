import time
import json
from pathlib import Path
import re
import pyFAI
import pyFAI.detectors as dets
import numpy as np

from ..operations.hitp import (create_single, load_image, create_scan, save_Itth, save_qchi, 
                                integrate_1d_mg, fit_peak, save_dict, save_curve_fit, 
                                bkgd_sub, summarize_params)
from ..operations.utils import single_select, folder_select

def hitpWorkflow(configPath):
    # Configuration setup
    # Grab configs, using calibration from 1-5, 11-2021
    print(configPath)
    with open(configPath) as jp:
        cfg = json.load(jp)

    # Fill out experimental information
    expInfo = cfg['expInfo']
    
    det = pyFAI.detector_factory(expInfo['detector'])
    mask = det.mask
    print('Experimental Info used: \n')
    print(expInfo)

    # Pull out Fit info
    fitInfo = cfg['fitInfo']

    print(' =========== Starting processing script =========== ')
    # Process image files
    imList = folder_select(Path(cfg['imagePath']), f'*{cfg["searchString"]}*.tiff')
    ai = create_single(expInfo=expInfo)
    template = cfg['searchString']
    for im in imList:
        
        int1d = ai.integrate1d(im, 2000, mask=det.mask)
        # restrict range?
        subx, suby = int1d[0], int1d[1]
        # Background subtract/move to zero
        suby = suby - np.min(suby)
        subx, suby = bkgd_sub(subx, suby)

        # segment range into list of blocks
        xList = []
        yList = []
        bnds = expInfo['blockBounds']
        for leftBnd in range(len(bnds) - 1): # indexes
            xList.append(subx[np.where((subx >= bnds[leftBnd]) & (subx < bnds[leftBnd + 1]))])
            yList.append(suby[np.where((subx >= bnds[leftBnd]) & (subx < bnds[leftBnd + 1]))])
        
        for xbit, ybit, i in zip(xList, yList, range(len(xList))):
            # Restrict range and fit peaks
            curveParams, derivedParams = fit_peak(xbit, ybit,
                                peakShape=fitInfo['peakShape'],
                                fitMode=fitInfo['fitMode'],
                                numCurves=fitInfo['numCurves'])
            print(f'    ----Saving data for block between {np.min(xbit):.2f} - {np.max(xbit):.2f}')
            # output/saving of blocks
            save_dict(curveParams, cfg['exportPath'], template + f'_block{i}_curve')
            save_dict(derivedParams, cfg['exportPath'], template + f'_block{i}_derived')
            save_curve_fit(xbit, ybit, curveParams, cfg['exportPath'], 
                            template + f'_block{i}', peakShape=fitInfo['peakShape'])
        
        # Save overall images
        print(f'    ----Saving whole scan data')
        save_qchi(ai, im, mask, cfg['exportPath'], template + f'_block{i}')
        save_Itth(ai, im, mask, cfg['exportPath'], template +  f'_block{i}')