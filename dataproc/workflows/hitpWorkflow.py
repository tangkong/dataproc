import time
import json
from pathlib import Path
import re
import pyFAI
import pyFAI.detectors as dets
import numpy as np
import matplotlib.pyplot as plt

from dataproc.operations.hitp import (create_single, load_image, create_scan, save_Itth, save_qchi, 
                                integrate_1d_mg, fit_peak, save_dict, save_curve_fit, 
                                bkgd_sub, summarize_params, bayesian_block_finder)
from dataproc.operations.utils import single_select, folder_select

def hitpWorkflow(configPath, downsample_int=10, noise_estimate=None, background=None, **kwargs):
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
    base_template = cfg['searchString']
    for im in imList:
        print(im)
        data_pt = re.search('(\d+).tiff', str(im))[1]
        template = base_template +'-'+ str(data_pt)
        img = load_image(im) 
        new_mask = det.mask | (img > 65500)
        int1d = ai.integrate1d(img, 2000, mask=new_mask, unit='q_A^-1')
        # restrict range?
        subx, suby = int1d[0], int1d[1]
        # Background subtract/move to zero
        suby = suby - np.min(suby)
        subx, suby = bkgd_sub(subx, suby)

        # segment range into list of blocks
        xList = []
        yList = []
        noiseList = []
        bnds = bayesian_block_finder(subx, suby)
        print(bnds)
        for leftBnd in range(len(bnds) - 1): # indexes
            # bnds are in index space
            selector = np.where((subx >= subx[int(bnds[leftBnd])]) & 
                                  (subx < subx[int(bnds[leftBnd + 1])]))
            xList.append(subx[selector])
            yList.append(suby[selector])
            if noise_estimate is not None:
                noiseList.append(noise_estimate[selector] + 1e-9) 
            else:
                noiseList.append(None)
        print(f'begin peak fitting for data point {data_pt}')
        
        for i, (xbit, ybit, noisebit) in enumerate(zip(xList, yList, noiseList)):
            if not (len(xbit)>0):
                print('skipping a block with nothing in it?')
                continue
            # Restrict range and fit peaks
            curveParams, derivedParams = fit_peak(xbit, ybit,
                                peakShape=fitInfo['peakShape'],
                                fitMode=fitInfo['fitMode'],
                                numCurves=fitInfo['numCurves'],
                                noise_estimate = noisebit
                                                 )
            print(f'    ----Saving data for block between {np.min(xbit):.2f} - {np.max(xbit):.2f}')
            # output/saving of blocks
            save_dict(curveParams, cfg['exportPath'], template + f'_block{i}_curve')
            save_dict(derivedParams, cfg['exportPath'], template + f'_block{i}_derived')
            save_curve_fit(xbit, ybit, curveParams, cfg['exportPath'], 
                            template + f'_block{i}', peakShape=fitInfo['peakShape'])
    
        # Save overall images
        print(f'    ----Saving whole scan data')
        save_qchi(ai, img, new_mask, cfg['exportPath'], template)
        save_Itth(ai, img, new_mask, cfg['exportPath'], template)
        

        

if __name__ == '__main__':
    hitpWorkflow('/home/b_spec/roberttk/dataproc/dataproc/workflows/hitpConfig')
