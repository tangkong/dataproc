import time
import json
from pathlib import Path
import re
import pyFAI.detectors as dets
import numpy as np

from ..operations.hitp import (load_image, create_scan, save_Itth, save_qchi, 
                                integrate_1d_mg, fit_peak, save_dict, save_curve_fit, 
                                bkgd_sub, summarize_params)
from ..operations.utils import single_select, folder_select

def alanWorkflow(configPath):
    # Configuration setup
    # Grab configs
    print(configPath)
    with open(configPath) as jp:
        cfg = json.load(jp)

    # Fill out experimental information
    expInfo = {}
    expInfo['dist'] = cfg['expInfo']['dist']
    expInfo['wavelength'] = cfg['expInfo']['wavelength']
    expInfo['orientation'] = cfg['expInfo']['orientation']

    det = getattr(dets, cfg['expInfo']['detector'])()
    expInfo['detector'] = det
    pxSize = det.pixel1

    expInfo['poni1'] = cfg['expInfo']['center1'] * pxSize
    expInfo['poni2'] = cfg['expInfo']['center2'] * pxSize
    expInfo['blockBounds'] = cfg['fitInfo']['blockBounds']

    print('Experimental Info used: \n')
    print(expInfo)

    # Pull out Fit info
    fitInfo = cfg['fitInfo']

    # Start processing loop =======================================================
    run_enable = True

    # cache names of processed spec files
    processed = []
    currLen = 0
    print(' =========== Starting processing script =========== ')
    print('On keypress, script will process available files')
    while run_enable:
        input('Press enter to continue...')

        print('    ----Checking for new spec files')
        # Gather spec files
        specPath = Path(cfg['specPath'])
        specList = folder_select(specPath, '*scan1.csv')


        # process unseen spec files
        for s in specList:
            if not specList: # is empty:
                print('no spec files found')
                break 
            if s in processed:
                continue
            else:
                # Continue processing
                # Cache spec file
                processed.append(s)

                print(f'    ----Processing: {s}')
                # Grab image files
                # ----------- Customize condition for identifying image files ---------
                template = s.stem  # Grabs file name without extension
                # -----------------------------------------------------------------

                # Process image files
                imList = folder_select(Path(cfg['imagePath']), f'*{template}*.raw')
                mg, ims = create_scan(imList, s, expInfo=expInfo)
                print(np.shape(ims))
                # Mask image.... This is bad here ----------------------------------
                mask = np.ones(np.shape(ims[0]))
                mask[100:400, :] = 0
                # ------------------------------------------------------------------

                int1d = integrate_1d_mg(mg, ims, mask=mask)
                
                # restrict range?
                subx, suby = int1d[0], int1d[1]
                # Background subtract/move to zero
                suby = suby - np.min(suby)
                subx, suby = bkgd_sub(subx, suby)

                # segment rangeinto two...
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
                save_qchi(mg, ims, mask, cfg['exportPath'], template + f'_block{i}')
                save_Itth(mg, ims, mask, cfg['exportPath'], template +  f'_block{i}')
        newLen = len(processed)
        
        if currLen < newLen:
            # Gather and summarize
            print('summarizing information...')
            summarize_params(cfg['exportPath'], '*_derived_params.csv', '_derived_summary.csv')
            summarize_params(cfg['exportPath'], '*_curve_params.csv', '_curve_summary.csv')

        currLen = newLen
