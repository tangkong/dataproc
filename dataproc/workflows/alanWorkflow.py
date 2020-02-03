import time
import json
from pathlib import Path
import re
import pyFAI.detectors as dets
import numpy as np

from ..operations.hitp import (load_image, create_scan, save_Itth, save_qchi, 
                                integrate_1d_mg, fit_peak, save_dict, save_curve_fit, 
                                summarize_params)
from ..operations.utils import single_select, folder_select

def alanWorkflow():
    # Configuration setup
    # Grab configs
    configPath = 'C:\\Users\\roberttk\\Desktop\\SLAC_RA\\dataProc\\dataproc\\workflows\\alanConfig'
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
    

    # Pull out Fit info
    fitInfo = cfg['fitInfo']

    # Start processing loop =======================================================
    run_enable = True

    # cache names of processed spec files
    processed = []
    while run_enable:
        print('Checking for new spec files')
        # Gather spec files
        specPath = Path(cfg['specPath'])
        specList = folder_select(specPath, '*scan1.csv')

        # process unseen spec files
        for s in specList:
            if s in processed:
                continue
            else:
                # Continue processing
                # Cache spec file
                processed.append(s)

                # Grab image files
                # ----------- Customize condition for identifying image files ---------
                template = s.stem  # Grabs file name without extension
                # -----------------------------------------------------------------

                # Process image files
                imList = folder_select(Path(cfg['imagePath']), f'*{template}*.raw')
                mg, ims = create_scan(imList, s, expInfo=expInfo)
                
                # Mask image.... This is bad here ----------------------------------
                mask = np.ones(np.shape(ims[0]))
                mask[100:400, :] = 0
                # ------------------------------------------------------------------

                int1d = integrate_1d_mg(mg, ims, mask=mask)
                
                # restrict range?
                subx, suby = int1d[0], int1d[1]

                # Restrict range and fit peaks
                curveParams, derivedParams = fit_peak(subx, suby,
                                    peakShape=fitInfo['peakShape'],
                                    fitMode=fitInfo['fitMode'],
                                    numCurves=fitInfo['numCurves'])

                # output/saving 
                save_qchi(mg, ims, mask, cfg['exportPath'], template)
                break
                save_Itth(mg, ims, mask, cfg['exportPath'], template)
                save_dict(curveParams, cfg['exportPath'], template  + '_curve')
                save_dict(derivedParams, cfg['exportPath'], template+'_derived')
                save_curve_fit(subx, suby, curveParams, cfg['exportPath'], 
                                template, peakShape=fitInfo['peakShape'])
                
        # Gather and summarize
        summarize_params(cfg['exportPath'], '*_derived_params.csv', '_derived_summary.csv')
        summarize_params(cfg['exportPath'], '*_curve_params.csv', '_curve_summary.csv')


        # run conditions
        t=300
        print(f'sleeping for {t} seconds')
        time.sleep(t)
        run_enable = False
