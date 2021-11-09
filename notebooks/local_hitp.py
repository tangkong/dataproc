from pathlib import Path
import pandas as pd
import numpy as np
import json

config_path = Path('C:\\Users\\roberttk\\Desktop\\SLAC_RA\\dataproc\\dataproc\\workflows\\hitpConfig')

from dataproc.operations.hitp import create_single, files_to_pattern_array, load_image

print(config_path)
with open(config_path) as jp:
    cfg = json.load(jp)

img_path = Path(cfg['filesInfo']['img_path']) 
csv_path = Path(cfg['filesInfo']['csv_path'])
search_string = cfg['filesInfo']['search_string']

df = pd.read_csv(sorted(csv_path.glob(search_string+'primary*'))[0])
im_list = list(img_path.glob(search_string))

im = load_image(im_list[0])
import pyFAI
det = getattr(pyFAI.detectors, cfg['expInfo']['detector'])()
ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator( dist=cfg['expInfo']['dist'],
                    wavelength=cfg['expInfo']['wavelength'], rot2=np.pi/2, rot3=-np.pi/2,
                    poni1=cfg['expInfo']['center1']*1e-4,poni2=cfg['expInfo']['center2']*1e-4,
                    pixel1=1e-4, pixel2=1e-4)
                    # detector=det) # If we have the right detector
mg = pyFAI.multi_geometry.MultiGeometry([ai])
