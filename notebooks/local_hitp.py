from pathlib import Path
import pandas as pd
import numpy as np
import json

config_path = Path('C:\\Users\\roberttk\\Desktop\\SLAC_RA\\dataproc\\workflows\\hitpConfig')

from dataproc.operations.hitp import create_single, files_to_pattern_array, load_image

print(config_path)
with open(config_path) as jp:
    cfg = json.load(jp)

img_path = Path(cfg['filesInfo']['img_path']) 
csv_path = Path(cfg['filesInfo']['csv_path'])
search_string = cfg['filesInfo']['search_string']

df = pd.read_csv(sorted(csv_path.glob(search_string+'.csv'))[0])
im_list = list(img_path.glob(search_string))

im = load_image(im_list[0])
import pyFAI

ai = create_single(expInfo=cfg['expInfo'])
mg = pyFAI.multi_geometry.MultiGeometry([ai], unit='q_A^-1', radial_range=(0,8))