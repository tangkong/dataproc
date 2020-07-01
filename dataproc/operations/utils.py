"""
Basic utility operations

Includes:
regex file search
grab files in folder
grab single file


"""
from pathlib import Path

def single_select(path: str='') -> Path: 
    # Check existence?
    # Check end 
    return Path(path)

def folder_select(path: str='', filter: str='*') -> list:
    fpath = Path(path)
    flist = fpath.glob(filter)
    return list(flist)