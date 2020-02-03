from pathlib import Path
from dataproc.operations.hitp import summarize_params

expPath = Path('C:\\Users\\roberttk\\Desktop\\SLAC_RA\\dataproc\\fstore\\export')



summarize_params(expPath, '*_derived_params.csv', 'derived_summary.csv')
