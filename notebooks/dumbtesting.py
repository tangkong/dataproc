from dataproc.operations.hitp import bayesian_block_finder, bkgd_sub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fpath = 'C:\\Users\\roberttk\\\Desktop\\SLAC_RA\\dataproc\\fstore\\k3_012918_1_24x24_t45b_0001_1D.csv'

df = pd.read_csv(fpath, names=['TwoTheta', 'I'])
df = df.rename(columns=lambda x: x.strip())

x = df['TwoTheta'].values
y = df['I'].values

x, y = bkgd_sub(x, y)

# default y scaling yields a single datapoint (everything is noise)
# max=1000 --> 178 blocks
z = bayesian_block_finder(x, y)
z = z.astype(int)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.vlines(x[z], 0, np.max(y))