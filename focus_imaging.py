import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from datasets.FocusedTxData import WUSData
from beamforming.DAS import DAS_FT, DAS_FT_B
from beamforming.PixelGrid import make_foctx_grid, grid_scan_convert

F = WUSData('configs\phase_array_2.5M.json')

rmax = F.scan_depth
wvln = F.c / F.fc
dr = wvln / 4
rlims = [0, rmax]
grid = make_foctx_grid(rlims, dr, F.rx_ori, F.rx_dir)
scan_convert = True
drange = F.drange
fnum = F.fnum

das = DAS_FT_B(F, grid, rxfnum=fnum)


rfdata = []
for j in range(F.nxmits):
    df = pd.read_csv(f'rfdata/rfdata_1_{j+1}.csv', sep=',', header=None)
    data = df.values
    data = (data - 512) / 512
    data = data.T
    rfdata.append(data)

rfdata = np.array(rfdata)
F.load_data(rfdata)

idata = torch.tensor(F.idata, dtype=torch.float, device=torch.device("cuda:0"))
qdata = torch.tensor(F.qdata, dtype=torch.float, device=torch.device("cuda:0"))
x = (idata, qdata)

idas, qdas = das(x)
idas, qdas = idas.detach().cpu().numpy(), qdas.detach().cpu().numpy()
iq = idas + 1j * qdas
bimg = np.abs(iq).T

# Scan convert
bimg, grid = grid_scan_convert(bimg, grid, rlims, wvln / 2, wvln / 2)


bimg = 20 * np.log10(bimg)  # Log-compress
bimg -= np.amax(bimg)  # Normalize by max value

# Display images via matplotlib
extent = [grid[0, 0, 0], grid[-1, 0, 0], grid[0, -1, 2], grid[0, 0, 2]]
extent = np.array(extent) * 1e3  # Convert to mm
plt.imshow(bimg, vmin=-drange, cmap="gray", extent=extent, origin="upper")
plt.xlabel("Lateral distance [mm]")
plt.ylabel("Axis distance [mm]")
plt.show()