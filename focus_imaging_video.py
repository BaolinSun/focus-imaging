import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import griddata
from datasets.FocusedTxData import WUSData
from beamforming.DAS import DAS_FT, DAS_FT_B
from beamforming.PixelGrid import make_foctx_grid, make_pixel_grid

F = WUSData('configs\phase_array_2.5M.json')

rmax = F.scan_depth
wvln = F.c / F.fc
dr = wvln / 4
rlims = [0, rmax]
scan_grid = make_foctx_grid(rlims, dr, F.rx_ori, F.rx_dir)
scan_convert = True
drange = F.drange
fnum = F.fnum

das = DAS_FT_B(F, scan_grid, rxfnum=fnum)

for i in tqdm(range(18)):
    rfdata = []
    for j in range(F.nxmits):
        df = pd.read_csv(f'rfdata/rfdata_{i+1}_{j+1}.csv', sep=',', header=None)
        data = df.values
        data = (data - 512) / 512
        data = data.T
        rfdata.append(data)

    rfdata = np.array(rfdata)
    F.load_data(rfdata)

    idata = torch.tensor(F.idata, dtype=torch.float, device=torch.device("cuda:0"))
    qdata = torch.tensor(F.qdata, dtype=torch.float, device=torch.device("cuda:0"))
    x = (idata, qdata)

    # start_time = time.time()
    # idas, qdas = das(x)
    # iq = idas + 1j * qdas
    # bimg = torch.abs(iq).T

    # start_time = time.time()
    idas, qdas = das(x)
    idas, qdas = idas.detach().cpu().numpy(), qdas.detach().cpu().numpy()
    iq = idas + 1j * qdas
    bimg = np.abs(iq).T



    # Scan convert if necessary
    if scan_convert:
        xlims = rlims[1] * np.array([-0.7, 0.7])
        zlims = rlims[1] * np.array([0, 1])
        img_grid = make_pixel_grid(xlims, zlims, wvln / 2, wvln / 2)
        grid = np.transpose(scan_grid, (1, 0, 2))
        g1 = np.stack((grid[:, :, 2], grid[:, :, 0]), -1).reshape(-1, 2)
        g2 = np.stack((img_grid[:, :, 2], img_grid[:, :, 0]), -1).reshape(-1, 2)
        bsc = griddata(g1, bimg.reshape(-1), g2, "linear", 1e-10)
        bimg = np.reshape(bsc, img_grid.shape[:2])
        grid = img_grid.transpose(1, 0, 2)

    # print(time.time() - start_time)

    bimg = 20 * np.log10(bimg)  # Log-compress
    bimg -= np.amax(bimg)  # Normalize by max value

    # Display images via matplotlib
    extent = [grid[0, 0, 0], grid[-1, 0, 0], grid[0, -1, 2], grid[0, 0, 2]]
    extent = np.array(extent) * 1e3  # Convert to mm
    plt.imshow(bimg, vmin=-drange, cmap="gray", extent=extent, origin="upper")
    plt.axis('off')
    plt.savefig(os.path.join('usimage', f"./usImage{i+1}.jpg"), bbox_inches='tight', pad_inches=0)
    # plt.show()