# File:       das_torch.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-03-09
import math
import torch
from tqdm import tqdm
from torch.nn.functional import grid_sample

PI = 3.14159265359

class DAS_FT(torch.nn.Module):
    """ PyTorch implementation of DAS focused transmit beamforming.

    This class implements DAS focused transmit beamforming as a neural network via a
    PyTorch nn.Module. Subclasses derived from this class can choose to make certain
    parameters trainable. All components can be turned into trainable parameters.
    """

    def __init__(
        self, F, grid, rxfnum=2, dtype=torch.float, device=torch.device("cuda:0")
    ):
        """ Initialization method for DAS_FT.

        All inputs are specified in SI units, and stored in self as PyTorch tensors.
        INPUTS
        F           A FocusedData object that describes the acquisition
        grid        A [ncols, nrows, 3] numpy array of the reconstruction grid
        rxfnum      The f-number to use for receive apodization
        dtype       The torch Tensor datatype (defaults to torch.float)
        device      The torch Tensor device (defaults to GPU execution)
        """
        super().__init__()

        # Convert focused transmit data to tensors
        self.tx_ori = torch.tensor(F.tx_ori, dtype=dtype, device=device)
        self.rx_ori = torch.tensor(F.rx_ori, dtype=dtype, device=device)
        self.tx_dir = torch.tensor(F.tx_dir, dtype=dtype, device=device)
        self.rx_dir = torch.tensor(F.rx_dir, dtype=dtype, device=device)
        self.ele_pos = torch.tensor(F.ele_pos, dtype=dtype, device=device)
        self.fc = torch.tensor(F.fc, dtype=dtype, device=device)
        self.fs = torch.tensor(F.fs, dtype=dtype, device=device)
        self.fdemod = torch.tensor(F.fdemod, dtype=dtype, device=device)
        self.c = torch.tensor(F.c, dtype=dtype, device=device)
        self.tstart = torch.tensor(F.tstart, dtype=dtype, device=device)

        self.rx_line_num = F.rx_line_num

        # Convert grid to tensor
        self.grid = torch.tensor(grid, dtype=dtype, device=device)
        self.out_shape = grid.shape[:-1]

        # Store other information as well
        self.dtype = dtype
        self.device = device
        self.rxfnum = torch.tensor(rxfnum)

    def forward(self, x):
        """ Forward pass for DAS_FT neural network.

        """
        idata, qdata = x
        dtype, device = self.dtype, self.device
        nxmits, nelems, nsamps = idata.shape
        parallel_beam = self.rx_line_num / nxmits

        nx, nz = self.grid.shape[:2]

        idas = torch.zeros((nx, nz), dtype=dtype, device=device)
        qdas = torch.zeros((nx, nz), dtype=dtype, device=device)

        # Loop through all transmits
        for t in tqdm(range(self.rx_line_num)):
            data_line = math.floor(t / parallel_beam)
            txdel = torch.norm(self.grid[t] - self.rx_ori[t].unsqueeze(0), dim=-1)
            rxdel = delay_focus(self.grid[t].view(-1, 1, 3), self.ele_pos).T
            delays = ((txdel + rxdel) / self.c - self.tstart[data_line]) * self.fs
            # Grab data from t-th transmit (N, C, H_in, W_in)
            iq = torch.stack((idata[data_line], qdata[data_line]), axis=0).unsqueeze(0)
            # Convert delays to be used with grid_sample (N, H_out, W_out, 2)
            dgsz = (delays.unsqueeze(0) * 2 + 1) / idata.shape[-1] - 1
            dgsx = torch.arange(nelems, dtype=dtype, device=device)
            dgsx = ((dgsx * 2 + 1) / nelems - 1).view(1, -1, 1)
            dgsx = dgsx + 0 * dgsz  # Match shape to dgsz via broadcasting
            dgs = torch.stack((dgsz, dgsx), axis=-1)
            ifoc, qfoc = grid_sample(iq, dgs, align_corners=False)[0]

            # Apply phase-rotation if focusing demodulated data
            if self.fdemod != 0:
                tshift = delays / self.fs - self.grid[[t], :, 2] * 2 / self.c
                theta = 2 * PI * self.fdemod * tshift
                ifoc, qfoc = _complex_rotate(ifoc, qfoc, theta)

            # Compute apodization
            apods = apod_focus(self.grid[t], self.ele_pos, fnum=self.rxfnum)

            # Apply apodization, reshape, and add to running sum
            ifoc *= apods
            qfoc *= apods
            idas[t] = ifoc.sum(axis=0, keepdim=False)
            qdas[t] = qfoc.sum(axis=0, keepdim=False)

        return idas, qdas


## Compute distance to user-defined pixels from elements
# Expects all inputs to be torch tensors specified in SI units.
# INPUTS
#   grid    Pixel positions in x,y,z    [npixels, 3]
#   ele_pos Element positions in x,y,z  [nelems, 3]
# OUTPUTS
#   dist    Distance from each pixel to each element [nelems, npixels]
def delay_focus(grid, ele_pos):
    # Get norm of distance vector between elements and pixels via broadcasting
    dist = torch.norm(grid - ele_pos.unsqueeze(0), dim=-1)
    # Output has shape [nelems, npixels]
    return dist


## Compute rect apodization to user-defined pixels for desired f-number
# Expects all inputs to be torch tensors specified in SI units.
# INPUTS
#   grid        Pixel positions in x,y,z        [npixels, 3]
#   ele_pos     Element positions in x,y,z      [nelems, 3]
#   fnum        Desired f-number                scalar
#   min_width   Minimum width to retain         scalar
# OUTPUTS
#   apod    Apodization for each pixel to each element  [nelems, npixels]
def apod_focus(grid, ele_pos, fnum=1, min_width=1e-3):
    # Get vector between elements and pixels via broadcasting
    ppos = grid.unsqueeze(0)
    epos = ele_pos.view(-1, 1, 3)
    v = ppos - epos
    
    # 动态孔径
    # Select (ele,pix) pairs whose effective fnum is greater than fnum
    mask = torch.abs(v[:, :, 2] / v[:, :, 0]) >= fnum

    # 保留的最小孔径大小
    mask = mask | (torch.abs(v[:, :, 0]) <= min_width)

    # win = torch.hamming_window(mask.shape[0]).unsqueeze(1)
    # win = win.repeat(1, mask.shape[1])
    # win = win.to('cuda')
    # mask = mask * win

    hamming_matrix = torch.zeros_like(mask, dtype=torch.float32)
    for col in range(grid.shape[0]):
        active_indices = torch.where(mask[:, col])[0]  # 获取当前列 `True` 的索引
        active_count = len(active_indices)

        if active_count > 1:
            hann_win = torch.hamming_window(active_count, device=torch.device("cuda:0"))  # 生成该列所需的 Hanning 窗
            hamming_matrix[active_indices, col] = hann_win  # 将 Hanning 窗填充到 `True` 位置
        elif active_count == 1:  # 只有 1 个 True，直接赋值 1
            hamming_matrix[active_indices, col] = 1.0

    mask = mask * hamming_matrix

    # Also account for edges of aperture
    # mask = mask | ((v[:, :, 0] >= min_width) & (ppos[:, :, 0] <= epos[0, 0, 0]))
    # mask = mask | ((v[:, :, 0] <= -min_width) & (ppos[:, :, 0] >= epos[-1, 0, 0]))

    # Convert to float and normalize across elements (i.e., delay-and-"average")
    apod = mask.float()
    # apod /= torch.sum(apod, 0, keepdim=True)
    # Output has shape [nelems, npixels]
    return apod


## Simple phase rotation of I and Q component by complex angle theta
def _complex_rotate(I, Q, theta):
    Ir = I * torch.cos(theta) - Q * torch.sin(theta)
    Qr = Q * torch.cos(theta) + I * torch.sin(theta)
    return Ir, Qr

