# File:       FocusedTxData.py
# Author:     SunBaolin
# Created on: 2025-02-18

import os
import json
import numpy as np

from scipy.fft import fft, ifft
from scipy.signal import hilbert


class FocusedTxData:
    """ A template class that contains the focused transmit data.

    FocusedTxData is a container or dataclass that holds all of the information describing
    a focused transmit acquisition. Users should create a subclass that reimplements
    __init__() according to how their data is stored.

    The required information is:
    idata       In-phase (real) data with shape (nxmits, nchans, nsamps)
    qdata       Quadrature (imag) data with shape (nxmits, nchans, nsamps)
    tx_ori      List of transmit origins with shape (N,3) [m]
    tx_dir      List of transmit directions with shape (N,2) [radians]
    ele_pos     Element positions with shape (N,3) [m]
    fc          Center frequency [Hz]
    fs          Sampling frequency [Hz]
    fdemod      Demodulation frequency, if data is demodulated [Hz]
    c           Speed of sound [m/s]
    time_zero   List of time zeroes for each acquisition [s]

    Correct implementation can be checked by using the validate() method.
    """

    def __init__(self):
        """ Users must re-implement this function to load their own data. """
        # Do not actually use FocusedTxData.__init__() as is.
        raise NotImplementedError

        # We provide the following as a visual example for a __init__() method.
        nxmits, nchans, nsamps = 2, 3, 4
        # Initialize the parameters that *must* be populated by child classes.
        self.idata = np.zeros((nxmits, nchans, nsamps), dtype="float32")
        self.qdata = np.zeros((nxmits, nchans, nsamps), dtype="float32")
        self.tx_ori = np.zeros((nxmits, 3), dtype="float32")
        self.tx_dir = np.zeros((nxmits, 2), dtype="float32")
        self.ele_pos = np.zeros((nchans, 3), dtype="float32")
        self.fc = 5e6
        self.fs = 20e6
        self.fdemod = 0
        self.c = 1540
        self.time_zero = np.zeros((nxmits,), dtype="float32")

    def validate(self):
        """ Check to make sure that all information is loaded and valid. """
        # Check size of idata, qdata, tx_ori, tx_dir, ele_pos
        assert self.idata.shape == self.qdata.shape
        assert self.idata.ndim == self.qdata.ndim == 3
        nxmits, nchans, nsamps = self.idata.shape
        assert self.tx_ori.shape == (nxmits, 3)
        assert self.tx_dir.shape == (nxmits, 2)
        assert self.ele_pos.shape == (nchans, 3)
        # Check frequencies (expecting more than 0.1 MHz)
        assert self.fc > 1e5
        assert self.fs > 1e5
        assert self.fdemod > 1e5 or self.fdemod == 0
        # Check speed of sound (should be between 1000-2000 for medical imaging)
        assert 1000 <= self.c <= 2000
        # Check that a separate time zero is provided for each transmit
        # assert self.time_zero.ndim == 1 and self.time_zero.size == nxmits
        assert self.tstart.ndim == 1 and self.tstart.size == nxmits



class WUSData(FocusedTxData):
    def __init__(self, config_file):

        with open(config_file, "r") as file:
            probe_params = json.load(file)

        self.element_num = probe_params["element_num"]
        self.fc = probe_params["fc"]
        self.fs = probe_params["fs"]
        self.c = probe_params["c"]
        self.pitch = probe_params["pitch"]
        self.width = probe_params["width"]
        self.kerf = probe_params["kerf"]
        self.nxmits = probe_params["tx_line_num"]
        self.rx_line_num = probe_params["rx_line_num"]
        self.sample_num = probe_params["sample_num"]
        self.fdemod = probe_params["fdemod"]
        self.z_focus = probe_params["z_focus"]

        self.ele_pos = np.zeros((self.element_num, 3), dtype="float32")
        self.ele_pos[:, 0] = np.arange(self.element_num) * self.pitch
        self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])

        self.tx_ori = np.zeros((self.nxmits, 3), dtype="float32")
        self.tx_ori[:, 0] = np.arange(self.nxmits) * self.pitch * 64.0 / self.nxmits
        self.tx_ori[:, 0] -= np.mean(self.tx_ori[:, 0])

        self.rx_ori = np.zeros((self.element_num, 3), dtype="float32")
        self.rx_ori[:, 0] = np.arange(self.element_num) * self.pitch * 64.0 / self.element_num
        self.rx_ori[:, 0] -= np.mean(self.rx_ori[:, 0])
    
        self.tstart = np.ones((self.nxmits,), dtype="float32") * 3e-6

        self.tx_foc = np.ones((self.nxmits,), dtype="float32") * self.z_focus

        self.tx_dir = np.zeros((self.nxmits, 2), dtype="float32")
        self.rx_dir = np.zeros((self.element_num, 2), dtype="float32")

        self.idata = np.zeros((self.nxmits, self.element_num, self.sample_num), dtype="float32")
        self.qdata = np.zeros((self.nxmits, self.element_num, self.sample_num), dtype="float32")

        self.display_params()
        self.validate()


    def load_data(self, data):

        for n in range(self.nxmits):
            for i in range(self.element_num):
                data[n, i, :] = self.bandpass_filter_rf_data(data[n, i, :], data.shape[2], 25e6, 1.0e6, 5.0e6)

        iqdata = hilbert(data, axis=-1)
        self.idata = np.real(iqdata)
        self.qdata = np.imag(iqdata)


    # 带通滤波器
    def bandpass_filter_rf_data(self, x, length, sampling_frequency, low_cutoff, high_cutoff):
        size = int(length)

        fft_result = fft(x)

        df = sampling_frequency / size  # 计算频率分辨率

        # 创建频率数组
        frequencies = np.arange(size) * df

        # 应用带通滤波器
        mask = (frequencies >= low_cutoff) & (frequencies <= high_cutoff)
        fft_result[~mask] = 0.0  # 将不在范围内的频率分量置零

        res = ifft(fft_result)

        return np.real(res)
    
    def display_params(self):
        print("Ultrasound System Parameters:")
        print(f"Element Num: {self.element_num}")
        print(f"Pitch: {self.pitch} m")
        print(f"width: {self.width} m")
        print(f"kerf: {self.kerf} m")
        print(f"Central Frequency: {self.fc} Hz")
        print(f"Sampling Frequency: {self.fs} Hz")
        print(f"Speed of Sound: {self.c} m/s")
        print(f"Transmit Events: {self.nxmits}")
        print(f"Sample Num: {self.sample_num}")
        print(f"Rx Line Num: {self.rx_line_num}")
        print(f"Demodulation Frequency: {self.fdemod} Hz")
