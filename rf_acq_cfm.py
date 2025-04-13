import usb.core
import usb.util
import usb.backend
import os
import time
import shutil
import threading
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# =================================================================
def byte_to_double(data):
    if len(data)%2 != 0:
        print("ERROR!!!!")
        return
    dobule_bytes = []
    for i in range(0,len(data),2):
        dobule_bytes.append(data[i] + data[i+1]*256)
    return dobule_bytes

def process_data(data):
    mid_index = len(data) // 2
    first32 = data[:mid_index]
    second32 = data[mid_index:]
    first32 = np.array(first32)
    second32 = np.array(second32)
    first32 = np.reshape(first32, (-1, 16, 10, 4096, 32))
    second32 = np.reshape(second32, (-1, 16, 10, 4096, 32))

    data = np.concatenate((first32, second32), axis=4)

    print(data.shape)
    frame = data.shape[0]
    angle = data.shape[1]
    fprf = data.shape[2]

    rfdata_path = 'rfdata'
    if os.path.exists(rfdata_path):
        shutil.rmtree(rfdata_path)
    os.makedirs(rfdata_path)

    for i in tqdm(range(frame)):
        for j in range(angle):
            for k in range(fprf):
                rfdata = data[i, j, k, :, :]
                np.savetxt(os.path.join('rfdata', f"rfdata_{i+1}_{j+1}_{k+1}.csv"), rfdata, delimiter=",", fmt="%d")

    return data

# =================================================================




dev = usb.core.find(idVendor=0x0424, idProduct=0x4940)
# print(dev)
dev.set_configuration()
cfg = dev.get_active_configuration()
intf = cfg[(0,0)]
ep = usb.util.find_descriptor(intf,  custom_match = lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT)
print(ep)

read_length = 4096
ready_data = [0xef,0x01,0x10,0x00]
ready_data[1] = 0x01

total_cnt = 0



# 
ready_data[1] = 0x01
dev.write(0x1, ready_data, timeout=1000)

recv_data = []
recv_flag = True
cnt = 0
start_time = time.time()
while(recv_flag):
    data = dev.read(0x81, read_length, timeout=20000)  

    if data[2] == 0x01 and data[3] == 0x01 and data[1] == 0xef:
        recv_flag = False
        total_cnt += 1

        # if total_cnt == 32:
        end_time = time.time()
        print("usb_speed:{0} kbytes/s".format(4096*cnt*total_cnt/((end_time-start_time)*1024)))
        total_cnt = 0
        start_time = end_time

        process_data(recv_data)

        cnt = 0
    else:
        recv_data.extend(byte_to_double(data))
        print(cnt)
        cnt += 1





usb.util.dispose_resources(dev)