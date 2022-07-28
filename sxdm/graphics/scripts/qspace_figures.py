#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 00:12:30 2020

@author: dzhigd
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from tifffile import imsave
import tifffile as tf
from scipy import ndimage, interpolate
from skimage.transform import resize
import matplotlib.mlab as ml
from scipy.optimize import minimize
import sys
sys.path.append('/home/dzhigd/Software/nanomax_tools')

import nanomax_tools.processing.qspace_utils as qu
import nanomax_tools.graphics.graphics_utils as gu

year        = "2020"                                                           #The year for the experiemnt
beamtimeID  ="2020101408"                                                      #The beamtimeID
sample_name = r"0002_sample_P246_AB"                                           #The name for the p10 newfile
scan_number = np.linspace(109,128,20)                                          #The scan numbers, can be the list

q_coordinate_path = r'/home/dzhigd/work/projects/Qdevs_2020_NanoMAX/data/scan_109_128/q_20_220_150.npz'
path_root = r"/media/dzhigd/My Passport/DDzhigaev/Data/MAXIV/NanoMax/%s/process/scan_%d_%d"%(beamtimeID,scan_number[0],scan_number[-1])
qx,qy,qz = qu.load_q_coordinates(q_coordinate_path)

data = np.zeros((20,220,150))
for ii in range(0,len(scan_number)):
    t = np.load(os.path.join(path_root, 'scan_%06d_merlin.npz'%scan_number[ii]));
    
#    data = permute(data,[2,3,1]);
    data_xrd = np.nan_to_num(t['data_xrd'])
    
    data[ii,:,:] = np.sum(data_xrd,axis=0)
    print(ii)

# Temporary transform from matlab code
qx = np.reshape(qx,(150,20,220))
qy = np.reshape(qy,(150,20,220))
qz = np.reshape(qz,(150,20,220))

qx = np.transpose(qx,(1,2,0))
qy = np.transpose(qy,(1,2,0))
qz = np.transpose(qz,(1,2,0))

qx = np.concatenate(np.concatenate(qx))
qy = np.concatenate(np.concatenate(qy))
qz = np.concatenate(np.concatenate(qz))

gu.show_q_space_projections(qx,qy,qz,data)

