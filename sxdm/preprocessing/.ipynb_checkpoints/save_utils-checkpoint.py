#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 22:29:50 2020

@author: dzhigd
"""
import os
import scipy.io as scio
import numpy as np
from tifffile import imsave

def save_data(save_path, save_type, save_dictionary):
    if save_type == "mat":  
        scio.savemat(save_path, {'scan': save_dictionary}, do_compression=True, oned_as="column")
        print(('\033[1;32;40m ++ \033[1;37;40m Saved XRD and meta data to %s')%(save_path))
    elif save_type == "npz": 
        np.savez_compressed(save_path, save_dictionary)
        print(('\033[1;32;40m ++ \033[1;37;40m Saved XRD and meta data to %s')%(save_path))
    elif save_type == 'tif':
        imsave(save_path, save_dictionary)
        print(('\033[1;32;40m ++ \033[1;37;40m Saved XRF data to %s')%(save_path))
    else:
        print('\033[1;37;40m -- Unrecognized save type! Do nothing ...')
        
