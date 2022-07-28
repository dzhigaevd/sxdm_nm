#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:25:27 2020

@author: dzhigd
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from tifffile import imsave
import tifffile as tf
from scipy import ndimage, interpolate
from skimage.transform import resize
import matplotlib.mlab as ml
from scipy.optimize import minimize

# Map correction methods
def generate_ref_image(fluo_image, ref_image, scan_pixel_size, orig_pixel_size,tl_offset):
    """
    for the known structure of the sample
    modifies reference image to match pixel size and roi of the scan area
    fluo_path - path to fluorescence image
    reference_path - path to reference image
    scan_pixel_size - [#, #] pixel size in y and x
    reference_pixel_size - [#,#] pixel size of original reference image
    tl_offset - [y, x] offset of the top left corner the scan area with respect to the reference image
    
    returns:
    referene image, matching to the scan area
    mask, zero where there should not be any objects accordint to the design
    """

    scan_size = fluo_image.shape # measured image size in pixels
    padsize = 1000
    beam_footprint = 0.08*2/scan_pixel_size[1]
      
    ref_image[ref_image>0]=255 
    ref_image[ref_image==0]=1
    ref_image_pad = np.pad(ref_image[:,:,0], pad_width=padsize,mode='constant')
    
    #scan range (um)
    scan_range_top_left = ((np.array(tl_offset))/orig_pixel_size).astype(np.int32)+np.array([padsize,padsize])
    scan_range_bottom_right = ((np.array(tl_offset) + np.array(scan_size)*np.array(scan_pixel_size))/np.array(orig_pixel_size)).astype(np.int32)+np.array([padsize,padsize])

    ref_image_cropped = ref_image_pad[scan_range_top_left[0]:scan_range_bottom_right[0],scan_range_top_left[1]:scan_range_bottom_right[1]]
    mask = ref_image_cropped.copy()
    mask[mask<1] = 0
    mask[mask>0] = 255
    mask = resize(mask, scan_size)
    
    ref_image_cropped = resize(ref_image_cropped, scan_size)
    
    smoothed = ndimage.gaussian_filter(ref_image_cropped, beam_footprint)
    return smoothed, mask

def normalize(data):
    """
    normalize data for cross correlation
    """
    return (data - np.mean(data, axis=0)) / (np.std(data, axis=0))

def transform(y, params):
    """
    polynomial transformation of x coordinate into x' and interpolation of the input data y
    params - polynomial coefficients [scaling, scaling gradient]
    returns distorted data y
    """
    x = np.linspace(0,len(y),len(y))
    # generate new x values - xprime by adding a polynomial
    if len(params)==1:
        xprime = x +  params[0]*x 
    if len(params)==2:
        xprime = x +  params[0]*x+ params[1]*x**2
    if len(params)==3:
        xprime = x +  params[0]*x+ params[1]*x**2 + params[2]*x**3
        
    if len(params)==4:
        xprime = x +  params[0]*x+ params[1]*x**2 + params[2]*x**3 + params[3]*x**4
        
    return np.interp(xprime,x,y), xprime

def cross_corr(y1, y2):
    """Calculates the cross correlation and lags without normalization.

    The definition of the discrete cross-correlation is in:
    https://www.mathworks.com/help/matlab/ref/xcorr.html

    Args:
    y1, y2: Should have the same length.

    Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
    """
    if len(y1) != len(y2):
        raise ValueError('The lengths of the inputs should be the same.')

    y1_auto_corr = np.dot(y1, y1) / len(y1)
    y2_auto_corr = np.dot(y2, y2) / len(y1)
    corr = np.correlate(y1, y2, mode='same')
    # The unbiased sample size is N - lag.
    unbiased_sample_size = np.correlate(
    np.ones(len(y1)), np.ones(len(y1)), mode='same')
    corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
    shift = len(y1) // 2

    max_corr = np.max(corr)
    argmax_corr = np.argmax(corr)#[shift-5:shift+5])
    return max_corr, argmax_corr - shift

def errorf(params, y1, y2):
    """
    caucluates error function for minimization. 
    params - distortion polynomial coefficients
    y1 - data
    y2 - reference
    minimization parameter is negative max of cross-correlation
    """
    y,_ = transform(y1, params)
    maxcorr, lag = cross_corr(y, y2)
    return -maxcorr

def find_distortion(y1, y2, x0 = [0,0]):
    """
    y1 - data
    y2 - reference
    finds optimal distortion parameters based on minimization of cross-correlation function
    returns optimial distortion parameters
    """
    x0 = [0,0]
    res = minimize(errorf, x0, method='Nelder-Mead', tol=1e-8, args=(y1,y2))
    return res.x

def image_to_grid(image,positions):
    X = positions[1,:]
    Y = positions[0,:]
    xi = np.linspace(np.min(X), np.max(X), image.shape[0])
    yi = np.linspace(np.min(Y), np.max(Y), image.shape[1])
    xi, yi = np.meshgrid(xi, yi)
    zi = interpolate.griddata((X.transpose().ravel(),Y.transpose().ravel()),    
                                   image.transpose().ravel(),          
                                   (xi,yi),    
                                   method='linear').transpose()
    zi[zi==np.nan] = 0
    return zi

def grid_to_image(image,positions,X_shift):
    X = positions[1,:]
    Xr = np.reshape(X,image.transpose().shape)
    xi = np.linspace(np.min(X), np.max(X), image.shape[0])
    image_int = X_shift.copy()
    i=0
#    plt.figure()
    for line in X_shift.transpose():
        x = Xr[i,:]
        line_int = np.interp(x,xi,line)
        image_int[:,i] = line_int
        i=i+1
        #image_int[image_int==np.nan] = 0
    return image_int

def align_image(fluo_image, reference_image, positions, scan_pixel_size, scaleY=False):
    fluo_image = image_to_grid(fluo_image,positions)
    fluo_aligned = fluo_image.copy() #placeholder array
    X_shift = fluo_image.copy()
    Y_shift = fluo_image.copy()
    # find y shift
    fluo_profile_x = normalize(np.diff(np.nansum(fluo_image,axis=0)))
    fluo_ref_profile_x = normalize(np.diff(np.nansum(reference_image,axis=0)))
    corr,Ylag = cross_corr(fluo_profile_x,fluo_ref_profile_x) # determine shift by xcorrelation
    fluo_image = np.roll(fluo_image,-Ylag, axis=1)
    i=0
    for line_data in fluo_image.transpose():
        line_data[np.isnan(line_data)] = 0
        line_ref = normalize(reference_image[:,i]) #normalize reference
        x = np.linspace(0,len(line_data),len(line_data)) # original x coordinates
        if np.nanmean(line_data)/np.nanstd(line_data) < 2: # check if signal to noise ratio is good (to avoid excessive random shifts of empty lines)
            line_data_norm = normalize(line_data) # normalize data
            params = find_distortion(line_data_norm,line_ref,[0,0,0,0]) # find distortion parameters
            line_data_aligned, xprime = transform(line_data, params) # transform data based on found parameters. does not include shift
            line_data_aligned_norm = normalize(line_data_aligned) # normalize transformed data
            corr,lag = cross_corr(line_data_aligned_norm,line_ref) # determine shift by xcorrelation
        else:
            xprime = x
            lag = 0
        xprime = xprime + lag # add shift to xprime coordinates
        line_data_aligned = np.interp((xprime), x, line_data) # interpolate data from x to xprime
        fluo_aligned[:,i] = line_data_aligned
        X_shift[:,i] = (xprime-x)*scan_pixel_size[0]
        i = i+1
    
    Y_shift = Y_shift*0 + Ylag*scan_pixel_size[1]
    return fluo_aligned, X_shift, Y_shift

# End of correction methods #
