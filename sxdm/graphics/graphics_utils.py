#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 23:43:44 2020

@author: dzhigd
"""
from scipy import ndimage, interpolate
import numpy as np
import matplotlib.pyplot as plt

class IndexTracker:
    def __init__(self, ax, data, axis):
        self.ax = ax
        self.scroll_axis = axis
        ax.set_title('use scroll wheel to navigate images')
        self.data = data                
        
        # Start slice to show
        self.ind = 0
        
        if self.scroll_axis == 0:
            self.slices, rows, cols = data.shape
            self.im = ax.imshow(self.data[self.ind, :, :])
        elif self.scroll_axis == 1:
            rows, self.slices, cols = data.shape
            self.im = ax.imshow(self.data[:, self.ind, :])
        elif self.scroll_axis == 2:
            rows, cols, self.slices = data.shape
            self.im = ax.imshow(self.data[:, :, self.ind])
                
        # plt.colorbar(self.im, self.ax)
        self.update()

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        self.update()

    def update(self):
        if self.scroll_axis == 0:
            self.im.set_data(self.data[self.ind, :, :])
        elif self.scroll_axis == 1:
            self.im.set_data(self.data[:, self.ind, :])
        elif self.scroll_axis == 2:
            self.im.set_data(self.data[:, :, self.ind])
            
        self.ax.set_title('Slice %d along axis %d ' % (self.ind, self.scroll_axis))
        
        self.im.axes.figure.canvas.draw()        


def scroll_data(data, axis=None, colormap=None):
    fig, ax = plt.subplots(1, 1)    
    
    if axis:
        tracker = IndexTracker(ax, data, axis)
    else:
        tracker = IndexTracker(ax, data, 0)        
    
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    if colormap:
        plt.set_cmap(colormap)
    else:
        plt.set_cmap('turbo')
    
    plt.show()
    
    return tracker
    
def show_q_space_projections(qx,qy,qz,data):    
#     Constants. They are needed for correct labeling of axes
    h                       = 4.1357e-15;                                  # Plank's constant
    c                       = 2.99792458e8;                                # Speed of light in vacuum
    energy = 11000
    pitch = 55e-6
    radius = 0.4
    
    wavelength = h*c/energy

    dq = (2*np.pi*pitch/(radius*wavelength))
    
    qxv = np.arange(np.min(qx),np.max(qx),dq)
    qyv = np.arange(np.min(qy),np.max(qy),dq)
    qzv = np.arange(np.min(qz),np.max(qz),dq)
    
    Qx,Qy,Qz = np.meshgrid(qxv, qyv, qzv)
    
    # find interpolant
    q = np.array((qz,qx,qy)).transpose()
    F = interpolate.LinearNDInterpolator(q, np.concatenate(np.concatenate(data)), fill_value=np.nan, rescale=True)
    
    # interpolate, this is the output of aligned diffraction data
    qspace_interpolated = np.nan_to_num(F(Qz.transpose(),Qx.transpose(),Qy.transpose()))
    
#    qspace_interpolated[qspace_interpolated == np.nan] = 0
    
    # sum data to plot total xrd map    
    plt.figure(num=1)
    plt.imshow(np.sum(qspace_interpolated,axis=0))
    
    plt.figure(num=2)
    plt.imshow(np.log10(np.sum(qspace_interpolated,axis=1)))
    
    plt.figure(num=3)
    plt.imshow(np.sum(qspace_interpolated,axis=2))