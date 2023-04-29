# -*- coding: utf-8 -*-

import os
import numpy as np
import ast
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

class p10_scan_reader:
    
    def __init__(self, pathfio):
        """
        reading scan information (fio files) from a p10 scan
        
        :param:
            -pathfio: the general path to the fio file
        """
        assert os.path.exists(pathfio), "The scan file does not exist, please check the p10_newfile name or the scan number again!"
        
        scan_data=np.array([])
        self.position_infor={}
        motors=[]
        f = open(pathfio, 'r')
        markp=False
        markd=False
        markc=False
        step_num=0
        for line in f:
            if line[0]=="!":
                markp=False
                markd=False
                markc=False
            elif markc:
                self.command=line.rstrip('\n')
                markc=False
            elif markp:
                try:
                    self.position_infor[line.split()[0]]=ast.literal_eval(line.split()[-1])
                except:
                    self.position_infor[line.split()[0]]=line.split()[-1].rstrip()
            elif markd and (line[0:5]==" Col "):
                motors.append(line.split()[2])
            elif markd:
                value= np.fromstring(line, dtype=float, sep=" ")
                scan_data=np.append(scan_data, value, axis=0)
                step_num+=1
            if line[0:2] == "%p":
                markp=True
            elif line[0:2] == "%d":
                markd=True
            elif line[0:2] == "%c":
                markc=True
        f.close()
        scan_data=np.reshape(scan_data, (step_num,len(motors)))
        self.scan_infor=pd.DataFrame(scan_data, columns=motors)
        return
    
    def get_command(self):
        return self.command
    
    def get_motor_pos(self, motor_name):
        return self.position_infor[motor_name]
    
    def get_scan_data(self, counter_name):
        return np.array(self.scan_infor[counter_name])
    
    def __str__(self):
        return 'scan #%d: %s'%(self.position_infor['_scan'], self.command)
    
    def knife_edge_estimation(self, counter_name, smooth=True, plot=False):
        motor=self.command.split()[1]
        motor_scan_value=np.array(self.scan_infor[motor])
        counter_scan_value=np.array(self.scan_infor[counter_name]/self.scan_infor['curpetra'])
        #normaize data
        counter_scan_value=(counter_scan_value-np.amin(counter_scan_value))/((np.amax(counter_scan_value)-np.amin(counter_scan_value)))
        diff_motor=(motor_scan_value[:-1]+motor_scan_value[1:])/2
        diff_counter=np.diff(counter_scan_value)
        if smooth:
            diff_counter=savgol_filter(diff_counter, 5, 2)   
        if 0.5*np.abs(np.amin(diff_counter))>np.abs(np.amax(diff_counter)):
            diff_counter=diff_counter*-1.0
        p0=[np.argmax(diff_counter), diff_motor[np.argmax(diff_counter)], 1.0]
        popt, pcov=curve_fit(self.gaussian, diff_motor, diff_counter, p0=p0)
        cen=popt[1]
        FWHM=2.35482*popt[2]
        if plot:
            plt.subplot(1,2,1)
            scan_range=np.ptp(motor_scan_value)
            range_index=np.logical_and(motor_scan_value>cen-0.2*scan_range, motor_scan_value<cen+0.2*scan_range)
            plt.plot(motor_scan_value[range_index], counter_scan_value[range_index], "x-")
            plt.ylabel('%s (a.u.)'%counter_name)
            plt.xlabel(motor)
            plt.subplot(1,2,2)
            range_index=np.logical_and(diff_motor>cen-0.2*scan_range, diff_motor<cen+0.2*scan_range)
            plt.plot(diff_motor[range_index], diff_counter[range_index], 'o', label=str(self.get_motor_pos('_scan')))
            plt.plot(diff_motor[range_index], self.gaussian(diff_motor[range_index], popt[0], popt[1], popt[2]), color=plt.gca().lines[-1].get_color(), label='FWHM %0.2f'%FWHM)
            plt.ylabel('Intensity (a.u.)')
            plt.xlabel(motor)
        return cen, FWHM

    def gaussian(self, x, amp, cen, wid):
        return amp*np.exp(-(x-cen)**2/(2*wid**2))

def main():       
    path=r'T:\2020\data\11010278\raw\%s\%s_%05d.fio'
    p10_newfile=r'real_align_01'
    scan_num=104
    scan=p10_scan_reader(path%(p10_newfile, p10_newfile, scan_num))
    xcen, FWHM=scan.knife_edge_estimation('diffdio', smooth=True, plot=True)
    print(scan)

if __name__ == '__main__':
    main()