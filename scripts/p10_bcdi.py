#!/usr/local/bin/python2.7.3 -tttt
"""
The code to estimate the quality of the data and generate for the 3D reciprocal space for a typical Bragg CDI studies at P10.

Example command: e4mdscan om -.5 .5 400 1 0 1.

The code performs several functions:
    1. 'Gif':Convert the images in the rocking curve into the gif file 
    2. 'Direct_cut': Directly cut the stacked detector images for Phase retrieval with PyNX
    3. 'Reciprocal_space_map': Calculate the three dimensional reciprocal space map and cut it to prepare for the PyNX.
    4. '2D_cuts': Generate the 2D cuts at the maximum intensity for the 2D phase retrieval studies

Important: 
    1. The number of steps in the rocking curve should be smaller than 2000.
    2. The hdf5plugin library should be already installed (pip install hdf5plugin) 
    3. If the gif images could not be generated, please install imagemagick (https://imagemagick.org/index.php)
    4. Please load the Information file generator and the fioreader with the correct path (change the path in the line 52 to the correct folder: sys.path.append(r'E:\Work place 3\testprog\X-ray diffraction\Common functions')) or put the two files in the same folder as this code
    
Input:
    1. general information: year, beamtimeID, p10_file name, scan number  together determines the path to read the experimental data
    2. Detector parameters: 
        distance: distance of the detector
        pixelsize: pixel size of the detector
        cch: Direct beam position on the detector Y, X 
        wxy: The half width of selected ROI on the detector
    3. paths:
        path defines the folder to read the images
        pathsave defines the folder to save the generated imagesand the infomation file
        path mask defines the path to load the prefined mask for the detector
    4. Detector parameters including the distance from the sample to the detector (distance), pixel_size of the detector (pixel_size), the direct beam position (cch), and the half width of the region of interest (wxy) 
    5. box size for the direct cut in pixels: The half width for the direct cut of the stacked detector images
    6. reciprocal space box size in pixels: The half width for the reciprocal space map


    
In case of questions, please contact me. Detailed explaination file of the code could be sent upon requist.
Author: Ren Zhe
Date: 2020/08/20
Email: zhe.ren@desy.de or renzhetu001@gmail.com
"""

import os
import numpy as np
from scipy.ndimage import measurements
from scipy.ndimage.interpolation import affine_transform
import scipy.io as scio
import hdf5plugin
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import sys
sys.path.append(r'E:\Work place 3\testprog\X-ray diffraction\Common functions')
from Information_file_generator import BCDI_Information
from fio_reader import p10_scan_reader

def Finding_peak_position(dataset):
    """
    Finding the maximum peak position in the rocking curve
    
    :param:
        -dataset: the stacked detector images
    :return:
        -pch: pch[0] the frame with the maximum intensity in the rocking curve
              pch[1] the peak position Y on the detector
              pch[2] the peak position X on the detector
    """
    print("Finding the frames with the highest intensity....")
    npoints=(dataset.shape)[0]
    intensity_sum=np.zeros(npoints)
    pch=np.zeros(3, dtype=int)
    for i in range(npoints):
        img=np.array(dataset[i,:,:], dtype=float)
        img[img>1.0e7]=0                                                        #remove the hotpixels and the channels between the panel
        intensity_sum[i]=np.sum(img)
        sys.stdout.write('\rprogress:%d%%'%((i+1)*100.0/npoints))
        sys.stdout.flush()
    img=np.array(dataset[np.argmax(intensity_sum),:,:], dtype=float)
    img[img>1.0e7]=0                                                            #remove the hotpixels and the channels between the panel
    pch[0]=np.argmax(intensity_sum)
    pch[-2:]=np.unravel_index(np.argmax(img), img.shape)
    print("") 
    print("peak position on the detector: "+str(pch))
    return pch

def Cal_rebinfactor(om_ar, delta, distance, pixelsize):
    """
    Check whether there is a need to increase the step number in the rocking curve
    The rebinfactor: Step size in the rocking direction/correponding detector pixel size in the reciprocal space
    This determines the lowest resolution of the reciprocal space map generated
    
    :param:
        -omstep: the step size in the rocking curve
        -delta: the 2theta angle in the rocking curve
        -distance: distance from the sample to the detector
        -pixelsize: the pixel size of the 2D detector 
    :return:
        -rebinfactor: The ratio between the step size in the rocking direction and the pixel size on the detector
    """
    omstep=np.radians((om_ar[-1]-om_ar[0])/(len(om_ar)-1.0))
    delta=np.radians(delta)
    rebinfactor=2.0*np.sin(delta/2.0)*distance/pixelsize*omstep
    
    print("rebinfactor calculated:%f"%rebinfactor)
    if rebinfactor>1.5:
        rebinfactor=int(rebinfactor)+1
        print('Maybe consider increasing the step numbers in the scan.')
    else:
        rebinfactor=1
        print('The number of steps for the scan shoulde be fine.')
    return rebinfactor


def Cal_reciprocal_space_range(om_ar, delta, distance, pixelsize, wxy, cch, pch, data_shape):
    """
    Calculate the range for the final reciprocal space map
    
    :param:
        -om_ar: the measured omega angles in the rocking curve
        -delta: the 2theta angle in the rocking curve
        -distance: sample detector distance
        -pixelsize: pixelsize of the detector in mm
        -wxy: the half width of the selected detector roi in Y, X direction
        -cch: direct beam position on the detector
        -pch: the maximum peak position
        -data_shape: the shape of the aimed data
    :return:
        -q_range: the array containing the q range of the reciprocal space
        q_range=[[qzmin, qzmax],
                 [qymin, qymax],
                 [qxmin, qxmax]]
    """
    zd, yd, xd=data_shape
    om_ar=np.radians(om_ar)
    delta=np.radians(delta)
    om2d, Y_ar=np.meshgrid(om_ar, np.arange(yd)[(pch[1]-wxy[0]):(pch[1]+wxy[0])])
    X_ar=np.arange(xd)[(pch[2]-wxy[1]):(pch[2]+wxy[1])]
    x0ar=np.sin(delta-om2d)*(Y_ar-cch[0])+(np.cos(delta-om2d)-np.cos(om2d))*distance/pixelsize
    y0ar=X_ar-cch[1]
    z0ar=np.cos(delta-om2d)*(cch[0]-Y_ar)+(np.sin(delta-om2d)+np.sin(om2d))*distance/pixelsize
    q_range=np.array([[np.amin(z0ar),np.amax(z0ar)], [np.amin(y0ar),np.amax(y0ar)], [np.amin(x0ar),np.amax(x0ar)]])
    return q_range


def RSM_conversion(dataset, om_ar, delta, distance, pixelsize, rebinfactor, cval, nx, ny, nz):
    '''
    Convert the stacked detector images to the orthogonal reciprocal space map with interpolation
    
    :param:
        -ar3d: cutted stacked detector images of masks
        -om_ar: the omega angle in the measurement
        -delta: the 2theat angle in the measurement
        -distance: sample detector distance
        -pixelsize: pixelsize of the detector in mm
        -rebinfactor: rebinfactor calculated
        -cval: the value to fill for the missing points in the interpolation
        -nx, ny, nz: the number of points for the aimed reciprocal space map
    
    :return:
        -RSM_int: the gridded reciprocal space map
    '''
    #prepare for the conversion
    om_ar=np.radians(om_ar)
    delta=np.radians(delta)
    zd, yd, xd= dataset.shape
    om=om_ar[0]
    omstep=(om_ar[-1]-om_ar[0])/(zd-1)
    
    #calculate the transformation matrix for the RMS
    om_C=distance*omstep/pixelsize
    Coords_transform=np.array([[(-np.cos(delta-om)+np.cos(om))*om_C, -np.cos(delta-om)], [(np.sin(delta-om)+np.sin(om))*om_C, np.sin(delta-om)]])
    invCoords=rebinfactor*np.linalg.inv(Coords_transform)
    offset=np.array([zd/2, yd/2])-np.dot(invCoords, np.array([nz/2, nx/2]))
    #Interpolation
    intensityfinal=np.zeros((nz, ny, nx))
    for X in np.arange(ny):
        intensity2d=dataset[:,:,int(X)]
        intensity2dinterpolation=affine_transform(intensity2d, invCoords, offset=offset, output_shape=(nz, nx), order=3, cval=cval, output=float)
        intensityfinal[:,X,:]=intensity2dinterpolation
        sys.stdout.write('\rprogress:%d%%'%((X+1)*100.0/ny))
        sys.stdout.flush()
    print('')
    intensityfinal=np.clip(intensityfinal, 0, 1.0e7)
    return intensityfinal

def Cut_central(dataset, bs, peak_pos, mode='weight_center'):
    '''
    Cut the 3D intensity around its center of the mass
    
    :param:
        -dataset: the 3D intensity
        -bs: box size for the cut
        -peak_pos: estimated peak_position
    
    :return:
        -intcut: the cutted intensity
        -peak_pos: position for the center of the mass
    '''
    
    #Cutting the three dimensional data with the center of mass in the center of the intensity distribution
    print('finding the centeral position for the cutting')
    cen_pos=np.array(bs, dtype=float)-0.5
    intcut=np.array(dataset[(peak_pos[0]-bs[0]):(peak_pos[0]+bs[0]),(peak_pos[1]-bs[1]):(peak_pos[1]+bs[1]), (peak_pos[2]-bs[2]):(peak_pos[2]+bs[2])])
    intcut[intcut>1.0e7]=0
    if mode=='weight_center':
        print('cut according to the weight center')
        while not np.allclose(measurements.center_of_mass(intcut), cen_pos, atol=0.5):
            peak_pos=np.array(peak_pos+np.around((measurements.center_of_mass(intcut)-cen_pos)), dtype=int)
            intcut=np.array(dataset[(peak_pos[0]-bs[0]):(peak_pos[0]+bs[0]),(peak_pos[1]-bs[1]):(peak_pos[1]+bs[1]), (peak_pos[2]-bs[2]):(peak_pos[2]+bs[2])])
            intcut[intcut>1.0e7]=0
    else:
        print('cut according to the maximum intensity')
    return intcut, peak_pos

def plotandsave(RSM_int, q_range, unit, pathsavetmp, qmax=np.array([])):
    dz, dy, dx=RSM_int.shape
    qz=np.arange(dz)*unit+q_range[0,0]
    qy=np.arange(dy)*unit+q_range[1,0]
    qx=np.arange(dx)*unit+q_range[2,0]
    #save the qx qy qz cut of the 3D intensity
    print('Saving the qx qy qz cuts......')
    plt.figure(figsize=(12,12))
    pathsaveimg=pathsavetmp%('cutqz')
    if len(qmax)==0:
        plt.contourf(qx,qy,np.log10(np.sum(RSM_int, axis=0)+1.0), 150, cmap='jet')
    else:
        plt.contourf(qx,qy,np.log10(RSM_int[qmax[0], :, :]+1.0), 150, cmap='jet')
    plt.xlabel(r'Q$_x$ ($1/\AA$)', fontsize= 14)
    plt.ylabel(r'Q$_y$ ($1/\AA$)', fontsize= 14)
    plt.axis('scaled')
    plt.savefig(pathsaveimg)
    # plt.close()
    
    plt.figure(figsize=(12,12))
    pathsaveimg=pathsavetmp%('cutqy')
    if len(qmax)==0:
        plt.contourf(qx,qz,np.log10(np.sum(RSM_int, axis=1)+1.0), 150, cmap='jet')
    else:
        plt.contourf(qx,qz,np.log10(RSM_int[:, qmax[1], :]+1.0), 150, cmap='jet')
    plt.xlabel(r'Q$_x$ ($1/\AA$)', fontsize= 14)
    plt.ylabel(r'Q$_z$ ($1/\AA$)', fontsize= 14)
    plt.axis('scaled')
    plt.savefig(pathsaveimg)
    # plt.close()
    
    plt.figure(figsize=(12,12))
    pathsaveimg=pathsavetmp%('cutqx')
    if len(qmax)==0:
        plt.contourf(qy,qz,np.log10(np.sum(RSM_int, axis=2)+1.0), 150, cmap='jet')
    else:
        plt.contourf(qy,qz,np.log10(RSM_int[:, :, qmax[2]]+1.0), 150, cmap='jet')
    plt.xlabel(r'Q$_y$ ($1/\AA$)', fontsize= 14)
    plt.ylabel(r'Q$_z$ ($1/\AA$)', fontsize= 14)
    plt.axis('scaled')
    plt.savefig(pathsaveimg)
    plt.show()
    # plt.close()
    return

def plotandsave2(RSM_int,  pathsavetmp, mask=np.array([])):
    mask=np.ma.masked_where(mask==0, mask)
    dz, dy, dx=RSM_int.shape
    #save the qx qy qz cut of the 3D intensity
    print('Saving the qx qy qz cuts......')
    plt.figure(figsize=(12,12))
    pathsaveimg=pathsavetmp%('cutqz')
    plt.imshow(np.log10(RSM_int[int(dz/2), :,:]+1.0), cmap='Blues')
    if mask.ndim!=1:
        plt.imshow(mask[int(dz/2), :, :], cmap='Reds', alpha=0.8, vmin=0, vmax=1)
    plt.xlabel(r'Q$_x$ ($1/\AA$)', fontsize= 14)
    plt.ylabel(r'Q$_y$ ($1/\AA$)', fontsize= 14)
    plt.axis('scaled')
    plt.savefig(pathsaveimg)
    plt.show()
    # plt.close()
    
    plt.figure(figsize=(12,12))
    pathsaveimg=pathsavetmp%('cutqy')
    plt.imshow(np.log10(RSM_int[:, int(dy/2), :]+1.0), cmap='Blues')
    if mask.ndim!=1:
        plt.imshow(mask[:, int(dy/2), :], cmap='Reds', alpha=0.8, vmin=0, vmax=1)
    plt.xlabel(r'Q$_x$ ($1/\AA$)', fontsize= 14)
    plt.ylabel(r'Q$_z$ ($1/\AA$)', fontsize= 14)
    plt.axis('scaled')
    plt.savefig(pathsaveimg)
    plt.show()
    # plt.close()
    
    plt.figure(figsize=(12,12))
    pathsaveimg=pathsavetmp%('cutqx')
    plt.imshow(np.log10(RSM_int[:,:,int(dx/2)]+1.0), cmap='Blues')
    if mask.ndim!=1:
        plt.imshow(mask[:, :, int(dx/2)], cmap='Reds', alpha=0.8, vmin=0, vmax=1)
    plt.xlabel(r'Q$_y$ ($1/\AA$)', fontsize= 14)
    plt.ylabel(r'Q$_z$ ($1/\AA$)', fontsize= 14)
    plt.axis('scaled')
    plt.savefig(pathsaveimg)
    plt.show()
    # plt.close()
    return


def numpy2vtk(filename, a, dx=1.0,dy=1.0,dz=1.0,x0=0.0,y0=0.0,z0=0.0):
   # http://www.vtk.org/pdf/file-formats.pdf
   f=open(filename,'w')
   nx,ny,nz=a.shape
   f.write("# vtk DataFile Version 2.0\n")
   f.write("Test data\n")
   f.write("ASCII\n")
   f.write("DATASET STRUCTURED_POINTS\n")
   f.write("DIMENSIONS %u %u %u\n"%(nz,ny,nx))
   f.write("SPACING %f %f %f\n"%(dx,dy,dz))
   f.write("ORIGIN %f %f %f\n"%(x0,y0,z0))
   f.write("POINT_DATA %u\n"%len(a.flat))
   f.write("SCALARS volume_scalars float 1\n")
   f.write("LOOKUP_TABLE default\n")
   for i in a.flat:
     f.write("%f "%i)
   f.close()
   return ()


def main(scan):
    #Inputs: select the functions of the code
    Functions_selected=['Gif', 'Direct_cut', 'Reciprocal_space_map', '2D_cuts']
    
    #Inputs: general information
    year="2020"                                                                     #The year for the experiemnt
    beamtimeID="11010265"                                                           #The beamtimeID
    p10_file=r"M32323"                                                            #The name for the p10 newfile #The scan number
    generating_mask=True
    #generating_mask=False                                                          #select whether you want to generate the mask for the hotpixel and the channels
    
    #Inputs:Detector parameters
    distance=1834.7                                                                 #distance of the detector
    pixelsize=0.075                                                                 #pixel size of the detector
    #cch=[1346, 1301]         #(beamtime in July)                                   #Direct beam position on the detector Y, X 
    cch=[1342, 1487]         #(beamtime in August)                                  #Direct beam position on the detector Y, X
    wxy=[400, 400]                                                                  #The half width of selected ROI
    
    #If you have performed the calibration scan
    om_error=0.5335
    delta_error=0.6684
    
    #Inputs: box size for the direct cut in pixels
    DC_bs=[30, 150, 150]                                                           #The box size for the direct cut
    #DC_cut_mode='weight_center'
    DC_cut_mode='maxium_intensity'
    
    #Inputs: reciprocal space box size in pixels
    RSM_bs=[150, 90, 90]
    generating_3D_vtk_file=False
    #RSM_cut_mode='weight_center'
    RSM_cut_mode='maxium_intensity'
    
    #Inputs: paths
    # Nazanin test: /media/dzhigdStorage/nazanin/data/original_data/11010265/raw/M32323_00071
    path=r"/media/dzhigdStorage/nazanin/data/original_data/11010265/raw/%s_%05d"%(p10_file, scan)            #The path for the scan folder
    pathsave=r"/media/dzhigdStorage/nazanin/data/scan%05d"%scan #The path to save the results
    pathmask=r'/home/dzhigd/Software/general_mask.npy'
    pathfio=os.path.join(path, "%s_%05d.fio"%(p10_file, scan))
    pathimg=os.path.join(path, "e4m/%s_%05d_data_000001.h5")%(p10_file, scan)
    pathinfor=os.path.join(pathsave,"scan_%04d_information.txt"%scan)
    path3dintensity=os.path.join(pathsave, "scan%04d_fast_cubic.npz"%(scan))
    if generating_mask:
        pathsavemask=os.path.join(pathsave,"scan_%04d_detector_mask.npy"%scan)
        path3dmask=os.path.join(pathsave,"scan%04d_mask.npz"%(scan))
    
    #generating the folder to save the data
    if not os.path.exists(pathsave):
        os.mkdir(pathsave)
    
    print("#################")
    print("Basic information")
    print("#################")
    assert os.path.exists(pathimg), 'The image files does not exists, please check the paths again'
    
    #reading images and fio files
    scan_data=p10_scan_reader(pathfio)
    delta=scan_data.get_motor_pos('del')+delta_error                                #reading the delta values in the rocking curve
    om_ar=scan_data.get_scan_data('om')+om_error                                    #reading the omega values for each step in the rocking curve
    om_step=(om_ar[-1]-om_ar[0])/(len(om_ar)-1)
    f=h5py.File(pathimg, "r")
    dataset=f['entry/data/data']                                                    #read the detector images
    print('scan%03d: '%scan+scan_data.get_command())
    
    #Finding the maximum peak position
    pch=Finding_peak_position(dataset)
    om=om_ar[pch[0]]                                                                #the omega value of the diffraction peak
    print("peak at omega = %f"%(om))                                               
        
    #determining the rebin parameter
    rebinfactor=Cal_rebinfactor(om_ar, delta, distance, pixelsize)
    
    #generate the mask
    if generating_mask:
        if os.path.exists(pathmask):
            print('The predefined mask loaded')
            mask=np.load(pathmask)
        else:
            print('Did not find the predefined mask, generate the mask based on the detector image instead')
            mask=np.zeros((dataset.shape[1], dataset.shape[2]))
        img=np.array(dataset[pch[0],:,:], dtype=float)
        mask[img>1.0e7]=1
        np.save(pathsavemask, mask)
    plt.subplot(1,2,1)
    plt.imshow(np.log10(img+1.0))
    plt.subplot(1,2,2)
    plt.imshow(np.log10(mask+1.0))
    plt.show()
    
    #calculate the unit of the grid intensity
    hc=1.23984*10000.0
    wavelength=hc/scan_data.get_motor_pos('fmbenergy')
    units=2.0*np.pi*pixelsize/wavelength/distance
    
    #writing the scan information to the aimed file
    section_ar=['General Information', 'Paths', 'Scan Information', 'Routine1: Reciprocal space map', 'Routine2: direct cutting']
    infor=BCDI_Information(pathinfor)
    infor.add_para('command', section_ar[0], scan_data.get_command())
    infor.add_para('year', section_ar[0], year)
    infor.add_para('beamtimeID', section_ar[0], beamtimeID)
    infor.add_para('p10_newfile', section_ar[0], p10_file)
    infor.add_para('scan_number', section_ar[0], scan)
    
    infor.add_para('pathsave', section_ar[1], pathsave)
    infor.add_para('pathfio', section_ar[1], pathfio)
    infor.add_para('pathimage', section_ar[1], pathimg)
    infor.add_para('pathinfor', section_ar[1], pathinfor)
    infor.add_para('path3dintensity', section_ar[1], path3dintensity)
    if generating_mask:
        infor.add_para('pathmask', section_ar[1], pathmask)
        infor.add_para('path3dmask', section_ar[1], path3dmask)
        
    infor.add_para('peak_position', section_ar[2], list(pch))
    infor.add_para('omega', section_ar[2], om)
    infor.add_para('delta', section_ar[2], delta)
    infor.add_para('omega_error', section_ar[2], om_error)
    infor.add_para('delta_error', section_ar[2], delta_error)
    infor.add_para('om_step', section_ar[2], om_step)
    infor.add_para('direct_beam_position', section_ar[2], list(cch))
    infor.add_para('detector_distance', section_ar[2], distance)
    infor.add_para('pixelsize', section_ar[2], pixelsize)
    infor.add_para('generating_mask', section_ar[2], generating_mask)
    
    infor.add_para('roi_width', section_ar[3], list(wxy))
    infor.add_para('RSM_unit', section_ar[3], units*rebinfactor)
    
    infor.add_para('DC_unit', section_ar[4], units)
    
    infor.infor_writer()
    
    
    if 'Gif' in Functions_selected:
        print("")
        print("########################")
        print("Generating the gif image")
        print("########################")
        #save images into gif
        fig=plt.figure(figsize=(6,6))
        plt.axis("off")
        img_frames=[]
        for i in range(len(om_ar)):
            if i%5==0:
                img=np.array(dataset[i,pch[1]-wxy[0]:pch[1]+wxy[0],pch[2]-wxy[1]:pch[2]+wxy[1]], dtype=float)
                img[img>1.0e7]=0
                plt_im=plt.imshow(np.log10(img+1.0), cmap="hot")
                img_frames.append([plt_im])
        gif_img=anim.ArtistAnimation(fig, img_frames)
        pathsavegif=os.path.join(pathsave, "scan%04d.gif"%scan)
        gif_img.save(pathsavegif, writer= 'pillow', fps=10)
        print('GIF image saved')
        # plt.close()
    
    if 'Direct_cut' in Functions_selected:
        print("")
        print("##########################################")
        print("Generating the stack of the detector image")
        print("##########################################")
              
        #Save the data
        pathtmp=os.path.join(pathsave, "pynxpre")
        if not os.path.exists(pathtmp):
            os.mkdir(pathtmp)
        pathtmp=os.path.join(pathtmp, "Routine2")
        if not os.path.exists(pathtmp):
            os.mkdir(pathtmp)
            
        #cutting the stacked detector images
        Direct_cut, npch=Cut_central(dataset, DC_bs, pch, mode=DC_cut_mode)
    
        print("saving the data...")
        pathsavenpy=os.path.join(pathtmp, "scan%04d_direct_cut.npz"%scan)
        np.savez_compressed(pathsavenpy, data=Direct_cut)
        if generating_mask:
            print('saving mask')
            Direct_mask=np.repeat(mask[np.newaxis, (npch[1]-DC_bs[1]):(npch[1]+DC_bs[1]), (npch[2]-DC_bs[2]):(npch[2]+DC_bs[2])], 2*DC_bs[0], axis=0)
            pathsavemask=os.path.join(pathtmp, "scan%04d_maskcut.npz"%scan)
            np.savez_compressed(pathsavemask, data=Direct_mask)
       
        #ploting the ycut of the stacking images to estimate the quality of the image
        ycut=Direct_cut[:, :, DC_bs[2]]
        plt.imshow(np.log10(ycut.T+1.0), cmap="Blues")
        if generating_mask:
            maskycut=Direct_mask[:, :, DC_bs[2]]
            plt.imshow(np.ma.masked_where(maskycut==0, maskycut).T, cmap="Reds", alpha=0.8, vmin=0.1, vmax=0.5)
        plt.xlabel('img_num')
        plt.ylabel('detector_Y')
        plt.savefig(os.path.join(pathtmp, 'ycut.png'))
        plt.show()    
        # plt.close()       
        
        infor.add_para('direct_cut_mode', section_ar[4], DC_cut_mode)
        infor.add_para('direct_cut_box_size', section_ar[4], DC_bs)
        infor.add_para('direct_cut_centeral_pixel', section_ar[4], list(npch))
        infor.infor_writer()    
        
        
    if 'Reciprocal_space_map' in Functions_selected:
        print("")
        print("##################")
        print("Generating the RSM")
        print("##################")
    
        #Creating the aimed folder
        pathtmp=os.path.join(pathsave, "pynxpre")
        if not os.path.exists(pathtmp):
            os.mkdir(pathtmp)
        pathtmp=os.path.join(pathtmp, "Routine1")
        if not os.path.exists(pathtmp):
            os.mkdir(pathtmp)
        
        #calculate the qx, qy, qz ranges of the scan
        q_range=Cal_reciprocal_space_range(om_ar, delta, distance, pixelsize, wxy, cch, pch, dataset.shape)
        nz=int((q_range[0,1]-q_range[0,0])/rebinfactor)+1                  #the number of point in the qz direction
        ny=int((q_range[1,1]-q_range[1,0])/rebinfactor)+1                  #the number of point in the qy direction
        nx=int((q_range[2,1]-q_range[2,0])/rebinfactor)+1                  #the number of point in the qx direction
        print("number of points for the reciprocal space:")
        print(" qz  qy  qx")
        print(nz, ny, nx)
    
        #crop the images and put them into memory
        ar3d=np.array(dataset[:, pch[1]-wxy[0]:pch[1]+wxy[0],pch[2]-wxy[1]:pch[2]+wxy[1]:rebinfactor], dtype=float)
        ar3d[ar3d>1.0e7]=0
        
        #generate the 3D reciprocal space map
        print('Calculating intensity...')
        RSM_int=RSM_conversion(ar3d, om_ar, delta,distance, pixelsize, rebinfactor, 0, nx, ny, nz)
        del ar3d  
            
        #Cutting the three dimensional data with the center of mass in the center of cut intensity
        qxmax=np.argmax(np.sum(RSM_int, axis=(0,1)))
        qymax=np.argmax(np.sum(RSM_int, axis=(0,2)))
        qzmax=np.argmax(np.sum(RSM_int, axis=(1,2)))
        RSM_cut, qcen= Cut_central(RSM_int, RSM_bs, np.array([qzmax, qymax, qxmax]), mode=RSM_cut_mode)
        
        print("saving the RSM cut for pynx...")
        pathsavenpy=os.path.join(pathtmp, "scan%04d_fast_cubic.npz"%scan)
        pathsavemat=os.path.join(pathtmp, "scan%04d_fast_cubic.mat"%scan)
        np.savez_compressed(pathsavenpy, data=RSM_cut)
        scio.savemat(pathsavemat, mdict={'rsm_data': RSM_int, 'q_range': q_range})
        
        #load the mask and generate the new mask for the 3D reciprocal space map
        if generating_mask:
            print('Loading the mask...')
            mask=np.repeat(mask[np.newaxis, pch[1]-wxy[0]:pch[1]+wxy[0],pch[2]-wxy[1]:pch[2]+wxy[1]:rebinfactor], len(om_ar), axis=0)
            print('Calculating the mask...')    
            RSM_mask=RSM_conversion(mask, om_ar, delta, distance, pixelsize, rebinfactor,  1, nx, ny, nz)
            del mask   
    
            RSM_mask[RSM_mask>=0.1]=1
            RSM_mask[RSM_mask<0.1]=0
            RSM_cut_mask=RSM_mask[(qcen[0]-RSM_bs[0]):(qcen[0]+RSM_bs[0]),(qcen[1]-RSM_bs[1]):(qcen[1]+RSM_bs[1]), (qcen[2]-RSM_bs[2]):(qcen[2]+RSM_bs[2])]
            
            print("saving the mask...")
            pathsavemask=os.path.join(pathtmp, "scan%04d_mask.npz"%scan)
            np.savez_compressed(pathsavemask, data=RSM_cut_mask)    
    
        if generating_3D_vtk_file:
            filename="scan%04d_fast_cubic.vtk"%scan
            pathsavevtk=os.path.join(pathtmp, filename)
            numpy2vtk(pathsavevtk, np.log10(RSM_cut+1.0))
    
        #Generate the images of the reciprocal space map
        print('Generating the images of the RSM')
        pathsavetmp=os.path.join(pathsave, 'scan%04d'%scan+'_%s.png')
        plotandsave(RSM_int, q_range*units, units*rebinfactor, pathsavetmp, qmax=np.array([qzmax, qymax, qxmax]))
        #plotandsave(RSM_int, q_range, units*rebinfactor, pathsavetmp)
        pathsavetmp=os.path.join(pathtmp, 'scan%04d'%scan+'_%s.png')
        plotandsave2(RSM_cut,  pathsavetmp, RSM_cut_mask)
        
        #save the information
        infor.add_para('nx', section_ar[3], nx)
        infor.add_para('ny', section_ar[3], ny)
        infor.add_para('nz', section_ar[3], nz)
        infor.add_para('rebinfactor', section_ar[3], rebinfactor)
        infor.add_para('qx_range', section_ar[3], [q_range[2,0]*units, q_range[2,1]*units])
        infor.add_para('qy_range', section_ar[3], [q_range[1,0]*units, q_range[1,1]*units])
        infor.add_para('qz_range', section_ar[3], [q_range[0,0]*units, q_range[0,1]*units])
        infor.add_para('qmax', section_ar[3], [qzmax, qymax, qxmax])
        infor.add_para('RSM_cut_mode', section_ar[3], RSM_cut_mode)
        infor.add_para('pynx_box_size', section_ar[3], RSM_bs)
        infor.add_para('qcen', section_ar[3], list(qcen))
        infor.infor_writer()
    
    if "2D_cuts" in Functions_selected:
        if 'Direct_cut' in Functions_selected:
            pathtmp=os.path.join(pathsave, "cuty")
            if not os.path.exists(pathtmp):
                os.mkdir(pathtmp)
            pathtmp2=os.path.join(pathtmp, "cuty.npy")
            np.save(pathtmp2, Direct_cut[:, :, DC_bs[2]])
            infor.add_para('path_cuty', '2D cuts', pathtmp)
            if generating_mask:
                pathtmp2=os.path.join(pathtmp, "cuty_mask.npy")
                np.save(pathtmp2, Direct_mask[:, :, DC_bs[2]])
                
        #Saving the intensity cuts for the 2D phase retrieval
        if 'Reciprocal_space_map' in Functions_selected:
            pathtmp=os.path.join(pathsave, "cutqz")
            if not os.path.exists(pathtmp):
                os.mkdir(pathtmp)
            pathtmp2=os.path.join(pathtmp, "cutqz.npy")
            np.save(pathtmp2, RSM_cut[RSM_bs[0], :, :])
            infor.add_para('path_cutqz', '2D cuts', pathtmp)
            if generating_mask:
                pathtmp2=os.path.join(pathtmp, "cutqz_mask.npy")
                np.save(pathtmp2, RSM_cut_mask[RSM_bs[0], :, :])
        
            pathtmp=os.path.join(pathsave, "cutqy")
            if not os.path.exists(pathtmp):
                os.mkdir(pathtmp)
            pathtmp2=os.path.join(pathtmp, "cutqy.npy")
            infor.add_para('path_cutqy', '2D cuts', pathtmp)
            np.save(pathtmp2, RSM_cut[:, RSM_bs[1], :])
            if generating_mask:
                pathtmp2=os.path.join(pathtmp, "cutqy_mask.npy")
                np.save(pathtmp2, RSM_cut_mask[:, RSM_bs[1], :])
                
            pathtmp=os.path.join(pathsave, "cutqx")
            if not os.path.exists(pathtmp):
                os.mkdir(pathtmp)
            pathtmp2=os.path.join(pathtmp, "cutqx.npy")
            np.save(pathtmp2, RSM_cut[:, :, RSM_bs[2]])
            infor.add_para('path_cutqx', '2D cuts', pathtmp)
            if generating_mask:
                pathtmp2=os.path.join(pathtmp, "cutqx_mask.npy")
                np.save(pathtmp2, RSM_cut_mask[:, :, RSM_bs[2]])
    
    infor.infor_writer()
    return

if __name__=='__main__':
    scan = 71
    main(scan)