#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 22:00:25 2020

@author: dzhigd
"""

# Select the rocking curve

from scipy.interpolate import griddata as gd

def get_qspace_coordinates(xrd_data):

	scan_numbers = 109:128;

	% Processing path
	processing_root = sprintf('/home/dzhigd/work/projects/Qdevs_2020_NanoMAX/data/scan_%d_%d',scan_numbers(1),scan_numbers(end));

	% Data path
	path_root = sprintf('/media/dzhigd/My Passport/DDzhigaev/Data/MAXIV/NanoMax/2020101408/process/scan_%d_%d',scan_numbers(1),scan_numbers(end));

	for ii = 1:numel(scan_numbers)
	    ii
	    load(fullfile(path_root, sprintf('scan_%06d_merlin.mat',scan_numbers(ii))));
	    data(isnan(data)) = 0;
	    data_t(:,:,ii) = squeeze(sum(data,[1,2]));
	    gontheta(ii) = motor_positions.gontheta;
	end

	data = data_t;

	% Vertical coordinate on the detector is 1st
	% Horizontal coordinate on the detector is 2nd

	% Experiment parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% NanoMax convention:
	% gamma - horizontal detector
	% delta - vertical detector
	% gonphi - rotation about vertical axis
	% gontheta - rotation about horizontal axis
	nanomax.photon_energy   = motor_positions.energy;
	nanomax.gonphi          = motor_positions.gonphi; % [deg] - can be a range of angles
	nanomax.gontheta        = gontheta; % [deg] - can be a range of angles

	nanomax.radius          = motor_positions.radius*1e-3; % [m]
	nanomax.delta           = -motor_positions.delta;%2.1; % 2.1[deg] these angles are corrected with the sign respecting the rotation rules
	nanomax.gamma           = motor_positions.gamma; %0.48 12.72[deg] +0.467
	nanomax.detector_pitch  = 55e-6; % [m]

	nanomax.direct_beam     = round([1,  50]);       
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% Constants. They are needed for correct labeling of axes
	h                       = 4.1357e-15;                                  % Plank's constant
	c                       = 2.99792458e8;                                % Speed of light in vacuum

	wavelength = h*c/nanomax.photon_energy;

	k = 2*pi/wavelength; % wave vector

	dq = k*2*atan(nanomax.detector_pitch/(2*nanomax.radius)); % q-space pitch at the detector plane

	[hd,vd] = meshgrid(size(data,2)/2:-1:-(size(data,2)/2-1),size(data,1)/2:-1:-(size(data,1)/2-1));

	hd = (hd+size(data,2)/2-nanomax.direct_beam(2)).*nanomax.detector_pitch;
	vd = (vd+size(data,1)/2-nanomax.direct_beam(1)).*nanomax.detector_pitch;
	zd = ones(size(vd)).*nanomax.radius;

	d = [hd(:),vd(:),zd(:)]';

	r = squeeze(sqrt(sum(d.^2,1)));

	hq = k*(d(1,:)./(r));
	vq = k*(d(2,:)./(r));
	zq = k*(1-d(3,:)./r);

	q = [hq;vq;zq];

	% Check the coordinate mapping - real space m, direct scattering
	% test = data(:,:,10);
	% figure;
	% scatter3(q(1,:),q(2,:),q(3,:),10,log10(test(:)),'fill','s');
	% xlabel('qh');
	% ylabel('qv');
	% zlabel('qz');
	% view(7.7549,53.9161);
	% PROVED!

	%% Sample orientation matrix. Bounds the sample crystal with the laboratory frame
	% Angles alpha beta gamma were manually adjusted so that known peaks 
	% are exactly in their places

	% X is horizontal, perp to the beam, Y is vertical

	Rh = [1         0                    0; % detector rotation around horizintal axis 
	      0         cosd(nanomax.delta) -sind(nanomax.delta);
	      0         sind(nanomax.delta)  cosd(nanomax.delta)]; 

	Rv = [cosd(nanomax.gamma)  0  sind(nanomax.gamma); % detector rotation around vertical axis 
	      0                    1  0;
	      -sind(nanomax.gamma) 0  cosd(nanomax.gamma)];

	Rz = [cosd(0) -sind(0) 0; % detector rotation around beam axis 
	      sind(0)  cosd(0)  0;
	      0        0         1];

	U = Rh*Rv*Rz;
	    
	qR = (U*q); % correct so far in real space

	% Check the coordinate mapping - real space m, Bragg condition
	% figure;
	% scatter3(qR(1,:),qR(2,:),qR(3,:),10,test(:),'fill','s');
	% xlabel('qh');
	% ylabel('qv');
	% zlabel('qz');
	% view(7.7549,53.9161);
	% PROVED!

	%
	%% Initial coordinate of ki
	ki = [0,0,k]';

	kf = U*ki;

	Q = kf-ki;

	% Lab coordinate system: accosiated with the ki
	QLab(1,:) = (qR(1,:)+Q(1));
	QLab(2,:) = (qR(2,:)+Q(2));
	QLab(3,:) = (qR(3,:)+Q(3));

	% Check the lab space mapping
	% figure;
	% scatter3(QLab(1,:),QLab(2,:),QLab(3,:),10,scan.data_average(:),'fill','s');
	% xlabel('Qx');
	% ylabel('Qy');
	% zlabel('Qz');
	% view(7.7549,53.9161);
	% PROVED!

	%%
	% Small corrections to misalignment of the sample
	% Here the rocking curve should be introduced
	% alpha
	% beta

	% Gonphi correction
	alpha = 0.20;%11; Qx+Qz 
	beta = 0;%-6.7 Qz
	gamma = 0; % Qz

	for ii = 1:length(nanomax.gontheta)
	    % Rotations to bring the q vector into sample coordinate system
	    Rsh = [1         0                             0; % detector rotation around horizontal axis 
	           0         cosd(nanomax.gontheta(ii)+alpha) -sind(nanomax.gontheta(ii)+alpha);
	           0         sind(nanomax.gontheta(ii)+alpha)  cosd(nanomax.gontheta(ii)+alpha)]; 

	    Rsv = [cosd( nanomax.gonphi+beta)  0  sind( nanomax.gonphi+beta); % detector rotation around vertical axis 
	           0                           1  0;
	          -sind( nanomax.gonphi+beta)  0  cosd( nanomax.gonphi+beta)];

	    Rsz = [cosd(gamma) -sind(gamma)  0; 
	           sind(gamma)  cosd(gamma)  0;
	           0        0         1];

	    Rs = Rsh*Rsv*Rsz; 

	    % Sample coordinate system: accosiated with the ki
	    QSample(:,:,ii) = Rs*QLab;
	end

	scaleCoefficient = 1;

	qx = squeeze(QSample(1,:,:));
	qy = squeeze(QSample(2,:,:));
	qz = squeeze(QSample(3,:,:));

	dqX = scaleCoefficient*(2*pi*nanomax.detector_pitch/(nanomax.radius*wavelength));
	dqY = scaleCoefficient*(2*pi*nanomax.detector_pitch/(nanomax.radius*wavelength));
	dqZ = scaleCoefficient*(2*pi*nanomax.detector_pitch/(nanomax.radius*wavelength));

	[Qx,Qy,Qz] = meshgrid(min(qx(:)):dqX:max(qx(:)), min(qy(:)):dqY:max(qy(:)),min(qz(:)):dqZ:max(qz(:)));

	########################################################################################################
	
def interpolate_q_space():
	%% Scatter plot
	% treshold = 0.01;
	% 
	% data = data(:);
	% qx = qx(:);
	% qy = qy(:);
	% qz = qz(:);
	% 
	% qx(data<treshold*max(data)) = [];
	% qy(data<treshold*max(data)) = [];
	% qz(data<treshold*max(data)) = [];
	% data(data<treshold*max(data)) = [];
	% 
	% figure; scatter3(qx,qy,qz,10,double(data),'filled','s');

	
	tic         
	    F = TriScatteredInterp(qx(:),qy(:),qz(:),double(data(:))); 
	    lab_data = F(Qx,Qy,Qz);
	    lab_data(isnan(lab_data))=0;

	    scan.q_space.data(:,:,:) = lab_data;
	toc

	%%
	scan.q_space.qx = Qx;
	scan.q_space.qy = Qy;
	scan.q_space.qz = Qz;

	% save(scan_path,'scan');

	%%
	% 3D view
	% 
	figure;isosurface(Qx(1,:,1).*1e-10,Qy(:,1,1).*1e-10,squeeze(Qz(1,1,:)).*1e-10,squeeze(sum((scan.q_space.data),[4,5])));
	xlabel('Qx, [A^{-1}]');
	ylabel('Qy, [A^{-1}]');
	zlabel('Qz, [A^{-1}]');
	grid on;axis image

	%% Vis single points
	pos = [3,3];
	% save_path_figures = sprintf('/home/dzhigd/work/projects/CsPbBr3_NC_BCDI_NanoMAX/data/sample0609_%d/figures',scan_number);
	% mkdir(save_path_figures);

	% roi = [1,1];

	H_CsPbBr3 = 2*3.14/5.8795; % A^-1

	scale_min = 1.5;

	handles.figHandle = figure('Units','normalized','Position', [0.1 0.1 0.900 0.800]);

	sum_img = log10(squeeze(sum(scan.q_space.data,3)));
	min_val = scale_min*min(sum_img(sum_img(:)>-Inf));
	max_val = max(sum_img(:));

	subplot(1,3,1);imagesc(squeeze(Qx(1,:,1)).*1e-10,squeeze(Qy(:,1,1)).*1e-10,sum_img,[min_val max_val]); axis image; title('Integrated intensity');hold on;
	% xline(H_CsPbBr3,'--','color','white','linewidth',3);
	% yline(0,'--','color','white','linewidth',3);
	% xline(H_GaSb,'--','color','white');
	xlabel('Qx, [A^{-1}]');
	ylabel('Qy, [A^{-1}]');
	set(gca,'FontSize',12);
	colormap jet;

	sum_img = log10(squeeze(sum(scan.q_space.data,1)));
	min_val = scale_min*min(sum_img(sum_img(:)>-Inf));
	max_val = max(sum_img(:));

	subplot(1,3,2);imagesc(squeeze(Qz(1,1,:)).*1e-10,squeeze(Qx(1,:,1)).*1e-10,sum_img,[min_val max_val]); axis image; title('Integrated intensity');
	xlabel('Qz, [A^{-1}]');
	ylabel('Qx, [A^{-1}]');
	% xline(0,'--','color','white','linewidth',3);
	% yline(H_CsPbBr3,'--','color','white','linewidth',3);
	set(gca,'FontSize',12);
	colormap jet;

	sum_img = log10(squeeze(sum(scan.q_space.data,2)));
	min_val = scale_min*min(sum_img(sum_img(:)>-Inf));
	max_val = max(sum_img(:));

	subplot(1,3,3);imagesc(squeeze(Qz(1,1,:)).*1e-10,squeeze(Qy(:,1,1)).*1e-10,sum_img,[min_val max_val]); axis image; title('Integrated intensity');
	colormap jet;
	% xline(0,'--','color','white','linewidth',3);
	% yline(0,'--','color','white','linewidth',3);
	xlabel('Qz, [A^{-1}]');
	ylabel('Qy, [A^{-1}]');
	set(gca,'FontSize',12);
