#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.ndimage import median_filter
import numpy as np
import cmath
from scipy.stats import chi2

#For DEBUG
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#***********************************************************************************************
def find_shocks(ref, refroi, shot, phaseroi, time, mult, W, F, p_thresh, ds):

	#---------------------------------------------------------------
	#Find fringe carrier wavelength (used to define FFT filter)
	#---------------------------------------------------------------
	left      = refroi[0]
	width     = refroi[1]
	top       = refroi[2]
	height    = refroi[3]

	refroi_im = ref[top:(top+height), left:(left+width)]

	refroi_im_mean = np.mean(refroi_im, axis=1)

	carrier_wave   = find_carrier(refroi_im_mean)

	#---------------------------------------------------------------
	#Analyse phase roi
	#---------------------------------------------------------------
	left      = phaseroi[0]
	width     = phaseroi[1]
	top       = phaseroi[2]
	height    = phaseroi[3]

	phaseroi_shot = shot[top:(top+height), left:(left+width)]
	
	#Use FFT method to find space averaged phase and amplitude
	phase_shot, amplitude_shot = analyze_roi(phaseroi_shot, carrier_wave)

	#Subtract reference fringe phase
	phaseroi_ref = ref[top:(top+height), left:(left+width)]
	phase_ref, _ = analyze_roi(phaseroi_ref, carrier_wave)

	phase = (phase_shot - phase_ref) * mult

	#Downsample phase and amplitude signals priot to shock search
	time_roi = time[left:(left+width)]
	tmin     = time_roi[0]
	tmax     = time_roi[-1]

	time_ds  = np.linspace(tmin, tmax, num=ds)

	phase_ds          = np.interp(time_ds, time_roi, phase)
	amplitude_shot_ds = np.interp(time_ds, time_roi, amplitude_shot)

	#Run statistical algorithm on phase
	t_list, S_t_list, p_t_list = p_t(phase_ds, W, F, 'UP')

	#Find shock locations from p_t crossings
	phase_shocklist = find_crossings(t_list, p_t_list, p_thresh)

	#Run statistical algorithm on amplitude
	t_list, S_t_list, p_t_list = p_t(amplitude_shot_ds, W, F, 'UP_OR_DOWN')

	#Find shock locations from p_t crossings
	amplitude_shocklist = find_crossings(t_list, p_t_list, p_thresh)

	shocklist = np.append(phase_shocklist, amplitude_shocklist)
	
	if len(shocklist) == 0:
		print('No shocks found')
		exit()

	#Clean up list
	tolerance    = 2*F
	shocklist, _ = clean_list(shocklist, tolerance)

	#Return shocktime(s)
	shocktimes = time_ds[shocklist]

	return shocktimes

#***********************************************************************************************
def find_carrier(lineout):

	n_points = len(lineout)
	
	#---------------------------------------------------------------
	#Find approximate carrier wavelength from FFT spectrum
	#---------------------------------------------------------------
	lineout_fft     = np.fft.fft(lineout)
	freq            = np.fft.fftfreq(lineout.shape[-1])
	flipped         = np.abs(lineout_fft[:n_points//2:-1])
	carrier_loc     = np.argmax(flipped) + 1
	carrier_wave_fft = round(1./freq[carrier_loc])

	return carrier_wave_fft

#***********************************************************************************************
def analyze_roi(phaseroi, carrier_lamda):

	#---------------------------------------------------------------
	#Apply median filter
	#---------------------------------------------------------------
	phaseroi = median_filter(phaseroi, size=5)

	#-------------------------------------------------
	#Clip ROI height to have integer number of fringes
	#-------------------------------------------------
	nfringes = int(phaseroi.shape[0] / carrier_lamda)

	#---------------------------------------------------------------
	#Define truncated ROI
	#---------------------------------------------------------------
	n_rows   = int(round(nfringes*carrier_lamda))
	print(n_rows, '+++')
	roitrunc = phaseroi[0:n_rows,:]

	n_cols   = phaseroi.shape[1]

	if n_rows == 0: #Failed to find carrier wavelength

		print('Failed to find fringe carrier wavelength. Check fringe contrast and/or reference ROI')
		exit()

	#---------------------------------------------------------------
	#Find carrier location in FFT array
	#---------------------------------------------------------------
	carrier_loc = round(n_rows/carrier_lamda)

	#---------------------------------------------------------------
	#Define FFT filter
	#---------------------------------------------------------------
	f_min = carrier_loc - 1
	f_max = carrier_loc + 1
		
	fft_filter = [(f>=f_min and f<=f_max) for f in np.arange(n_rows)]

	#---------------------------------------------------------------
	#Retrieve amplitude and phase
	#--------------------------------------------------------------	
	roi_filtered = []	
	
	for index in np.arange(n_cols):
		col          = roitrunc[:, index]		
		col_fft      = np.fft.fft(col)	
		col_fft_filt = col_fft * fft_filter
		col_ifft     = np.fft.ifft(col_fft_filt)
	
		roi_filtered.append(col_ifft)
	
	#Transpose image to right way round
	roi_filtered = np.transpose(roi_filtered) #Complex numbers

	#Record amplitude
	amplitude = np.mean(np.abs(roi_filtered), axis=0)

	phase = []

	for row in roi_filtered:
	
		shifted_row = np.roll(row, -1)
		phi_row     = np.zeros(n_cols)
	
		for n in np.arange(n_cols):
			ratio       = shifted_row[n] / row[n]
			dphi        = cmath.phase(ratio)
			phi_row[n]  = dphi
	
		phi_row    = np.roll(phi_row, 1)
		phi_row[0] = 0.0

		phi_cumsum = np.cumsum(phi_row)
				
		phase.append(phi_cumsum)
	
	#Unwrap phase image in time direction
	phase = np.unwrap(phase, axis=1)
			
	#Unwrap phase image in the spatial direction
	#phase = np.unwrap(phase, axis=0)
						
	#Average in spatial direction	
	phase = np.mean(phase, axis=0)

	return phase, amplitude

#***********************************************************************************************
#Jonty's statistical algorithm
#***********************************************************************************************
def p_t(sig_t, W, F, change_dir):

	siglength = len(sig_t)
	
	delta_t = np.diff(sig_t) #first element is delta_1...
	
	#Stats tracked
	S_t_list = []
	t_list   = np.arange(W, siglength-F)
	p_t_list = []
	
	for t in np.arange(W, siglength-F):
		
		#**********************************************************
		#S_t
		#**********************************************************
		#Null hypothesis
		past    = delta_t[t-W:t]
		future  = delta_t[t:t+F]

		delta_0 = np.append(past, future)		
		muhat_0 = np.mean(delta_0)
		RSS_0   = np.sum(np.square(delta_0 - muhat_0))
		s2hat_0 = RSS_0 / (W+F)
			
		#G stat
		muhat_W = np.mean(past)
		RSS_W   = np.sum(np.square(past - muhat_W))
			
		muhat_F = np.mean(future)
		RSS_F   = np.sum(np.square(future - muhat_F))
			
		s2hat   = (RSS_W + RSS_F) / (W+F)
		
		#Adapt if only looking for positive changes	
		if muhat_F <= muhat_W and change_dir == 'UP':
			
			s2hat = s2hat_0
			
		S_t = (W+F) * (np.log(s2hat_0)-np.log(s2hat))
				
		#Record S_t
		S_t_list.append(S_t)
		
		#**********************************************************
		#p_t
		#**********************************************************
		p_t = 1. - chi2.cdf(x=S_t, df=1)
		
		#Record p_t
		p_t_list.append(p_t)
		
	return t_list, S_t_list, p_t_list


#***********************************************************************************************
#Find negative going zero crossings in p_t
#***********************************************************************************************
def find_crossings(t_list, p_t, p_thresh):

	threshcrossed = np.where(np.array(p_t) < p_thresh, 1, 0)
	crossings     = np.where(np.diff(threshcrossed, prepend=0)==1)
	t_shock_list  = t_list[crossings[0]]

	return t_shock_list

#***********************************************************************************************
#Clean up list of shock locations (discovered by shock detection algorithm)
#***********************************************************************************************
def clean_list(shocklist, tolerance):

	shocklist_sorted = np.sort(shocklist)

	diff = (np.abs(np.diff(shocklist_sorted)) > tolerance).astype(int)

	index = np.where(diff==1)[0]

	start = 0

	cleaned_list = []
	weights_list = []

	for group in index:

		end = group + 1

		current_group = shocklist_sorted[start:end]

		cleaned_list.append(round(np.mean(current_group)))

		#Weight: how many time shocks found
		current_weight = len(current_group)
		weights_list.append(current_weight)

		start = end

	#Last group
	last_group = shocklist_sorted[start:]

	cleaned_list.append(round(np.mean(last_group)))

	last_weight = len(last_group)
	weights_list.append(last_weight)

	return cleaned_list, weights_list

#***********************************************************************************************
#Used for visualizing ROI only. Can be omitted
#***********************************************************************************************
def show_roi(image, roi, name):

	left      = roi[0]
	width     = roi[1]
	top       = roi[2]
	height    = roi[3]

	fig = plt.figure(figsize = (6,6))
		
	plt.imshow(image)
	plt.plot((left, left + width),(top, top), color='red')              #top
	plt.plot((left, left+ width),(top+height, top+height), color='red') #bottom
	plt.plot((left, left),(top, top+height), color='red')               #left
	plt.plot((left+ width, left+width),(top, top+height), color='red')  #right
		
	plt.title(name)
	plt.xlabel('Time (pix)')
	plt.ylabel('Position (pix)')
		
	plt.show()
		
	plt.savefig(name + '_roi')
		
# 	plt.clf()
