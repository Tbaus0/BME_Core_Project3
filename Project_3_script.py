# -*- coding: utf-8 -*-
"""
Project3_script.py
Created on Thu Nov 30 18:05:48 2023

This project examines ECG data of the heart from 4 different activities.
This script will call functions from project_3_module to filter an ECG dataset, detect the heartbeats,
calulate heart rate variability(HRV), and frequency band power will also be calculated/compared. 
This is done to assess the autonomic nervous system (ANS) and quantitatively estimate ANS activity.
Through activity of the sympathetic nervous system and parasympathetic nervous syster identified by 
their different HRV.

@authors: Cole Richardson and Thomas Bausman
"""

# import packages 
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft as fft
from scipy import signal
import Project_3_module as p3m

#%% Part 1: Collect and Load Data 

# load data 
resting_data = np.loadtxt('p3_resting_meg.txt')
relaxing_data = np.loadtxt('p3_relaxing_meg.txt')
mental_data = np.loadtxt('p3_mentally_stress_meg.txt')
physical_data = np.loadtxt('p3_physical_stress_meg.txt')

# trim data sets to be the same length 
# they will be the same length as the lowest (physical_data)
resting_data = resting_data[:-21389]
relaxing_data = relaxing_data[:-22823]
mental_data = mental_data[:-835]
mental_data = np.flip(mental_data)

# divide data sets by conversion factor of 204.6 for conversion between 1023 bits and 4 volts 
resting_data = resting_data * (1/204.6)
relaxing_data = relaxing_data * (1/204.6)
mental_data = mental_data * (1/204.6)
physical_data = physical_data * (1/204.6)

# set sampling frequency 
fs = 500 # normal arduino code samples at 500 we did 250
t = np.arange(0,len(physical_data)/fs,1/fs) #use physical_data as a template for time array because lowest length of data

# generate plots vs time using function 
p3m.plot_activities(resting_data, relaxing_data, mental_data, physical_data,t)

#%% Part 2: Filter Your Data 

numtaps = 250 # half the sampling frequency 
fc1 = .5 #Beginning of frequency range
fc2 = 50 #Cutoff frequency
window = 'hann' #Filter shape
filter_type = 'bandpass'  # filter type
signal_filter = signal.firwin(numtaps,[fc1,fc2],window=window,pass_zero=filter_type,fs=fs)  # create filter using scipy 
signal_filter_freq = fft.rfft(signal_filter) #bring filter to frequency domain

# create filtered time array
time_filter = np.arange(0,len(signal_filter)/fs,1/fs)
# create filtered frequency array
freq_filter = fft.rfftfreq(len(signal_filter),1/fs)

# create side by side plot of the filters impulse and frequency response 
plt.figure(2,clear=True)

# plot filter's impulse response 
plt.subplot(1,2,1)
plt.plot(time_filter, signal_filter)
plt.title('Filter Impulse Response')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude (A.U.)')


# plot filter's frequency response 
plt.subplot(1,2,2)
plt.plot(freq_filter,signal_filter_freq)
plt.title('Filter Frequency Response')
plt.xlabel('Freq (Hz)')
plt.ylabel('Amplitude (A.U.)')
plt.tight_layout()

# plot before and after of data with filter
plt.figure(3,clear=True)
# plot the before
plt.subplot(1,2,1)
plt.title('Unfiltered Resting Data')
plt.plot(t,resting_data)
plt.xlim(250,255) #crop to 5 seconds
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
# plot the after
plt.subplot(1,2,2)
p3m.filter_data(resting_data,signal_filter)
plt.tight_layout()


# filter and plot all data  
plt.figure(4,clear=True)
# filter and plot resting_data
plt.figure(4,clear=True)
plt.subplot(2,2,1)
filtered_resting = p3m.filter_data(resting_data,signal_filter,'Resting ')

# filter and plot relaxing_data
plt.subplot(2,2,2)
filtered_relaxing = p3m.filter_data(relaxing_data,signal_filter,'Relaxing ')

# filter and plot mental_data
plt.subplot(2,2,3)
filtered_mental = p3m.filter_data(mental_data,signal_filter,'Mental ')

# filter and plot physical_data
plt.subplot(2,2,4)
filtered_physical = p3m.filter_data(physical_data,signal_filter,'Physical ')

#%% Part 3: Detect Heartbeats

# create figure of plots with heartbeats detected 
plt.figure(5,clear=True,figsize=(8,6))

# plot filtered resting_data
plt.subplot(2,2,1)
p3m.detect_heartbeat(filtered_resting, numtaps, 'Resting ')
plt.xlim(250,255)

# plot filtered relaxing_data
plt.subplot(2,2,2)
p3m.detect_heartbeat(filtered_relaxing, numtaps, 'Relaxing ')
plt.xlim(250,255)

# plot filtered mental_data
plt.subplot(2,2,3)
p3m.detect_heartbeat(filtered_mental,numtaps, 'Mental ')
plt.xlim(250,255)

# plot filtered physical_data
plt.subplot(2,2,4)
p3m.detect_heartbeat(filtered_physical, numtaps, 'Physical ')
plt.xlim(250,255)

#%% part 4: Heart Rate Variability

#get times for all heartbeats for every activity
rest_heartbeat_times = p3m.get_heartbeats(resting_data, numtaps, t)
relax_heartbeat_times = p3m.get_heartbeats(relaxing_data, numtaps, t)
mental_heartbeat_times = p3m.get_heartbeats(mental_data, numtaps, t)
physical_heartbeat_times = p3m.get_heartbeats(physical_data, numtaps, t)

# create new figure for bar graph of HRV
plt.figure(6,clear=True)

# use function to calculate HRV and plot as a histogram
p3m.calculate_hrv_plot(mental_heartbeat_times, 'Mental')
p3m.calculate_hrv_plot(physical_heartbeat_times, 'Physical')
p3m.calculate_hrv_plot(rest_heartbeat_times, 'Resting')
p3m.calculate_hrv_plot(relax_heartbeat_times, 'Relaxing')

#assign dt for x-axes
dt= 0.1

#Get iterpolated array in correct dt for graphing
rest_regular_time = np.arange(0,len(np.diff(rest_heartbeat_times)),dt) #create time array for activity
ibi_rest = np.interp(rest_regular_time, rest_heartbeat_times[1:], (np.diff(rest_heartbeat_times)))

#Get iterpolated array in correct dt for graphing
relax_regular_time = np.arange(0,len(np.diff(relax_heartbeat_times)),dt) #create time array for activity
ibi_relax = np.interp(relax_regular_time,relax_heartbeat_times[1:],(np.diff(relax_heartbeat_times)))

#Get iterpolated array in correct dt for graphing
mental_regular_time = np.arange(0,len(np.diff(mental_heartbeat_times)),dt) #create time array for activity
ibi_mental = np.interp(mental_regular_time,mental_heartbeat_times[1:],(np.diff(mental_heartbeat_times)))

#Get iterpolated array in correct dt for graphing
physical_regular_time = np.arange(0,len(np.diff(physical_heartbeat_times)),dt) #create time array for activity
ibi_physical = np.interp(physical_regular_time,physical_heartbeat_times[1:],(np.diff(physical_heartbeat_times)))

#%% Part 5: Get HRV Frequency Band Power

#create resolution integer to normalize data
resolution = rest_regular_time[-1]

#create new figure for plotting at correct size
plt.figure(8,clear=True,figsize=(10,7))
#use function to plot and calculate the power spectrum with averages
plt.subplot(2,2,1)
plt.title('Resting Power Spectrum')
ibi_rest_power, avg_rest_lf, avg_rest_hf = p3m.get_power_spectrum(ibi_rest, resolution)

#use function to plot and calculate the power spectrum with averages
plt.subplot(2,2,2)
plt.title('Relaxing Power Spectrum')
ibi_relax_power, avg_relax_lf, avg_relax_hf = p3m.get_power_spectrum(ibi_relax, resolution)

#use function to plot and calculate the power spectrum with averages
plt.subplot(2,2,3)
plt.title('Mental Power Spectrum')
ibi_mental_power, avg_mental_lf, avg_mental_hf = p3m.get_power_spectrum(ibi_mental, resolution)

#use function to plot and calculate the power spectrum with averages
plt.subplot(2,2,4)
plt.title('Physical Power Spectrum')
ibi_physical_power, avg_physical_lf, avg_physical_hf = p3m.get_power_spectrum(ibi_physical, resolution)
plt.tight_layout()

#create new figure
plt.figure(9,clear=True)
plt.ylabel('LF/HF Ratio')
plt.title('Ratios of Activites Power Spectrums')

#calculate ratios of LF/HF and plot
rest_ratio = avg_rest_lf/avg_rest_hf
plt.bar('Resting', rest_ratio, color='brown')

relax_ratio = avg_relax_lf/avg_relax_hf
plt.bar('Relaxing', relax_ratio, color='indianred')

mental_ratio = avg_mental_lf/avg_mental_hf
plt.bar('Mentally Stressed', mental_ratio, color='lightcoral') #lighter colors indicate lower ratio

physical_ratio = avg_physical_lf/avg_physical_hf
plt.bar('Physically Active', physical_ratio, color='firebrick')

plt.tight_layout()