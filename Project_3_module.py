# -*- coding: utf-8 -*-
"""
Project3_module.py
Created on Thu Nov 30 18:05:53 2023
This project examines ECG data of the heart from 4 different activities.
This module exists to be called in project3_script and further projects. 
It will take load in the data and filter it, and then detect the heartbeats. 
Following this, heart rate variability and it's frequency band power will also 
be calculated. This is done to assess the autonomic nervous system (ANS) and quantitatively 
estimate ANS activity.'

@authors: Cole Richardson and Thomas Bausman
"""

# import packages 
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft as fft
from scipy.signal import find_peaks

#%% Part 1: Collect and Load data 
    
# create a function that generates the 5 plots 
def plot_activities(data_1,data_2,data_3,data_4,time):
    '''
   The plot_activities function receives 4 1D data arrays, a 1D time array, 
   and an integer representing the sampling frequency. Using the inputs the 
   function will create 5 different plots on 1 figure. The first will be a 
   concatenated time array vs all of the data concatenated together. 
   The next 4 will be individual data arrays plotted against the time array. 

   Parameters
   ----------
   data_1 : Array of floats. 
       A 1D array of n terms where n is the voltage recorded from the arduino sensor for a particular activity.
   data_2 : Array of floats. 
       A 1D array of n terms where n is the voltage recorded from the arduino sensor for a particular activity.
   data_3 : Array of floats. 
       A 1D array of n terms where n is the voltage recorded from the arduino sensor for a particular activity.
   data_4 : Array of floats. 
       A 1D array of n terms where n is the voltage recorded from the arduino sensor for a particular activity.
   time : Array of floats. 
       A 1D array of n terms where n is the time recorded for each of the data samples.
   fs : Integer
       An integer representing the sampling frequency of the arduino sensor.

   Returns
   -------
   None.

   '''

    # assign fs 
    fs = 500
    # create figure 
    plt.figure(1,clear=True,figsize=(8,6))
    
    # concatenate the arrays 
    concatenated_data = np.concatenate([data_1,data_2, data_3, data_4])
    concatenated_time = np.arange(0,len(concatenated_data)/fs,1/fs)
    
    # plot concatenated data on one graph 
    plt.subplot(3,2,1)
    plt.subplot2grid((3,2),(0,0),colspan=2)
    plt.plot(concatenated_time,concatenated_data, label='Concatenated Data')
    # titles and legend 
    plt.title('Concatenated Activities')
    plt.xlabel('Time (sec)')
    plt.ylabel('Voltage (mV)')
    plt.tight_layout()
    
    # create individual plots (only five seconds)
    # plot activity 1 
    plt.subplot(3,2,3)
    plt.plot(time,data_1)
    # label graph 
    plt.title('Resting Activity')
    plt.xlabel('Time (sec)')
    plt.ylabel('Voltage (mV)')
    plt.xlim(250,255)  # show only 5 seconds of data
    plt.tight_layout()
    
    # plot activity 2 
    plt.subplot(3,2,4)
    plt.plot(time,data_2)
    # label graph 
    plt.title('Relaxing Activity')
    plt.xlabel('Time (sec)')
    plt.ylabel('Voltage (mV)')
    plt.xlim(250,255)  # show only 5 seconds of data
    plt.tight_layout()
    
    # plot activity 3 
    plt.subplot(3,2,5)
    plt.plot(time,data_3)
    # label graph 
    plt.title('Mental Activity')
    plt.xlabel('Time (sec)')
    plt.ylabel('Voltage (mV)')
    plt.xlim(250,255)  # show only 5 seconds of data
    plt.tight_layout()
    
    # plot activity 4 
    plt.subplot(3,2,6)
    plt.plot(time,data_4)
    # label graph 
    plt.title('Physical Activity')
    plt.xlabel('Time (sec)')
    plt.ylabel('Voltage (mV)')
    plt.xlim(250,255)  # show only 5 seconds of data
    plt.tight_layout()

    plt.show()  # show plots
    
    
#%% Part 2: Filter Your Data 

# create function to apply a filter to data and plot 
def filter_data(data, filt, title=''):
    '''
    The filter_data function will receive a 1D array of a data file, 
    a 1D array of a filter, a 1D time array, and a string of the desired 
    title of the plot. It will convolve the data set with the filter and 
    plot against time on a graph with the inputted title. The function will 
    return the filtered data set. 

    Parameters
    ----------
    data : Array of floats. 
        A 1D array of n terms where n is the voltage recorded from the arduino sensor for a particular activity.
    filt : Array of floats.
        A 1D array of n terms where n represents the values of a bandpass filter.
    time : Array of floats. 
        A 1D array of n terms where n is the time recorded for each of the data samples.
    title : String, optional
        String that represents the desired title of the graph. The default is ''.
    Returns
    -------
    convolved_data : Array of floats. 
        A 1D array of n terms where n represents the result of the data being convolved with the filter.

    '''

    #define fs
    fs= 500
    # convolve filter with data 
    convolved_data = np.convolve(data, filt, mode='same')
    #create time array
    time = np.arange(0,len(data)/fs,1/fs)
    
    # plot data
    plt.plot(time,convolved_data)
    plt.title(f'Filtered {title}Data')
    plt.xlabel('Time (sec)')
    plt.ylabel('Voltage (mV)')
    plt.xlim(250,255)
    plt.tight_layout()
    
    # show graph
    plt.show()
    
    return convolved_data
    

#%% Part 3: Detect Heartbeats 

# create function to detect and plot heartbeats 
def detect_heartbeat(ecg_data,numtaps,title=''):
    '''
    The detect_heartbeat function will input a 1D data array, an integer 
    representing the length of the filter, a 1D time array, and a desired 
    title for a graph. It will use the time and data arrays to find the 
    times of heartbeats and plot them on top of the data vs time graph.

    Parameters
    ----------
    ecg_data : Array of floats. 
        A 1D array of n terms where n is the voltage recorded from the arduino sensor for a particular activity.
    numtaps : Integer
        An integer representing the length of the filter array.
    time : Array of floats. 
        A 1D array of n terms where n is the time recorded for each of the data samples.
    title : String, optional
        A string representing the desired title for the graph. The default is ''.

    Returns
    -------
    None.

    '''

    
    #create fs
    fs = 500
    #create time array
    time = np.arange(0,len(ecg_data)/fs,1/fs)
    # Find peaks in the negative of the ECG signal (since find_peaks detects peaks)
    positive_peaks, _ = find_peaks(ecg_data,height=1.7, distance=numtaps)

    # Calculate the time of each heartbeat in seconds
    time_of_heartbeat = time[positive_peaks]

    # Plot the original ECG data with detected heartbeats
    plt.plot(time, ecg_data, label='ECG Data')
    plt.plot(time_of_heartbeat, ecg_data[positive_peaks], 'ro', label='Detected Heartbeats')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(f'{title} Data with Detected Heartbeats')
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% Getting heart beats
def get_heartbeats(ecg_data,numtaps,time):
    
    '''
   The get_heartbeats function will input a 1D data array, an integer 
   representing the length of the filter array, and a 1D time array. 
   It will use the data and time arrays to find the heartbeats and record 
   the time they occurred. It will return an array of these times. 

   Parameters
   ----------
   ecg_data : Array of floats. 
       A 1D array of n terms where n is the voltage recorded from the arduino sensor for a particular activity.
   numtaps : Integer
       An integer representing the length of the filter array.
   time : Array of floats
       A 1D array of n terms where n is the time recorded for each of the data samples.

   Returns
   -------
   time_of_heartbeat : Array of floats
       A 1D array of n terms where n is the time that heartbeats occurred in the data set.

   '''

    # Find peaks in the negative of the ECG signal (since find_peaks detects peaks)
    positive_peaks, _ = find_peaks(ecg_data,height=1.7, distance=numtaps)

    # Calculate the time of each heartbeat in seconds
    time_of_heartbeat = time[positive_peaks]

    return time_of_heartbeat

#%% plot histogram

def calculate_hrv_plot(time_of_heartbeat,activity_types):
    '''
    The calculate_hrv_plot will input a 1D array containing the times 
    heartbeats occurred and a list of the type of activity the data set 
    represented. It will calculate heart rate variability for the data set 
    and then plot it on a bar graph, labeled for the inputted activity type.

    Parameters
    ----------
    time_of_heartbeat : Array of floats.
        A 1D array of n terms where n represents the time recorded at each heartbeat from a specific activity's data set.
    activity_types : List
        A list of strings that represent the activity type for each of the data sets.

    Returns
    -------
    None 

    '''
    # find interval between the beats by differentiation
    time_intervals = np.diff(time_of_heartbeat)
    
    # hrv by calculating the standard deviation
    hrv = np.std(time_intervals)
    
    plt.bar(activity_types, [hrv])
    plt.xlabel('Activity Data')
    plt.ylabel('Heart Rate Variability (SDRR)')
    plt.title('Heart Rate Variability for Different Activities')
    plt.show()

#%% Part 5: Get HRV Frequency Band Power

def get_power_spectrum(interpolated_data, resolution):
    """
    The get_power_spectrum function takes in data sampled at a dt of 0.1 along with its resolution
    to return the power spectrum of this data array. It will also automatically graph the data in the frequency domain and separate LF and HF
    with different colors. low frequency (LF) is defined to be 0.04-0.15 and high frequency (HF) is defined to be 0.15-0.4. Along with the power spectrum
    The function will also return the averages of the LF and HF for the data. The graph is zoomed in to 0-.5 Hz on the x-axis
    and 10 units of power in the y-axis.


    Parameters
    ----------
    interpolated_data : Array of float
       This input expects an array of IBI data that has already been interpolated
       to a sample rate of dt=0.1
   resolution : integer
       This integer should be the last number in your time array to normalize your
       data for a more accurate result. This makes it so that it doesn't matter if the
       data sets are different lengths


     Returns
    -------
    power : array of float
        This is the power spectrum of the interpolated data. It is returned in
        arbitrary power units and comes with a graph separating LF and HF
    avg_lf : integer
        Average power of LF
    avg_hf : integer
        Average power of HF
"""
    #define dt
    dt = 0.1
    #take real fourier transform of interpolated data
    fft_result = fft.rfft(interpolated_data)
    
    #create frequency domain
    x_f = fft.rfftfreq(len(interpolated_data),dt)
    
    # get the max power of the array
    power = np.square(np.abs(fft_result)) / resolution
    
    power[0] = 0 #remove spike at 0
    #plot data
    plt.plot(x_f, power)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (A.U.)')
    plt.xlim(0,0.5)
    plt.ylim(0,10)
    
    #set cuttoff points in Hz
    lf_start = .04
    lf_cut = .15
    hf_start= .15
    hf_cut = .4
    
    #color the graph and highlight LF and HF
    
    out_range = np.where((x_f >= 0) & (x_f <= lf_start))[0] #data below low cutoff
    
    lf_range = np.where((x_f >= lf_start) & (x_f <= lf_cut))[0] #LF data between .04 and .15Hz

    hf_range = np.where((x_f >= hf_start) & (x_f <= hf_cut))[0] #HF data between .15 and .4Hz
    
    plt.fill_between(x_f[out_range], power[out_range], color='grey', alpha=0.4) #fill in with color
    
    plt.fill_between(x_f[lf_range], power[lf_range], color='palegreen', alpha=0.4, label='LF Area') #fill in with color
    
    plt.fill_between(x_f[hf_range], power[hf_range], color='moccasin', alpha=0.7, label='HF Area') #fill in with color
    plt.legend()
    
    avg_lf = np.average(power[lf_range])
    
    avg_hf = np.average(power[hf_range])
    
    return power, avg_lf, avg_hf