import numpy as np

import matplotlib.pyplot as plt

from spacepy import pycdf


def BackReduction(raw_PSD, BEflag, Burst_flag):

    ''' Ths class performs the reduction on PSDs, using previously determined thresholds ''' 

    if BEflag == "B" and Burst_flag == False:

        noise_file ='/data/hpcdata/users/rablack75/CountSurvey/data/Background_thresholds/BThreshold_201209-201909.dat'
        # load the power and std data
        thresholds = np.loadtxt(noise_file)[:,1]
        peaks = np.loadtxt(noise_file)[:,0]
        malas_thresh = peaks[0] - thresholds[0]

        thresholds = peaks + malas_thresh
        thresholds = 10**(thresholds) 

    
    elif BEflag == "E" and Burst_flag == False:

        noise_file ='/data/hpcdata/users/rablack75/CountSurvey/data/Background_thresholds/BThreshold_201209-201909.dat'

            # load the power and std data
        thresholds = np.loadtxt(noise_file)[:,1]
        
        thresholds = 10**(thresholds) 


    elif Burst_flag == 'K':

        noise_file = '/data/hpcdata/users/rablack75/CountSurvey/data/Background_thresholds/BurstThreshK.dat'
        # load the power and std data
        thresholds = np.loadtxt(noise_file)
    
        thresholds = 10**(thresholds)
 
    
    elif Burst_flag == 's':

        noise_file = '/data/hpcdata/users/rablack75/CountSurvey/data/Background_thresholds/BurstThreshS.dat'
        # load the power and std data
        thresholds = np.loadtxt(noise_file)
    
        thresholds = 10**(thresholds) 

    elif BEflag == "EH" and Burst_flag == False:

        noise_file ='/data/emfisis_burst/wip/rablack75/BackReduction/BThreshold_201209-201909.dat'

            # load the power and std data
        thresholds = np.loadtxt(noise_file)[:,1]
        
        thresholds = 10**(thresholds) 

    elif BEflag == "EW" and Burst_flag == False:

        noise_file ='/data/emfisis_burst/wip/rablack75/BackReduction/Thresholds/EWThreshold_201209-201909.dat'

            # load the power and std data
        thresholds = np.loadtxt(noise_file)[:,1]
        
        thresholds = 10**(thresholds)
        

    # Do the reduction
    reduced_PSD = np.zeros(np.shape(raw_PSD))
    for i in range(len(thresholds)):

        for j in range(np.shape(raw_PSD)[0]):

            if (raw_PSD[j,i] > thresholds[i]):
                reduced_PSD[j,i] = raw_PSD[j,i]
            else:
                reduced_PSD[j,i] = 0.

    return reduced_PSD


def HFRback(raw_PSD):

    noise_file = pycdf.CDF('/data/hpcdata/users/rablack75/CountSurvey/data/Background_thresholds/sn_threshold_RBSP-A.cdf')

    # finding the background noise from the thresholds and snr
    
    hfr_noise =[]
    for i in range(len(noise_file['frequencies'])):
        hfr_noise.append((noise_file['thresholds'][i]))
    print(hfr_noise)

    psd_reduced = np.zeros(np.shape(raw_PSD))
    print(np.shape(raw_PSD))
    for i in range(len(hfr_noise)):
        for j in range(np.shape(raw_PSD)[0]):
        
            if raw_PSD[j,i] < hfr_noise[i]:
                psd_reduced[j,i]=0
            else:
                psd_reduced[j,i]=raw_PSD[j,i]

    return psd_reduced

def MalaspinaReduction(raw_PSD, BEflag, Burst_flag):

    ''' Ths class performs the reduction on PSDs, using previously determined thresholds ''' 

    if BEflag == "B" and Burst_flag == False:

        noise_file ='/data/hpcdata/users/rablack75/CountSurvey/data/Background_thresholds/BThreshold_201209-201309.dat'
        # load the power and std data
        thresholds = np.loadtxt(noise_file)[:,1]
        print(thresholds)
        peaks = np.loadtxt(noise_file)[:,0]

        # now isolate the peak - threshold for the first frequency band - this is the contstant Malaspina et al 2-18 use
        malaspina_thresh = peaks[0] - thresholds[0] 

        # add to all distribution peaks
        thresholds = peaks + malaspina_thresh
        print(thresholds)
        thresholds = 10**(thresholds)

    
    elif BEflag == "E" and Burst_flag == False:

        noise_file ='/data/hpcdata/users/rablack75/CountSurvey/data/Background_thresholds/BThreshold_201209-201309.dat'

        # load the power and std data
        thresholds = np.loadtxt(noise_file)[:,1]
        peaks = np.loadtxt(noise_file)[:,0]
        
        # now isolate the peak - threshold for the first frequency band - this is the contstant Malaspina et al 2-18 use
        malaspina_thresh = thresholds[0] - peaks[0]

        # add to all distribution peaks
        thresholds = peaks + malaspina_thresh
    
        thresholds = 10**(thresholds) 


    elif Burst_flag == 'K':

        noise_file = '/data/hpcdata/users/rablack75/CountSurvey/data/Background_thresholds/BurstThreshK_Malas.dat'
        # load the power and std data
        thresholds = np.loadtxt(noise_file)
    
        thresholds = 10**(thresholds)
 
    
    elif Burst_flag == 's':

        noise_file = '/data/emfisis_burst/wip/rablack75/BackReduction/Thresholds/BurstThreshSMalas.dat'

        # load the power and std data
        thresholds = np.loadtxt(noise_file)
    
        thresholds = 10**(thresholds)
        

    # Do the reduction
    reduced_PSD = np.zeros(np.shape(raw_PSD))
    print(np.shape(raw_PSD))
    for i in range(len(thresholds)):

        for j in range(np.shape(raw_PSD)[0]):

            if (raw_PSD[j,i] > thresholds[i]):
                reduced_PSD[j,i] = raw_PSD[j,i] - thresholds[i]
            else:
                reduced_PSD[j,i] = 0.

    return reduced_PSD

                
        


def FeatureReduction(raw_PSD, ellip):

    ''' Ths class performs the reduction on PSDs, using previously determined thresholds ''' 

    noise_file ='/data/hpcdata/users/rablack75/CountSurvey/data/Background_thresholds/BThreshold_201209-201309.dat'
    # load the power and std data
    thresholds = np.loadtxt(noise_file)[:,1]
    peaks = np.loadtxt(noise_file)[:,0]
    malas_thresh = peaks[0] - thresholds[0]

    thresholds = peaks + malas_thresh
    thresholds = 10**(thresholds) #+ 0.57713065)


    # Do the reduction
    reduced_ellip = np.zeros(np.shape(ellip))
    for i in range(len(thresholds)):

        for j in range(np.shape(raw_PSD)[0]):

            if (raw_PSD[j,i] > thresholds[i]):
                reduced_ellip[j,i] = ellip[j,i]
            else:
                reduced_ellip[j,i] = np.nan

    return reduced_ellip
