# numerical essentials 
import numpy as np

# for cdf reading
from spacepy import toolbox
#spacepy.toolbox.update(leapsecs=True)
from spacepy import pycdf
import h5py
import csv

# other numerical tools
import os
from datetime import datetime,timedelta,date,time
import calendar

# my own misc. functions
import global_use_rocky as gl
import funcs_analysis as fa
import find_properties as fp
import sys


""" This code decides whether, for a given timestamp, the spacecraft is within the plasmasphere or the plasmatrough. It does this by considering each half (or <half orbit in edge cases) independetly.
The region can then be decided based upon one of three methods. In decreasing priority, these are:

1. via a density gradient > 5 within an Lstar range of 0.5 (Moldwin et al., 2002). This method identifies what is knwon as the plasmapause: a suddden and obvious density drop as you leave the plasmasphere
It is an ideal case and this density 'knee' is not always present and this method may miss more gradual density changes from plasmasphere to plasmatrough, or when density data is sparse. Therefore this
method should be used in the first instance, before moving onto methods 2 or 3. 

2. via identifying ECH waves.  Electron cyclotron harmonic (ECH) waves are emissions within multiples of the electron gyrofrequency, observed outside of the plasmasphere and often terminate at the plasmapause
boundary (Meredith et al., 2004). This can be incredibly useful when there is a lack of density data, as the sudden lack of emissions is indicative of a density gradient. However, ECH waves are not always observed, 
and in some cases, there is leakage into the plasmasphere (Liu et al., 2020). The amplitude required for ECH wave emission signal is set at 10^-4 V/m between fce<f<5fce.

3. via a density threshold. This is the method of last resort. The plasmapause density can drop from plasmasphere to plasmatrough densities(from 100cm−3 to 1) gradually over a width greater than that defined by the
gradient threshold. To account for this, a plausible absolute density between 30-100cm3 is often chosen as an alternative. The density threshold chosen is relativelyarbitrary and is therefore a non-case
specific estimate that may not positionally agree with results found in method 1, or 2. To utilise the density gradient as far as possible (Li et al., 2010a), the density threshold is taken as the largest value of 
50cm−3 and n = 10(6.6/L)^4, where the latter is from an empirical model found using satellite measurements (Sheeley et al., 2001).
"""

""" In the following code, a flag is used to indicate on each half orbit which method was used - 1, 2 or 3, Other flags indicating no crossing was found are either 4: always stayed in plasmapasphere on orbit, or 0: it was undertermined on given orbit via all methods.
"""


""" necesssary functions """


def find_ech(gyro_list,gyro_epoch,WFR_E,HFR_E,WFR_epoch,HFR_epoch,WFR_freqeucny,HFR_frequency,time):
        """This function integrates the electric field spcetrum at each time step to find ECH wave amplitude
        Input: gyrofrequency, gyrofrequency_epoch, WFR E spectra, HFR spectra, HFR epoch, WFR ftequency, HFR frequency, timestamp
        Output: E amplitude (V/m) """
        # define lower limit of HFR (in Hz), below which we must add in the WFR E field
        lower_lim = 10**(4)

        # find gyrofrequency 
        gyro = gyro_list[gl.find_closest(gyro_epoch,time)[1]]

        # finding HFR time/index asscoiated wth density stamp
        HFR_time,HFR_index = gl.find_closest(HFR_epoch,time)

        # Now decide whether we fall off the HFR lower limit into the WFR bounds: finding WFR data index for HFR lower limit
        index_lowerlim = gl.find_closest(WFR_freqeucny,lower_lim)[1]

        if (gyro < lower_lim ):
            frequency_add=[]
            spectra_add=[]
            # Find frequency in WFR data that corresponds closest to this
            # find all frequencies between integration limit...
            # ... and where HFR lower limit starts

            index_freqwfr =gl.find_closest(WFR_freqeucny,gyro)[1]
  
            index_epochwfr = gl.find_closest(WFR_epoch,HFR_time)[1]

        
        # now find all of the frequencies between this and lower limit

            for l in range(index_freqwfr,index_lowerlim-1,1):
                # add to the spectra and frequency list
                frequency_add.append(WFR_freqeucny[l])
                spectra_add.append(WFR_E[index_epochwfr,l])
            frequency_edit =np.concatenate((frequency_add,HFR_frequency))
            spectra_edit=np.concatenate((spectra_add,HFR_E[HFR_index,:]))

        # if not, carry on with just HFR - starting from index we are at the gyrofrequency
        else:
            frequency_edit =HFR_frequency
            spectra_edit=HFR_E[HFR_index,:]

        # Integrate using the Trapezium rule
        int = 0
        for j in range(len(spectra_edit)-1):
            # set bounds for the frequency range between 1 and 4 fce
            if (1.*gyro < frequency_edit[j] < 5.*gyro): # integration lmits between fce < f < 2*fce
                # set upper limit above which we see a significant instrument noise
                if (frequency_edit[j]<3*10**5)and(frequency_edit[j+1]<3*10**5):
                    int = int + 0.5*(spectra_edit[j+1]+spectra_edit[j])*(frequency_edit[j+1]-frequency_edit[j])
        
        # square root to get the amplitude (V/m)
        int = np.sqrt(int)
        return int



def outwards_journ(input_dict, gee_index):
    """ Finding the plasmapause crossings on half-orbits out of the plasmapause """

    def density_thresh(fpe_epoch, lanl_epoch, MLT, MLAT_N, MLAT_S, Lstar,density,d_perigee,d_apogee,epoch_omni,AE,epoch_omni_low,Dst,Kp):
        """ density threshold: max of 10*(6.6/L)^(-4) and 50 cm-3 """

        # all initial values of plasmapause crossings set to np.nan before finding
        pp_type = None
        pp_L = np.nan
        pp_time = np.nan
        pp_AE = np.nan
        pp_AEStar = np.nan
        pp_MLT = np.nan

        # cycle through the density data on that half orbit
        for k in range(d_perigee,d_apogee):
            
            # find correspinding Lstar and MLT values from LANL data
            findLstar= fp.FindLANLFeatures(fpe_epoch[k], lanl_epoch, MLT, MLAT_N, MLAT_S, Lstar)
            Lstar_stamp = findLstar.get_Lstar
            MLT_stamp = findLstar.get_MLT   

            # find corresponding AE and AE* from omni data
            findOmniFeats = fp.FindOMNIFeatures(fpe_epoch[k], epoch_omni, epoch_omni_low, AE, Kp, Dst)
            AE_stamp = findOmniFeats.get_AE
            AE_star = findOmniFeats.get_AEstar

            # find the density threshold - largest between 50 and empirical L relationship for threshold
            thresh = np.max([50,10*(6.6/Lstar_stamp)**4])
    
            # if current density below threshold, have found plasmapause and therefore update crossing details and leave loop
            if density[k]<thresh:
    
                pp_type = 3
                pp_L = Lstar_stamp
                pp_time = fpe_epoch[k]
                pp_MLT = MLT_stamp
                pp_AE = AE_stamp
                pp_AEStar = AE_star
                break
            
            # otherwie, have never crossed a boundary and still in plasmapause so set crossing type to 4
            else:

                pp_type = 4
                pp_L = np.nan
                pp_time = np.nan
                pp_MLT = MLT_stamp
                pp_AE = AE_stamp
                pp_AEStar = AE_star

        return pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEStar


    # find the index correpsonding to the apogee and perigee in the density data
    perigee = input_dict["all_times"][gee_index]
    apogee = input_dict["all_times"][gee_index+1]
    
    # find indicies for apogee and perigee in density data
    d_perigee = gl.find_closest(input_dict["fpe_epoch"],perigee)[1]
    d_apogee = gl.find_closest(input_dict["fpe_epoch"],apogee)[1]

    # set initial values of plasmapause crossing details to np.nan, and crossing type to undetermined
    pp_L = np.nan
    pp_type = 0
    pp_time = np.nan
    pp_AE = np.nan
    pp_AEstar = np.nan
    pp_MLT = np.nan

    # set flags for a crossing due to ECH waves and the density threshold as not found (False)
    stopECH = False
    stopThresh = False
    

    # initially check if we have no density data on half orbit, and therefore use ECH wave criteria
    if np.isnan(input_dict["density"]).all():
        # find the hfr spectra indicies correaponding to perigee and apogee
        d_spectra_perigee = gl.find_closest(input_dict["hfr_epoch"],perigee)[1]
        d_spectra_apogee = gl.find_closest(input_dict["hfr_epoch"],apogee)[1]


        # integrate in ech freq range and take find last crossing
        pp_index = np.nan
        integral = []
        for i in range(d_spectra_perigee,d_spectra_apogee):
            
            integral.append(find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][i]))

        # find last time we cross threshold by going through list of integrals!
        for i in range(1, len(integral)):
            if integral[i] > 1e-4 and integral[i-1] <= 1e-4:
                # found! starts from apogee location, so add to index
                pp_index = d_spectra_perigee+i
                break

        # set location and details of crossing. 
        if np.isnan(pp_index)==False:
            # crossing type = ECH
            pp_type = 2
            findLstar_ech= fp.FindLANLFeatures(input_dict["hfr_epoch"][pp_index], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
            Lstar_stamp_ech  = findLstar_ech.get_Lstar
            MLT_stamp_ech = findLstar_ech.get_MLT
            findOmniFeats_ech = fp.FindOMNIFeatures(input_dict["hfr_epoch"][pp_index],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
            AE_stamp_ech = findOmniFeats_ech.get_AE
            AE_star_ech = findOmniFeats_ech.get_AEstar
            findDensityTime,findDensityIndex = gl.find_closest(input_dict["fpe_epoch"],input_dict["hfr_epoch"][pp_index])

            
            pp_L = Lstar_stamp_ech
            pp_MLT = MLT_stamp_ech
            pp_AE = AE_stamp_ech
            pp_AEstar = AE_star_ech
            pp_time = input_dict["hfr_epoch"][pp_index]
            pp_density =  input_dict["density"][findDensityIndex]    

            # set the ECH wave criteria flag to found (True) so that we know not to keep checking with the threshold method
            stopECH=True
        
    # now cycle through density data to find gradient
    for i in range(d_perigee,d_apogee):

        # find the time difference between adjacent density values in order to check for gaps 
        # if there are gaps, we will have to call the ECH wave checker

        dt_density = input_dict["fpe_epoch"][i+1] - input_dict["fpe_epoch"][i]

        # find the Lstar, MLT
        findLstar= fp.FindLANLFeatures(input_dict["fpe_epoch"][i], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
        Lstar_stamp = findLstar.get_Lstar
        Lstar_index = findLstar.index()
        MLT_stamp = findLstar.get_MLT 

        # find the AE and AE* values
        findOmniFeats = fp.FindOMNIFeatures(input_dict["fpe_epoch"][i],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
        AE_stamp = findOmniFeats.get_AE
        AE_star = findOmniFeats.get_AEstar

        # find Lstar+0.5 for density gradient check
        Lstar_plus = Lstar+0.5
        findPlus = fp.FindDensity(input_dict["Lstar"], Lstar_plus, Lstar_index, input_dict["lanl_epoch"], input_dict["fpe_epoch"])
        d_plusIndex = findPlus.index()
        Lstar_plusIndex = findPlus.find_closest_L()[1]
        

        # If problem: density point or next density point is an np.nan value, or there is a data gap (larger than 10 minutes) - check ECH or set to no method worked
        # Otherwise, YAY find the gradient!!
        if (np.isnan(input_dict["density"][i])==False) and (np.isnan(d_plusIndex)==False) and (dt_density<timedelta(seconds=600)) and (input_dict["fpe_epoch"][i]<apogee):
            
            # calaculate gradient
            gradient = input_dict["density"][d_plusIndex]/input_dict["density"][i]
            print(gradient,"is the gradient")

            if (gradient<1/5):
                
                # set flag to gradient for that orbit
                pp_type = 1
                print("Before,after:",Lstar_stamp,input_dict["Lstar"][Lstar_plusIndex])
                findOmniFeats = fp.FindOMNIFeatures(input_dict["fpe_epoch"][d_plusIndex],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
                AE_stamp = findOmniFeats.get_AE
                AE_star = findOmniFeats.get_AEstar

                # set the plasmapause crossing details as the L+0.5 point
                pp_L = input_dict["Lstar"][Lstar_plusIndex]
                pp_time = input_dict["fpe_epoch"][d_plusIndex]
                pp_AE = AE_stamp
                pp_AEstar = AE_star
                pp_MLT = MLT_stamp
                pp_density = input_dict["density"][d_plusIndex]
                # stop looking as first gradient found!
                break
        
    if (pp_type!=1) and (pp_type!=2):
        # if we did not find the plasmapause via the density gradient, and no intitial density gap, chek ECH waves again

                # find the hfr spectra indicies correaponding to perigee and apogee
                d_spectra_perigee = gl.find_closest(input_dict["hfr_epoch"],perigee)[1]
                d_spectra_apogee = gl.find_closest(input_dict["hfr_epoch"],apogee)[1]

                pp_index = np.nan
                integral = []
                for k in range(d_spectra_perigee,d_spectra_apogee):
                    
                    integral.append(find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]))

                # now fnd last time we cross threshold by going through list!
                for k in range(1, len(integral)):
                    if integral[k] > 1e-4 and integral[k-1] <= 1e-4:
                        # starts from apogee index!!
                        pp_index = d_spectra_perigee+k
                        # set crossing type
                        pp_type = 2
                        
                        # find corresponding L, MLT, AE, AE*
                        findLstar=fp.FindLANLFeatures(input_dict["hfr_epoch"][pp_index], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
                        Lstar_stamp = findLstar.get_Lstar
                        MLT_stamp = findLstar.get_MLT   
                        findOmniFeats = fp.FindOMNIFeatures(input_dict["hfr_epoch"][pp_index],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
                        AE_stamp = findOmniFeats.get_AE
                        AE_star = findOmniFeats.get_AEstar

                        # set the plasmapause crossings details
                        pp_L = Lstar_stamp
                        pp_AE = AE_stamp
                        pp_AEstar = AE_star
                        pp_MLT = MLT_stamp
                        pp_time = input_dict["hfr_epoch"][pp_index]

                        # set the ECH wave criteria flag to found (True) so that we know not to keep checking with the threshold method
                        stopECH=True
                        break
                         
                    
                # if we go over the range, or haven't found a crossing, use threshold method and then leave
                if (pp_type!=2)and(stopThresh==False) :

                    # set the stopECH to True so that we do not check again on the next density value pass if already checked
                    print("Now stop checking ECH")
                    stopECH = True
                    pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEstar = density_thresh(input_dict["fpe_epoch"], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"],input_dict["density"],d_perigee,d_apogee,input_dict["epoch_omni"],input_dict["AE"],input_dict["epoch_omni_low"],input_dict["Dst"],input_dict["Kp"])
                    stopThresh = True

    # if we still haven't found it, rsort back to density threshold
    if pp_type == 0:
        # resort back to density threshold one final time, if none of the other options worked
        pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEstar = density_thresh(input_dict["fpe_epoch"], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"],input_dict["density"],d_perigee,d_apogee,input_dict["epoch_omni"],input_dict["AE"],input_dict["epoch_omni_low"],input_dict["Dst"],input_dict["Kp"])
            
    print("The half-orbit with peri ",gee_index," - apo ",gee_index+1, "outwards is done")

    return pp_type,pp_L,pp_time,pp_MLT,pp_AE, pp_AEstar


def inwards_journ(input_dict, gee_index):

    def density_thresh(fpe_epoch, lanl_epoch, MLT, MLAT_N, MLAT_S, Lstar,density,d_apogee,d_perigee,epoch_omni,AE,epoch_omni_low,Dst,Kp):
    # resort back to density threshold
   
        # first, make sure that the spacecraft is 'leaving' the plasmapause - otherwise we are always out!
        findLstar= fp.FindLANLFeatures(fpe_epoch[d_apogee], lanl_epoch, MLT, MLAT_N, MLAT_S, Lstar)
        Lstar_stamp = findLstar.get_Lstar
        MLT_stamp = findLstar.get_MLT   
        findOmniFeats = fp.FindOMNIFeatures(fpe_epoch[d_apogee],epoch_omni, epoch_omni_low, AE, Kp, Dst)
        AE_stamp = findOmniFeats.get_AE
        AE_star = findOmniFeats.get_AEstar


        # find the density threshold - largest between 50 and empirical L relationship for threshold
        thresh = np.max([50,10*(6.6/Lstar_stamp)**4])

        # set initial values of plasmapause crossing details to np.nan
        pp_type = None
        pp_L = np.nan
        pp_time = np.nan
        pp_MLT = np.nan
        pp_AE = np.nan
        pp_AEstar = np.nan

        # if at the beginning of the half orbit we are not already in the plasmasphere
        if density[d_apogee]<thresh:
            
            # loop through half orbit densities
            for k in range(d_apogee+1,d_perigee):

                # find correspinding Lstar,MLT value
                findLstar= fp.FindLANLFeatures(fpe_epoch[k], lanl_epoch, MLT, MLAT_N, MLAT_S, Lstar)
                Lstar_stamp = findLstar.get_Lstar
                MLT_stamp = findLstar.get_MLT

                # find AE and AE* values
                findOmniFeats = fp.FindOMNIFeatures(fpe_epoch[k],epoch_omni, epoch_omni_low, AE,Kp,Dst)
                AE_stamp = findOmniFeats.get_AE
                AE_star = findOmniFeats.get_AEstar
        
                # find the density threshold - largest between 50 and empirical L relationship for threshold
                thresh = np.max([50,10*(6.6/Lstar_stamp)**4])
        
                # if current density above threshold, have found plasmapause and therefore update crossing details and leave loop
                if density[k]>thresh:
        
                    pp_type = 3
                    pp_L = Lstar_stamp
                    pp_time = fpe_epoch[k]
                    pp_AE = AE_stamp
                    pp_AEstar = AE_star
                    pp_MLT = MLT_stamp
                    break
                
                # otherwise, not found :(
                else:

                    pp_type = 4.
                    pp_L = np.nan
                    pp_time = np.nan
                    pp_AE = np.nan
                    pp_AEstar = np.nan
                    pp_MLT = np.nan

        else:
            # always in plasmasphere on this orbit!!
            pp_type = 4
            pp_L = np.nan
            pp_time = np.nan
            pp_AE = np.nan
            pp_AEstar = np.nan
            pp_MLT = np.nan

        return pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEstar


    # first, look for density gradient

    # find the index correpsonding to the apogee and perigee in the density data
    apogee = input_dict["all_times"][gee_index]
    perigee = input_dict["all_times"][gee_index+1]
    
    d_perigee = gl.find_closest(input_dict["fpe_epoch"],perigee)[1]
    d_apogee = gl.find_closest(input_dict["fpe_epoch"],apogee)[1]

     # set the plasmapause crossing initial values to np.nan
    pp_L = np.nan
    pp_type = 0
    pp_time = np.nan
    pp_AE = np.nan
    pp_AEstar = np.nan
    pp_MLT = np.nan
    
    # initially check if we have no density data on half orbit, and therefore use ECH wave criteria
    if np.isnan(input_dict["density"]).all() or (input_dict["fpe_epoch"][d_perigee]-perigee<timedelta(seconds=60)):

        # find the hfr spectra indicies correaponding to perigee and apogee
        d_spectra_perigee = gl.find_closest(input_dict["hfr_epoch"],perigee)[1]
        d_spectra_apogee = gl.find_closest(input_dict["hfr_epoch"],apogee)[1]

        # integrate in ech freq range and take find last crossing
        pp_index = np.nan
        integral = []
        for i in range(d_spectra_apogee,d_spectra_perigee):
            
            integral.append(find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][i]))

        # now fnd last time we cross threshold by going through list!
        for i in range(1, len(integral)):
            if integral[i] < 1e-4 and integral[i-1] >= 1e-4:
                # starts from apogee index!!
                pp_index = d_spectra_apogee+i

        # set location etc 
        if np.isnan(pp_index)==False:
            # set crossing type
                pp_type = 2
                
                # find corresponding L, MLT, AE, AE*
                findLstar=fp.FindLANLFeatures(input_dict["hfr_epoch"][pp_index], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
                Lstar_stamp = findLstar.get_Lstar
                MLT_stamp = findLstar.get_MLT   
                findOmniFeats = fp.FindOMNIFeatures(input_dict["hfr_epoch"][pp_index],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
                AE_stamp = findOmniFeats.get_AE
                AE_star = findOmniFeats.get_AEstar

                # set the plasmapause crossings details
                pp_L = Lstar_stamp
                pp_AE = AE_stamp
                pp_AEstar = AE_star
                pp_MLT = MLT_stamp
                pp_time = input_dict["hfr_epoch"][pp_index]
               
    print("The apogee of this inward orbit is", apogee, 'and the perigee is', perigee)

   #  now check for gradient
    # cycle through the density data fro this half orbit
    for i in range(d_apogee,d_perigee):
        
        
        # find the time difference between adjacent density values in order to check for gaps 
        # if there are gaps, we will have to call the ECH wave checker
        dt_density = input_dict["fpe_epoch"][i+1] - input_dict["fpe_epoch"][i]
        
        # find the Lstar, and Lstar+0.5
        findLstar= fp.FindLANLFeatures(input_dict["fpe_epoch"][i], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
        Lstar_stamp = findLstar.get_Lstar
        Lstar_index = findLstar.index()
        MLT_stamp = findLstar.get_MLT
        
        # find AE and AE*
        findOmniFeats = fp.FindOMNIFeatures(input_dict["fpe_epoch"][i],input_dict["epoch_omni"],input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
        AE_stamp = findOmniFeats.get_AE
        AE_star = findOmniFeats.get_AEstar

        # find the Lstar+0.5 for the gradient calculation
        Lstar_plus = Lstar+0.5
        findPlus = fp.FindDensity(input_dict["Lstar"], Lstar_plus, Lstar_index, input_dict["lanl_epoch"], input_dict["fpe_epoch"])
        d_plusIndex = findPlus.index()
        Lstar_plusIndex = findPlus.find_closest_L()[1]
        

        # Otherwise, YAY find the gradient!!
        # density at L and density at L+0.5 must exist, there must be < 10 minutes between timesteps between the initial density and the following density, and the L+0.5 must not go beyond the half orbit
        gradient = 0
        if (np.isnan(input_dict["density"][i])==False) and (np.isnan(input_dict["density"][d_plusIndex])==False) and (input_dict["lanl_epoch"][Lstar_plusIndex]<perigee) and (dt_density<timedelta(seconds=600)) and (input_dict["fpe_epoch"][i]<perigee):

            gradient = input_dict["density"][d_plusIndex]/input_dict["density"][i]

            # making sure that we defo end up inside the plasmasphere (no funny buinsess!!)
            if (gradient>5)and(input_dict["density"][d_plusIndex]>50):
                
                # find 

                pp_AEstar = AE_star
                pp_type = 1
                pp_L = Lstar_stamp
                pp_MLT =MLT_stamp
                pp_time = input_dict["fpe_epoch"][i]
                pp_AE = AE_stamp
                break

    # cycle through the density data fro this half orbit again checking ech waves and density threshold
    if pp_type!=1: 
        # find the hfr spectra indicies correaponding to perigee and apogee
        d_spectra_apogee = gl.find_closest(input_dict["hfr_epoch"],apogee)[1]
        d_spectra_perigee = gl.find_closest(input_dict["hfr_epoch"],perigee)[1]
        
        # cycle through the HFR spectra for this orbit
        # integrate in ech freq range and take find last crossing
        pp_index = np.nan
        integral = []
        for k in range(d_spectra_apogee,d_spectra_perigee):
                
            integral.append(find_ech(input_dict["fce"],input_dict["fce_epoch"],input_dict["Etotal"],input_dict["hfr_E_reduced"],input_dict["survey_epoch"],input_dict["hfr_epoch"],input_dict["survey_freq"],input_dict["hfr_frequency"],input_dict["hfr_epoch"][k]))

        # now fnd last time we cross threshold by going through list!
        for k in range(1, len(integral)):
            if integral[k] < 1e-4 and integral[k-1] >= 1e-4:
                # starts from apogee index!!
                pp_index = d_spectra_apogee+k

            # set location etc 
            if np.isnan(pp_index)==False:

                pp_type = 2
                # find L, MLT, AE and AE*
                findLstar= fp.FindLANLFeatures(input_dict["hfr_epoch"][pp_index], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"])
                Lstar_stamp = findLstar.get_Lstar
                MLT_stamp = findLstar.get_MLT
                findOmniFeats = fp.FindOMNIFeatures(input_dict["hfr_epoch"][pp_index],input_dict["epoch_omni"], input_dict["epoch_omni_low"], input_dict["AE"], input_dict["Kp"], input_dict["Dst"])
                AE_stamp = findOmniFeats.get_AE
                AE_star = findOmniFeats.get_AEstar

                # set the plasmapause crossing details
                pp_L = Lstar_stamp
                pp_MLT = MLT_stamp
                pp_AE = AE_stamp
                pp_AEstar = AE_star
                pp_time = input_dict["hfr_epoch"][pp_index]

    # finally, if we still haven't found the crossing, then use density threshold              
    if pp_type == 0:
        # resort back to density threshold one final time, if none of the other options worked
        pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEstar= density_thresh(input_dict["fpe_epoch"], input_dict["lanl_epoch"], input_dict["MLT"], input_dict["MLAT_N"], input_dict["MLAT_S"], input_dict["Lstar"],input_dict["density"],d_apogee,d_perigee,input_dict["epoch_omni"],input_dict["AE"],input_dict["epoch_omni_low"],input_dict["Dst"],input_dict["Kp"])
            
    print("apo ",gee_index," - peri ",gee_index+1, "inwards done")

    return pp_type,pp_L,pp_time,pp_MLT,pp_AE,pp_AEstar


def combine_lastonorbit_with_firstonnextorbit(day, last_orb_type):
    """ This function combines the final half orbit from the last day, with the first half orbit on the next day 
    if the last ornit was an 'into the plasmasphere' orbit"""
    if last_orb_type=='a':
        orbit_type = 'inward'

    else:
        orbit_type = 'outward'

    return orbit_type,



""" Defining all global variables 
"""
''' running for a month... '''
date_string = sys.argv[1]
start_date = datetime.strptime(date_string, "%Y%m%d")
no_days = calendar.monthrange(start_date.year, start_date.month)[1]

# run through days
for single_day in (start_date + timedelta(n) for n in range(no_days)):
    # lists for saving all stats
    types_pps = []                      # crossing type
    L_pps = []                          # L* at crossing
    times_pps = []                      # time at crossing
    in_or_out=[]                        # inwards or outwards orbit (from the plasmasphere)
    AE_pps = []                         # AE at crossing time
    AEstar_pps =[]                      # AE* at crossing time
    MLT_pps = []                        # MLT at crossing time

    """ create object for accessing all files for given day """
    day_files = gl.DataFiles(single_day)
    # String version of dates for filename  
    date_string,year,month,day =gl.get_date_string(single_day)

    """ Create the OMNI dataset """
    omni_dataset = gl.omni_dataset(single_day-timedelta(days=1),single_day+timedelta(days=2))
    AE, epoch_omni = omni_dataset.omni_stats
    Kp, Dst, epoch_omni_low = omni_dataset.omni_stats_low_res

    """ Getting survey file and accessing survey frequencies, epoch and magnitude """
    survey_file = pycdf.CDF(day_files.survey_data)
    survey_data = gl.AccessSurveyAttrs(survey_file)

    """ get survey properties """
    survey_freq = survey_data.frequency
    survey_epoch = survey_data.epoch_convert()
    survey_bins = survey_data.bin_edges
    Btotal = survey_data.Bmagnitude
    Etotal = survey_data.Emagnitude

    """ do the background reduction on the electric field """
    Ewfr_reduced = fa.BackReduction(Etotal, 'EW', False)


    """ getting magnetometer data """
    # get the density, LANL, and magentometer data
    # Getting the magnetometer data
    mag_file = pycdf.CDF(day_files.magnetic_data[0])
    mag_data = gl.AccessL3Attrs(mag_file)

    # flag for when we have absolutley no density data
    zero_densdat = False
    if day_files.l4_data()[1] == False:
        print("OH NO - no density data for:",single_day)
        zero_densdat = True
        #continue
    fce, fce_05, fce_005,f_lhr = mag_data.f_ce
    fce_epoch = mag_data.epoch_convert()

    """ Getting survey HFR file and accessing frequencies, epoch and E magnitude """
    hfr_data =  pycdf.CDF(day_files.hfr_data)
    hfr_epoch = gl.AccessHFRAttrs(hfr_data).epoch_convert()
    hfr_frequency = gl.AccessHFRAttrs(hfr_data).frequency
    hfr_E = gl.AccessHFRAttrs(hfr_data).Emagnitude
    hfr_E_reduced = fa.HFRback(hfr_E)

    """ Getting the density data """
    if zero_densdat==False:
        density_file = pycdf.CDF(day_files.l4_data()[0])
        density_data = gl.AccessL4Attrs(density_file)
        """ get all densities - including in/out plasmapause flag """

        fpe_uncleaned = density_data.f_pe
        fpe_epoch = density_data.epoch_convert()
        density_uncleaned= density_data.density

    else:
        """ if there's no density data file at all, create array filled with -1 """
        # Generate times
        time_list = []
        # Given time start
        time_start = time(0, 0, 0)
        # Given time end
        time_end = time(23, 59, 54)
        # Combine date and time
        date_with_time_start = datetime.combine(single_day, time_start)
        # Combine date and time
        date_with_time_end = datetime.combine(single_day, time_end)
        # Define interval (6 seconds)
        interval = timedelta(seconds=6)

        current_time = date_with_time_start
        while current_time <= date_with_time_end:
            time_list.append(current_time)  # Convert to string if needed
            current_time += interval

        
        density_uncleaned = np.zeros(14400)
        density_uncleaned[:] = -1
        fpe_uncleaned = np.zeros(14400)
        fpe_uncleaned[:] = -1
        fpe_epoch = time_list

    """ Getting the LANL data """
    lanl_file = h5py.File(day_files.lanl_data)
    lanl_data = gl.AccessLANLAttrs(lanl_file)
    # Getting LANL attributes
    apogee, perigee = lanl_data.apogee_perigee
    Lstar = lanl_data.L_star
    MLT = lanl_data.MLT
    MLAT_N, MLAT_S = lanl_data.MLAT_N_S
    lanl_epoch = lanl_data.epoch

    # Find what the first and last half orbits begin with 
    all_times = np.concatenate((apogee,perigee))

    # Creating corresponding labels
    labels = np.array(['a'] * len(apogee) + ['p'] * len(perigee))
    
    # Sort dates and labels together
    sorted_pairs = sorted(zip(all_times, labels))

    # Unzip the sorted pairs
    all_times, labels = zip(*sorted_pairs)
   
    # Finding the label for the maximum value
    max_index,min_index= np.argmax(all_times),np.argmin(all_times)  # Index of the max value
    max_label,min_label= labels[max_index],labels[min_index]  # Corresponding label
    print(max_index)
    
    if len(density_uncleaned)<2:
        # Generate times
        time_list = []
        # Given time start
        time_start = time(0, 0, 0)
        # Given time end
        time_end = time(23, 59, 54)
        # Combine date and time
        date_with_time_start = datetime.combine(single_day, time_start)
        # Combine date and time
        date_with_time_end = datetime.combine(single_day, time_end)
        # Define interval (e.g., every 30 minutes)
        interval = timedelta(seconds=6)

        current_time = date_with_time_start
        while current_time <= date_with_time_end:
            time_list.append(current_time)  # Convert to string if needed
            current_time += interval

        density_uncleaned = np.zeros(14400)
        density_uncleaned[:] = -1
        fpe_uncleaned = np.zeros(14400)
        fpe_uncleaned[:] = -1
        fpe_epoch = time_list

    # clean fpe array
    fpe = np.zeros((len(density_uncleaned)))
    fpe[:] = np.nan
    density = np.zeros((len(density_uncleaned)))
    density[:] = np.nan


    # Set all of the rogue density/fpe values as np.nan
    for i in range(len(density)):

        if fpe_uncleaned[i] <0.:
            density[i] = np.nan
            fpe[i] = np.nan

        else:
            density[i] = density_uncleaned[i]
            fpe[i] = fpe_uncleaned[i]

    input_dict0 = {'apogee':apogee,'perigee':perigee,
                  'fpe_epoch':fpe_epoch,'density':density,'fpe': fpe,
                  'MLT':MLT, 'MLAT_N':MLAT_N, 'MLAT_S': MLAT_S, 'Lstar': Lstar, 'lanl_epoch':lanl_epoch,
                  'all_times': all_times, 
                  'hfr_E_reduced':hfr_E_reduced,'hfr_epoch':hfr_epoch,'hfr_frequency':hfr_frequency,'Etotal':Ewfr_reduced,
                  'epoch_omni': epoch_omni, 'AE':AE, 'epoch_omni_low':epoch_omni_low, 'Dst':Dst, 'Kp':Kp,
                  'fce': fce, 'fce_epoch':fce_epoch,'survey_freq':survey_freq,'survey_epoch':survey_epoch}
    
    
    # first, do the last <half orbit / first <half orbit of current day
    if single_day>start_date:

        combined_dict = {}

        for key in input_dict0:
            print(f"{key}: {type(input_dict0[key])}")
            if isinstance(input_dict0[key], (np.ndarray)):
                combined_dict[f"{key}"] = np.concatenate((lastday_input_dict[key],input_dict0[key]))

            if isinstance(input_dict0[key],(list)):
                combined_dict[f"{key}"] = lastday_input_dict[key]+input_dict0[key]

        combined_dict["all_times"] = [lastday_input_dict["all_times"][-1],all_times[0]]
        

        if last_half_label == 'a':
            
        
            print("On inward journey for last < half orbit")
            
            types_pps.append(pp_type)
            pp_type, pp_L, pp_time, pp_MLT, pp_AE,pp_AEstar = inwards_journ(combined_dict, 0)
            L_pps.append(pp_L)
            times_pps.append(pp_time)
            in_or_out.append('inward')
            AE_pps.append(pp_AE)
            MLT_pps.append(pp_MLT)
            AEstar_pps.append(pp_AEstar)


        else:

            print("On outward journey for last < half orbit")
            pp_type, pp_L, pp_time, pp_MLT, pp_AE, pp_AEstar = outwards_journ(combined_dict, 0)
            print(pp_type, pp_L, pp_time)
            types_pps.append(pp_type)
            L_pps.append(pp_L)
            times_pps.append(pp_time)
            in_or_out.append('outward')
            AE_pps.append(pp_AE)
            AEstar_pps.append(pp_AEstar)
            MLT_pps.append(pp_MLT)

    # then do the remiander of half orbits

    for gee_index in range(len(all_times)-1):
        print(gee_index,labels[gee_index])
        if labels[gee_index] == 'a':
            print("On inward journey")
            pp_type, pp_L, pp_time, pp_MLT, pp_AE, pp_AEstar = inwards_journ(input_dict0, gee_index)
            print(pp_type, pp_L, pp_time)
            types_pps.append(pp_type)
            L_pps.append(pp_L)
            times_pps.append(pp_time)
            in_or_out.append('inward')
            AE_pps.append(pp_AE)
            AEstar_pps.append(pp_AEstar)
            MLT_pps.append(pp_MLT)

        else: 
            print("On outward journey")
            pp_type, pp_L, pp_time,pp_MLT,pp_AE,pp_AEstar= outwards_journ(input_dict0, gee_index)
            print(pp_type, pp_L, pp_time)
            types_pps.append(pp_type)
            L_pps.append(pp_L)
            times_pps.append(pp_time)
            in_or_out.append('outward')
            AE_pps.append(pp_AE)
            AEstar_pps.append(pp_AEstar)
            MLT_pps.append(pp_MLT)

    
    lastday_input_dict = input_dict0.copy()
    last_half_label = max_label

    print(single_day,"done")
   
    # save all plasmatrough regions to file
    with open(f'plasmatrough_regions/pp_{start_date.year,start_date.month}', "a", newline="") as f:
        writer = csv.writer(f)
        for row in zip(types_pps, times_pps, L_pps, in_or_out, MLT_pps, AE_pps, AEstar_pps):
            writer.writerow(row)
            print("appended half orb")




