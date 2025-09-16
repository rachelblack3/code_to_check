# other numerical tools
import os
os.environ["CDF_LIB"] = "/data/hpcdata/users/rablack75/cdf37_1-dist/lib"
import sys

# numerical essentials 
import numpy as np

# for creating cdfs
import pandas as pd


# for cdf reading
from spacepy import toolbox
#spacepy.toolbox.update(leapsecs=True)
from spacepy import pycdf
import h5py

# other numerical tools
import os
from datetime import datetime,timedelta

# my own misc. functions
import global_use_rocky as gl
import funcs_stats as stats
import find_properties as fp
import funcs_analysis as fa

""" This code IDs chorus waves in the survey data. 

Criteria:
In every WFR frequency band, check emeission is:

1. within frequency range f_lhr - 0.9f_ce
2. above noise threshold
3. ellipticity > 0.7, planarity >0.6 and polarisation>0.5
4. outside of plasmasphere (use plasmatrough regions in 'plasmatrough_regions/{A}or{B}'

Outputs:
Each row of dataframe represents timestamp in WFR, with:
'ChorusID==True' - positive ID for each WFR frequency band (shape:65)
'SurveyIntegral' - power between f_lhr - 0.9f_ce           (shape:1)
'Timestamp'      - wfr timestamp                           (shape:1)
'MLT'            - MLT                                     (shape:1)
'MLAT'           - MLAT                                    (shape:1)
'Lstar'          - L*                                      (shape:1)
'AE'             - AE                                      (shape:1)
'Kp'             - Kp                                      (shape:1)
'Dst'            - Dst                                     (shape:1)
'Plasmapause'    - In or out                               (shape:1)

"""

date_string = sys.argv[1]
single_day = datetime.strptime(date_string, "%Y%m%d")
print(single_day)

""" Defining all global variables 
"""

day_files = gl.DataFiles(single_day)

# String version of dates for filename  
date_string,year,month,day =gl.get_date_string(single_day)

"""
make lists for saving stats
"""
df_stat = pd.DataFrame(columns=['Timestamp',
                           'MLT','MLAT','Lstar',
                           'AE','Kp','Dst',
                           'Plasmapause', 'ChorusID', 'Survey_integral'])

""" 
Acess OMNI data from folder
"""
# Create the OMNI dataset
omni_dataset = gl.omni_dataset(single_day,single_day+timedelta(days=1))
AE, epoch_omni = omni_dataset.omni_stats
Kp, Dst, epoch_omni_low = omni_dataset.omni_stats_low_res


date_params = {"year": year,
                "month": month,
                "day": day,
                "single_day": single_day}

# Getting the LANL data
lanl_file = h5py.File(day_files.lanl_data)
lanl_data = gl.AccessLANLAttrs(lanl_file)
# Getting LANL attributes
Lstar = lanl_data.L_star
MLT = lanl_data.MLT
MLAT_N, MLAT_S = lanl_data.MLAT_N_S
lanl_epoch = lanl_data.epoch


mag_dat, mag_check = day_files.magnetic_data()
if mag_check == False:
    print("No magnetometer data on this day")

else:
    mag_file = pycdf.CDF(mag_dat)
    mag_data = gl.AccessL3Attrs(mag_file)
    # Getting gyrofrequencies and plasma frequency for the full day
    fce, fce_05, fce_005, f_lhr = mag_data.f_ce
    fce_epoch = mag_data.epoch_convert()

# Getting the density data
density_dat, density_check = day_files.l4_data()

if density_check == False:
    print("No density data on this day")

else:
    print("have density!")
    density_file = pycdf.CDF(density_dat)
    density_data = gl.AccessL4Attrs(density_file)
    fpe = density_data.f_pe
    fpe_epoch = density_data.epoch_convert()
    density = density_data.density
    if len(density) == 0:
        density_check = False
    fpe_in = np.full((len(density),), np.nan)
    fpe_out = np.full((len(density),), np.nan)


    plasmatrough_regions = gl.plasmatrough(single_day).plasmatrough_regions()
    if len(plasmatrough_regions)==0:
        for i in range(len(density)):
            fpe_in[i] = fpe[i]
            findLANLFeats = fp.FindLANLFeatures(fpe_epoch[i], lanl_epoch, MLT, MLAT_N, MLAT_S, Lstar)
            Lstar_stamp = findLANLFeats.get_Lstar

            thresh = np.max([50.,10*(6.6/Lstar_stamp)**4])
            if density[i]<thresh:
                fpe_out[i] = fpe[i]
                fpe_in[i] = np.nan
    else:
        for i in range(len(density)):
            fpe_in[i] = fpe[i]
            for region in plasmatrough_regions:
                if region[0]<fpe_epoch[i]<region[1]:
                    fpe_out[i] = fpe[i]
                    fpe_in[i] = np.nan
        

print("made it here!")
# Getting survey file and accessing survey frequencies, epoch and magnitude
survey_file = pycdf.CDF(day_files.survey_data)
survey_data = gl.AccessSurveyAttrs(survey_file)
survey_freq = survey_data.frequency
survey_epoch = survey_data.epoch_convert()
survey_bins = survey_data.bin_edges
Btotal = survey_data.Bmagnitude
Breduced = fa.BackReduction(Btotal, 'B', False)

# WNA wave properties 
wna_data = pycdf.CDF(day_files.survey_WNA)
wna_planarity = wna_data["plansvd"]
wna_ellipticity = wna_data["ellsvd"]
wna_wna = wna_data["thsvd"]
wna_poynting = wna_data["thpoy1_2_3"]
wna_epoch = gl.get_epoch(wna_data["Epoch"])
wna_freq = wna_data["WFR_frequencies"][0]
wna_polar = wna_data["polsvd"] 


print("made it here!")
# start loop through whole survey file
k = 0    
for time in survey_epoch:
    # find the fce value 

    fce_t,fce_index = gl.find_closest(fce_epoch,time)

    # finding the relative MLT/MLAT/Lstar closest to a given 

    # Save global chorus statsitcis to a dataframe

    findLANLFeats = fp.FindLANLFeatures(time, lanl_epoch, MLT, MLAT_N, MLAT_S, Lstar)
    MLT_stamp = findLANLFeats.get_MLT
    MLAT_stamp = findLANLFeats.get_MLAT
    Lstar_stamp = findLANLFeats.get_Lstar

    findOmniFeats = fp.FindOMNIFeatures(time, epoch_omni, epoch_omni_low, AE, Kp, Dst)
    AE_stamp = findOmniFeats.get_AE
    Kp_stamp = findOmniFeats.get_Kp
    Dst_stamp = findOmniFeats.get_Dst

    ellip_stamp = wna_ellipticity[k,:]
    planar_stamp = wna_planarity[k,:]
    wna_stamp = wna_wna[k,:]
    polar_stamp = wna_polar[k,:]

    # Are we inside or outside of the plasmapause?
    if density_check == False:
        PP_flag = np.nan
    else:
        PP_flag = stats.Plasmapause(time, fpe_in, fpe_out, fpe_epoch).in_or_out()


    chorus_ID = np.zeros_like(survey_freq)
    psd_stamp = Breduced[k,:]
    for j in range(65):
        
        # Frequency criteria
        if f_lhr[fce_index]<survey_freq[j]<0.9*fce[fce_index]:

            # Above noise threshold criteria
            if psd_stamp[j]!=0.:
                chorus_ID[j]=True
            else:
                chorus_ID[j] = False
                continue
            # Ellipticity and polarisation criteria
            if (ellip_stamp[j] > 0.7) and (planar_stamp[j]>0.6) and (polar_stamp[j]>0.5):
                chorus_ID[j]=True
            else:
                chorus_ID[j]=False
                continue
            # Outside plasmapause criteria
            if PP_flag == 'In':
                chorus_ID[j]=False
                continue
            elif PP_flag == 'Out':
                chorus_ID[j]=True
            else:
                chorus_ID[j]=False
                continue
            
        else:
            chorus_ID[j]=False



    # Integrate in frequency using trapezoidal rule
    frequency_integral = 0.
    for m in range(len(survey_freq)-1):

        if f_lhr[fce_index] < survey_freq[m] < 0.9*fce[fce_index]:
            frequency_integral += 0.5*(psd_stamp[m] + psd_stamp[m+1])*(survey_freq[m+1]-survey_freq[m])

    print("The frequency integral for the survey is:",frequency_integral)
 

    # Row to append
    new_row = {'Timestamp': time,
                'MLT': MLT_stamp,'MLAT':MLAT_stamp,'Lstar':Lstar_stamp,
                'AE': AE_stamp, 'Kp': Kp_stamp, 'Dst': Dst_stamp,
                'Plasmapause': PP_flag,
                'ChorusID': chorus_ID,
                'Survey_integral':frequency_integral}
    
    # Append the row
    df_stat= pd.concat([df_stat, pd.DataFrame([new_row])], ignore_index=True)

    k = k+1 
print(str(single_day), "is done")

# save datafrake to a CSV file
df_stat.to_csv(f'/data/emfisis_burst/wip/rablack75/rablack75/CountSurvey/CSVschorus_surveypowerA_202509/{year}/{month}/{date_string}.csv')  
