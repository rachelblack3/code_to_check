import numpy as np
from datetime import datetime,date,timedelta,time
import pandas as pd
import glob
import os
from spacepy import pycdf

# constants

global_constants = {"Electron q": 1.6*10**(-19),
                    "Electron m":9.1*10**(-31),
                    "Proton m": 1.67*10**(-27),
                    "Convert to nT": (10**(-9)),
                    "Pi": 3.14159,
                    "f_s": 1.0/35000.0,
                    "Slider": int(1024/2),
                    "Kletzing box": 16384,
                    "Sliding box": 1024,
                    "N total": 208896, 
                    "N Kletzing": 12}  


Duration = global_constants["f_s"] * global_constants["N total"]

global_constants["Duration"] = Duration


# functions
def get_date_string(date):
        ''' Method that rpovides date strings
        Outputs:
    
        date_string: string object of date
        year: string object of date year
        month: string object of date month
        day: string object of date day '''

        date_string= str(date.strftime("%Y%m%d"))

        if (date.day <10):
            day = "0"+str(date.day)
        else:
            day = str(date.day)


        if (date.month<10):
            month = "0"+str(date.month)
        else:
            month = str(date.month)
        

        year = str(date.year)
        
        return date_string,year,month,day

def what_dates(default):

    ''' Function that asks command prompt the required date range
        Outputs:

        start_date: Datetime object of start date
        end_date: Datetime object of end date
        no_days: Integer number of days
    '''

    if (default == True):
        start_date = date(year=2012, month=9, day=1)
        end_date = date(year=2019, month=10, day=14)
    else:
        # Ask for start date and make datetime object
        year,month,day =  [eval(i) for i in input("Please enter start year, month, day: ").split(",")]
        start_date = date(year=year, month=month, day=day)                                                       

        # Ask for end date and make datetime object
        year_end,month_end,day_end =  [eval(i) for i in input("Please enter end year, month, day: ").split(",")]
        end_date = date(year=year_end, month=month_end, day=day_end)

    # Find integer number of days in given date range
    no_days = end_date - start_date
    no_days = int(no_days.days)                     

    return (start_date,end_date,no_days)

def h5_time_conversion(bytes_time) -> datetime:
    """
    Converts ISO 8601 datetime, which contains leap seconds, into Python
    datetime object

    :param bytes_time: time in bytes to convert
    :return: datetime in Python datetime object
    """
    date_str, time_str = bytes_time.decode("UTF-8").replace("Z", "").split("T")
    year, month, day = date_str.split("-")
    date = datetime(int(year), int(month), int(day))
    no_ms_time, dot, ms = time_str.partition(".")
    hours, minutes, seconds = no_ms_time.split(":")
    date = date + timedelta(
        hours=int(hours), minutes=int(minutes), seconds=int(seconds)
    )
    if dot:
        date = date + timedelta(milliseconds=int(ms))
    return date

def get_epoch(epoch):
    # get epoch in correct format for comparisons etc.
        epoch_new = []

        # First - saving the epoch elements to a list as a string - acording to a particular desired format
        for i in range(len(epoch)):
            epoch_new.append(datetime.strftime(epoch[i],'%Y-%m-%d %H-%M-%S'))

        # Chaning back to a datetime object of same format    
        for i in range(len(epoch_new)):
            epoch_new[i] = datetime.strptime(epoch_new[i],'%Y-%m-%d %H-%M-%S')

        return epoch_new

def find_closest(epoch_known, wanted_time):

        n = len(epoch_known)

        nup = n-1
        nlow=0
        mid=int((nup+nlow)/2)

        while (nup - nlow)>1:
            mid= int((nup+nlow)/2)
            if (wanted_time > epoch_known[mid]):
                nup = nup
                nlow=mid
            else:
                nup = mid
                nlow=nlow
        
        wanted_index = nup
        corresponding_time = epoch_known[wanted_index]

        return(corresponding_time,wanted_index)


class omni_dataset:
    ''' Class for creating omni_dataset over full duration specified '''

    def __init__(self,start,end):
        self.start_date = start.date()
        self.end_date = end.date()

        self.omni_cache = None

    @property
    def omni_stats(self):
            if self.omni_cache is None:
                print("filling cache")
                omni_file = '/data/hpcdata/users/rablack75/CountSurvey/data/OMNI/solar_wind/omni_high_res_combined_2000_to_2020.txt'
                file_format = '/data/hpcdata/users/rablack75/CountSurvey/data/OMNI/solar_wind/high_res_format'
                f_format = open(f'{file_format}.txt',"r")
                line_formats=f_format.readlines()

                for line in line_formats:
                    print(line)
                f_format.close()
                # From this, AE is line 11, so index line position 10
                # Initialise empty lists to store AE and omni_epoch

                AE=[]
                epoch_omni=[]
                Kp=[]
                print('the start and end are',self.start_date,self.end_date)
            
                start_d = str(self.start_date.strftime("%Y-%m-%d"))
                
                no_days = self.end_date - self.start_date
                no_days = int(no_days.days)


                f=open(omni_file,"r")
                lines=f.readlines()
                
                i=0
                for line in lines:
                    
                    line = [x for x in line.rstrip("\n").split(" ") if x != ""]
                    Date = date(
                        int(line[0].replace(" ", "")), 1, 1
                    ) + timedelta(
                        days=int(line[1]) - 1
                    )
                    
                    if Date == self.start_date:
                        print(start_d, 'string exists in file')
                        print('Line Number:', i)
                        first=i
                        start_d = self.start_date
                        break
                    i=i+1
                        
                for line in lines[first:(first+(no_days*24*60))]:
                
                    line = [x for x in line.rstrip("\n").split(" ") if x != ""]
                    Date = datetime(int(line[0].replace(" ", "")), 1, 1
                    ) + timedelta(
                        days=int(line[1]) - 1,
                        hours=int(line[2]),
                        minutes=int(line[3]),
                    )
                    epoch_omni.append(Date)
                    AE.append(float(line[10]))
                print('OMNI dataset created')
                f.close()

                #self.omni_cache = AE, epoch_omni

                return AE,epoch_omni
    
            else:
                print("data from cache")
                return self.omni_cache
            
    @property
    def omni_stats_low_res(self):
            """ This is for Kp and Dst, which are given on an hourly rate """

            omni_file = '/data/hpcdata/users/rablack75/CountSurvey/data/OMNI/solar_wind/omni_low_res_combined_2000_to_2020.txt'
            file_format = '/data/hpcdata/users/rablack75/CountSurvey/data/OMNI/solar_wind/low_res_format'
            f_format = open(f'{file_format}.txt',"r")
            line_formats=f_format.readlines()

            for line in line_formats:
                print(line)
            f_format.close()
            # From this, AE is line 11, so index line position 10
            # Initialise empty lists to store AE and omni_epoch

            Dst=[]
            epoch_omni=[]
            Kp=[]

            print('the start and end are',self.start_date,self.end_date)
            start_d = str(self.start_date.strftime("%Y-%m-%d"))
        
            no_days = self.end_date - self.start_date
            no_days = int(no_days.days)


            f=open(omni_file,"r")
            lines=f.readlines()

            i=0
            for line in lines:
                
                line = [x for x in line.rstrip("\n").split(" ") if x != ""]
                Date = date(
                    int(line[0].replace(" ", "")), 1, 1
                ) + timedelta(
                    days=int(line[1]) - 1
                )
                if Date == self.start_date:
                    print(start_d, 'string exists in file')
                    print('Line Number:', i)
                    first=i
                    start_d = self.start_date
                    break
                i=i+1
                    
            for line in lines[first:(first+(no_days*24*60))]:
            
                line = [x for x in line.rstrip("\n").split(" ") if x != ""]
                Date = datetime(int(line[0].replace(" ", "")), 1, 1
                ) + timedelta(
                    days=int(line[1]) - 1,
                    hours=int(line[2]),
                )
                epoch_omni.append(Date)
                Kp.append(float(line[3]))
                Dst.append(float(line[4]))

            print('OMNI dataset created')
            f.close()

            self.omni_cache = Kp, Dst, epoch_omni

            return Kp,Dst,epoch_omni


                
class DataFiles:
    ''' class for accessing all of the data files that I need on a given day.  
    
            PARAMETERS:
            date: DateTime object 
           '''

    def __init__(self,date):
        self.date = date
        

    # finding survey data filepath for given day
    @property
    def survey_data(self):
    
        # String version of dates for filename  
        date_string,year,month,day =self.get_date_string()

        # root folder
        survey_folder = f'/data/spacecast/wave_database_v2/RBSP-A/L2/{year}/{month}/{day}' # if not using rocky: '/data/hpcflash/users/rablack75/data/Survey'
        # 'stem' name for burst CDFs
        survey_file= 'rbsp-a_WFR-spectral-matrix-diagonal_emfisis-L2_'

        survey_path = os.path.join(survey_folder, survey_file + "*_v*.cdf")
        print(survey_path)
        # find the latest version
        survey_path = glob.glob(survey_path)[-1]
        print(survey_path)
        return survey_path
    
    # getting the high frequnecy survey data (i.e. for plasma frequencies)
    @property
    def hfr_data(self):

        # String version of dates for filename  
        date_string,year,month,day =self.get_date_string()

        # root folder
        hfr_folder = '/data/spacecast/satellite/RBSP/emfisis/data/RBSP-A/L2'

        # 'stem' name for burst cdfs
        hfr_file ='rbsp-a_HFR-spectra_emfisis-L2_'

        hfr_path = os.path.join(hfr_folder, year, month, day, hfr_file + date_string + "_v*.cdf")

        # find the latest version
        hfr_path = glob.glob(hfr_path)[-1]

        return hfr_path
    

   
    # finding the magnetic data filepath for given dayfdensity_checl
    @property
    def magnetic_data(self):

        # String version of dates for filename  
        date_string,year,month,day =self.get_date_string()

        # root folder
        l3_folder = f'/data/spacecast/wave_database_v2/RBSP-A/L3/{year}' # if not rocky: '/data/hpcflash/users/rablack75/data/magnetometer'
       
        # 'stem' name for burst CDFs
        l3_file= 'rbsp-a_magnetometer_1sec-geo_emfisis-L3_'

        l3_path = os.path.join(l3_folder,month, day, l3_file + date_string + "_v*.cdf")

        # find the latest version
        l3_path = glob.glob(l3_path)

        if not l3_path:
            exists = False
        else:
            exists = True
            l3_path = l3_path[-1]
        
        return l3_path, exists

    # finding the LANL data filepath for given day 
    @property
    def lanl_data(self):
        # String version of dates for filename  
        date_string,year,month,day =self.get_date_string()

        # root folder
        lanl_folder = f'/data/spacecast/satellite/RBSP/emfisis/data/RBSP-A/LANL/MagEphem/{year}' # if not usin rocky: '/data/hpcflash/users/rablack75/data/LANL'#'/data/spacecast/satellite/RBSP/emfisis/data/RBSP-A/LANL/MagEphem'
        # 'stem' name for burst CDFs
        lanl_file= 'rbspa_def_MagEphem_TS04D_'

        lanl_path = os.path.join(lanl_folder, lanl_file + date_string + "_v*.h5")

        # find the latest version
        lanl_path = glob.glob(lanl_path)[-1]
        
        return lanl_path
    


    # Get L4 data filepath (aka density file) for given day
    def l4_data(self):

        # String version of dates for filename  
        date_string,year,month,day =self.get_date_string()

        # root folder
        l4_folder =  f'/data/spacecast/satellite/RBSP/emfisis/data/RBSP-A/L4/{year}'# when not using rocky and have to put everything in flash '/data/hpcflash/users/rablack75/data/L4'
        # 'stem' name for burst CDFs
        l4_file= 'rbsp-a_density_emfisis-L4_'

        l4_path = os.path.join(l4_folder,month,day,l4_file + date_string + "_v*.cdf")

        # find the latest version
        l4_path = glob.glob(l4_path)
        print(l4_path)
        # check if there is actually any data files for this day:

        if not l4_path:
            exists = False
        else:
            exists = True
            l4_path = l4_path[-1]
        
        return l4_path, exists

    @property
    def survey_WNA(self):

        # String version of dates for filename  
        date_string,year,month,day =self.get_date_string()

        # root folder
        wna_folder =f'/data/spacecast/satellite/RBSP/emfisis/data/RBSP-A/wna-survey/{year}'
        # 'stem' name for burst CDFs
        wna_file= 'rbsp-a_wna-survey_emfisis-l4_'
        wna_path = os.path.join(wna_folder,wna_file + date_string + "_v*.cdf")
        print(wna_path)
        # find the latest version
        wna_path = glob.glob(wna_path)[-1]

        return wna_path

    def get_date_string(self):
        ''' Method that rpovides date strings
        Outputs:
    
        date_string: string object of date
        year: string object of date year
        month: string object of date month
        day: string object of date day '''

        date_string= str(self.date.strftime("%Y%m%d"))

        if (self.date.day <10):
            day = "0"+str(self.date.day)
        else:
            day = str(self.date.day)


        if (self.date.month<10):
            month = "0"+str(self.date.month)
        else:
            month = str(self.date.month)
        

        year = str(self.date.year)
        
        return date_string,year,month,day

class AccessHFRAttrs:
    ''' Class for finding and working on HFR survey data attributes
     
      PARAMETERS:
      survey_cdf: A CDF containing all hfr data '''

    def __init__(self, hfr_cdf):
        self.hfr_cdf = hfr_cdf

    @property
    def frequency(self):

        freq_hfr = self.hfr_cdf['HFR_frequencies'][0]

        return freq_hfr
    
    @property
    def epoch(self):
        # Epoch in DateTime format of: 
        epoch = self.hfr_cdf['Epoch']

        return epoch
    
    @property
    def Emagnitude(self):

        Etotal = self.hfr_cdf['HFR_Spectra']

        return Etotal
    
    def epoch_convert(self):
    # get epoch in correct format for comparisons etc.
        epoch_edit = []
        epoch = self.hfr_cdf['Epoch']

        # First - saving the epoch elements to a list as a string - acording to a particular desired format
        for i in range(len(epoch)):
            epoch_edit.append(datetime.strftime(epoch[i],'%Y-%m-%d %H-%M-%S'))

        # Chaning back to a datetime object of same format    
        for i in range(len(epoch_edit)):
            epoch_edit[i] = datetime.strptime(epoch_edit[i],'%Y-%m-%d %H-%M-%S')

        return epoch_edit

    
class AccessLANLAttrs:

    ''' Class for finding and working on Survey data attributes
     
      PARAMETERS:
      survey_cdf: A CDF containing all survey data '''

    def __init__(self, lanl_data):
        self.data = lanl_data

    @property
    def L_star(self):

        # get L*
        lstar = np.array(self.data['Lstar'][:, 0])

        
        return lstar
    
    @property
    def MLT(self):

        # get MLT
        MLT = np.array(self.data['EDMAG_MLT'])

        return MLT
    
    @property
    def MLAT_N_S(self):

        # get all MLAT
        MLAT = np.array(self.data['EDMAG_MLAT'])

        # want to plot north and south on same axis - so split up
        south_mask = MLAT<5.
        north_mask = MLAT>0.
        MLAT_north = np.where(north_mask,MLAT, np.nan)
        MLAT_south = np.where(south_mask,MLAT, np.nan)

        return MLAT_north,MLAT_south
    
    @property
    def epoch(self):

        lanl_times=np.array([h5_time_conversion(x) for x in self.data['IsoTime']])

        return lanl_times
    
    @property
    def day_limits(self):

        L_star = self.L_star
        
        left=np.array([h5_time_conversion(x) for x in self.data['IsoTime']])[L_star >= 0][0]
        right=np.array([h5_time_conversion(x) for x in self.data['IsoTime']])[L_star >= 0][-1]

        return left,right
    
    @property
    def apogee_perigee(self):

        apogee_times = np.array([h5_time_conversion(x) for x in self.data["ApogeeTimes"]])
        perigee_times = np.array([h5_time_conversion(x) for x in self.data["PerigeeTimes"]])

        return apogee_times,perigee_times


class AccessSurveyAttrs:
    ''' Class for finding and working on Survey data attributes
     
      PARAMETERS:
      survey_cdf: A CDF containing all survey data '''
    

    def __init__(self, survey_cdf):
        self.survey_cdf = survey_cdf

    @property
    def frequency(self):
        # Frequency in Hz
        frequency = self.survey_cdf['WFR_frequencies'][0]

        return frequency
    
    @property
    def epoch(self):
        # Epoch in DateTime format of: 
        epoch = self.survey_cdf['Epoch']

        return epoch
    
    @property
    def Bmagnitude(self):

        Bu2 = self.survey_cdf['BuBu']
        Bv2 = self.survey_cdf['BvBv']
        Bw2 = self.survey_cdf['BwBw']

        # Define empty list for total mag field array 

        Btotal = np.zeros(self.survey_cdf['BuBu'].shape)

        # Create total mag B array

        for p in range(0,np.shape(Btotal)[0]):
            Btotal[p,:] =Bu2[p,:]+ Bv2[p,:] + Bw2[p,:]

        return Btotal
    
    @property
    def Emagnitude(self):

        Eu2 = self.survey_cdf['EuEu']
        Ev2 = self.survey_cdf['EvEv']
        Ew2 = self.survey_cdf['EwEw']

        # Define empty list for total mag field array 

        Etotal = np.zeros(Eu2.shape)

        # Create total mag B array

        for p in range(0,np.shape(Etotal)[0]):
            Etotal[p,:] =Eu2[p,:]+ Ev2[p,:] + Ew2[p,:]

        return Etotal
    
    def epoch_convert(self):
    # get epoch in correct format for comparisons etc.
        epoch_edit = []
        epoch = self.survey_cdf['Epoch']

        # First - saving the epoch elements to a list as a string - acording to a particular desired format
        for i in range(len(epoch)):
            epoch_edit.append(datetime.strftime(epoch[i],'%Y-%m-%d %H-%M-%S'))

        # Chaning back to a datetime object of same format    
        for i in range(len(epoch_edit)):
            epoch_edit[i] = datetime.strptime(epoch_edit[i],'%Y-%m-%d %H-%M-%S')

        return epoch_edit
    
    @property
    def frequency_bin_widths(self):

        widths = self.survey_cdf['WFR_bandwidth'][0]
        return widths
    
    @property
    def bin_edges(self):
        """ 
        setting the bin edges 
        min_bin: lower edge of first bin 
        """
        survey_freq = self.frequency

        min_bin = survey_freq[0]-(self.frequency_bin_widths[0]/2)

        freq_bins = []
        freq_bins.append(min_bin)

        # Starting from the minimum bin, add all widths to get lower and upper bands on all semi_logarithmic bins

        for i in range(0,65):
            
            freq_bins.append(freq_bins[i]+self.frequency_bin_widths[i])
            min_bin=freq_bins[i]
        
        return freq_bins
    
class AccessL3Attrs:
    ''' Class for finding and working on L3 data attributes
     
      PARAMETERS:
      mag_cdf: A CDF containing all L3 data '''

    def __init__(self, mag_file):
        self.mag_cdf = mag_file

    @property
    def Bmagnitude(self):
        # Frequency in Hz
        Bmagnitude = self.mag_cdf['Magnitude']

        return Bmagnitude
    
    @property
    def epoch(self):
        # Epoch in DateTime format of: 
        epoch = self.mag_cdf['Epoch']

        return epoch
    
    def epoch_convert(self):
    # get epoch in correct format for comparisons etc.
        epoch_edit = []
        epoch = self.mag_cdf['Epoch']

        # First - saving the epoch elements to a list as a string - acording to a particular desired format
        for i in range(len(epoch)):
            epoch_edit.append(datetime.strftime(epoch[i],'%Y-%m-%d %H-%M-%S'))

        # Chaning back to a datetime object of same format    
        for i in range(len(epoch_edit)):
            epoch_edit[i] = datetime.strptime(epoch_edit[i],'%Y-%m-%d %H-%M-%S')

        return(epoch_edit)
    
    @property
    def f_ce(self):
        # Finding the gyrofrequencies for plotting
       
        gyro_1 = np.zeros(self.Bmagnitude.shape)
        gyro_p = np.zeros(self.Bmagnitude.shape)
        gyro_05 = np.zeros(self.Bmagnitude.shape)
        gyro_005 = np.zeros(self.Bmagnitude.shape)
        f_lhr = np.zeros(self.Bmagnitude.shape)
        # Clean magnetometer data
        mag_cleaned = self.clean_magnetometer(self.Bmagnitude)


        for i in range(0,len(gyro_1)):
            gyro_1[i] = global_constants["Electron q"]*mag_cleaned[i]*global_constants["Convert to nT"]/(2*global_constants["Pi"]*global_constants["Electron m"])
            gyro_p[i] = global_constants["Electron q"]*mag_cleaned[i]*global_constants["Convert to nT"]/(2*global_constants["Pi"]*global_constants["Proton m"])
            gyro_05[i] = 0.5*gyro_1[i]
            gyro_005[i] = 0.05*gyro_1[i]
            f_lhr[i] = np.sqrt(gyro_1[i]*gyro_p[i])
        
        return gyro_1, gyro_05, gyro_005, f_lhr

    def clean_magnetometer(self,unclean_data):

        magfill = self.mag_cdf['magFill']
        magInval = self.mag_cdf['magInvalid']
        magCal = self.mag_cdf['calState']

        # Find the indicies where we have invalid data, save to 'whereFill' 
        whereFill = []
        for i in range(len(magfill)):
            if (magfill[i] == 1 or magInval[i]==1 or magCal[i]==1):
                whereFill.append(i)
                
        # Make unclean data into list
        magnitude = np.array(unclean_data).tolist()

        # Use list of indicies to set all invlaid data points to NaNs
        for ele in sorted(whereFill, reverse = True):
            
            magnitude[ele] = np.nan
        
        return magnitude


class AccessL4Attrs:
    ''' Class for finding and working on L4 data attributes.
     
      PARAMETERS:
      density_cdf: A CDF containing all L4 data '''

    def __init__(self, density_file):
        self.density_cdf = density_file

    @property
    def density(self):
        # density in cm^(-3)
        density = self.density_cdf['density']

        return density
    
    @property 
    def f_pe(self):
        # plasma frequency in Hz
        f_pe = self.density_cdf['fpe']

        return f_pe
        
    @property
    def epoch(self):
        # Epoch in DateTime format of: 
        epoch = self.density_cdf['Epoch']

        return epoch
    
    def epoch_convert(self):
    # get epoch in correct format for comparisons etc.
        epoch_edit = []
        epoch = self.density_cdf['Epoch']

        # First - saving the epoch elements to a list as a string - acording to a particular desired format
        for i in range(len(epoch)):
            epoch_edit.append(datetime.strftime(epoch[i],'%Y-%m-%d %H-%M-%S'))

        # Chaning back to a datetime object of same format    
        for i in range(len(epoch)):
            epoch_edit[i] = datetime.strptime(epoch_edit[i],'%Y-%m-%d %H-%M-%S')

        return(epoch_edit)

class cross_dataset:
    ''' A class for performing operations across datasets '''
    def __init__(self,survey_data, l4_data, burst_time):

        self.survey_epoch = survey_data['Epoch']
        self.mag_epoch = l4_data['Epoch']
        self.Bmag = l4_data['Magnitude']
        self.burst_time = burst_time
    
    def calc_gyro(self):
        # get epoch in correct format for comparisons etc.

        epoch_mag = get_epoch(self.mag_epoch)

        mag_field=self.Bmag

        # Finding the closest index in this magnetometer list to the burst time object

        mag_t,mag_index = find_closest(epoch_mag,self.burst_time)

        # Finding the gyrofrequencies for plotting

        gyro_one= global_constants["Electron q"]*mag_field[mag_index]/(2*global_constants["Pi"]*global_constants["Electron m"]) # in T
        gyro_one = gyro_one*global_constants["Convert to nT"]                                                                   # in nT
        gyro_half=0.5*gyro_one
        gyro_low=0.05*gyro_one


        return gyro_one,gyro_half,gyro_low
    
    
    def get_epoch(epoch):
    # get epoch in correct format for comparisons etc.
        epoch_new = []

        # First - saving the epoch elements to a list as a string - acording to a particular desired format
        for i in range(len(epoch)):
            epoch_new.append(datetime.strftime(epoch[i],'%Y-%m-%d %H-%M-%S'))

        # Chaning back to a datetime object of same format    
        for i in range(len(epoch_new)):
            epoch_new[i] = datetime.strptime(epoch_new[i],'%Y-%m-%d %H-%M-%S')

        return epoch_new
    

class plasmatrough:
    def __init__(self,date):
            self.date = date.date()

    def plasmatrough_regions(self):
        # String version of dates for filename  
        df = pd.read_csv('/data/emfisis_burst/wip/rablack75/rablack75/CreateBurst/ppFINDmistake2_withTimes.csv')
        date_desired = self.date
    
        indicies=[]
        for i,day in enumerate(df["date"]):
            if day == date_desired:
                print("True!")
                indicies.append(i)

        print(len(indicies))
        plasmatrough = []
        if len(indicies)>0:
            for number,index in enumerate(indicies):
                
                # if first pass is into plasmapshere, then we were in trough
                if number==0:
                    if df.iloc[index]["InorOut"]=="inward":
                        plasmatrough.append((time(hour=0,minute=0,second=1),df.iloc[index]["time"]))
                
                # if last pass is out of plasmapshere, then we are into trough
                if number==len(indicies)-1:
                    if df.iloc[index]["InorOut"]=="outward":
                        plasmatrough.append((df.iloc[index]["time"],time(hour=23,minute=59,second=59)))

                if 0<number<len(indicies)-1:
                    if df.iloc[index]["InorOut"]=="outward":
                        plasmatrough.append((df.iloc[index]["time"],df.iloc[index+1]["time"]))
                
        return plasmatrough
                

   
