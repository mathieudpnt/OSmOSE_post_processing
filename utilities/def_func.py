import struct
from typing import Tuple
import pytz
import pandas as pd
import re
import datetime as dt
import random
import numpy as np
from tqdm import tqdm
import os
import glob
import wave
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import statistics as stat
from tkinter import filedialog
from tkinter import Tk
import gzip
import math
import easygui


# def read_header(file:str) -> Tuple[int, int, int, int, int]:
#     #reads header of a wav file to get info such as duration, samplerate etc...
    
#     with open(file, "rb") as fh:
#        _, size, _ = struct.unpack('<4sI4s', fh.read(12))

#        chunk_header = fh.read(8)
#        subchunkid, _ = struct.unpack('<4sI', chunk_header)

#        if (subchunkid == b'fmt '):
#            _, channels, samplerate, _, _, sampwidth = struct.unpack('HHIIHH', fh.read(16))

#        sampwidth = (sampwidth + 7) // 8
#        framesize = channels * sampwidth
#        frames = size // framesize 
#        return sampwidth, frames, samplerate, channels, frames/samplerate

def read_header(file:str) -> Tuple[int, int, int, int]:
    #reads header of a wav file to get info such as duration, samplerate etc...

    with open(file, "rb") as fh:
        _, size, _ = struct.unpack('<4sI4s', fh.read(12))
        chunk_header = fh.read(8)
        subchunkid, _ = struct.unpack('<4sI', chunk_header)

        if (subchunkid == b'fmt '):
            _, channels, samplerate, _, _, sampwidth = struct.unpack('HHIIHH', fh.read(16))

        chunkOffset = fh.tell()
        found_data = False
        while (chunkOffset < size and not found_data):
            fh.seek(chunkOffset)
            subchunk2id, subchunk2size = struct.unpack('<4sI', fh.read(8))
            if (subchunk2id == b'data'):
                found_data = True
                
            chunkOffset = chunkOffset + subchunk2size + 8

        if not found_data:
            print("No data chunk found while reading the header. Will fallback on the header size.")
            subchunk2size = (size - 36)

        sampwidth = (sampwidth + 7) // 8
        framesize = channels * sampwidth
        frames = subchunk2size // framesize

        if (size - 36) != subchunk2size:
            print(f"Warning : the size indicated in the header is not the same as the actual file size. This might mean that the file is truncated or otherwise corrupt.\
                \nSupposed size: {size} bytes \nActual size: {subchunk2size} bytes.")

        return sampwidth, frames, samplerate, channels, frames/samplerate


def get_wav_info(folder):
    durations=[]
    wav_files = glob.glob(os.path.join(folder, "**/*.wav"), recursive=True)
    for file in tqdm(wav_files, 'Getting wav durations...', position=0, leave=True):
        try:
            with wave.open(file, 'r') as wav_files:
                frames = wav_files.getnframes()
                rate = wav_files.getframerate()
                durations.append(frames / float(rate))
        except Exception as e:
            print(f'An error occured while reading the file {file} : {e}') 
    return durations


def extract_datetime(var, tz, formats=None):
    #extract datetime from filename such as Apocado / Cetiroise or custom ones
    
    if formats is None:
        formats = [r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', r'\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}', r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}'] #add more format if necessary
    match = None
    for f in formats:
        match = re.search(f, var)
        if match:
            break
    if match:
        dt_string = match.group()
        if f == r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}':
            dt_format = '%Y-%m-%dT%H-%M-%S'
        elif f == r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}':
            dt_format = '%Y-%m-%d_%H-%M-%S'
        elif f == r'\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}':
            dt_format = '%y%m%d%H%M%S'
        elif f == r'\d{2}\d{2}\d{2}_\d{2}\d{2}\d{2}':
            dt_format = '%y%m%d_%H%M%S'
        date_obj = dt.datetime.strptime(dt_string, dt_format)
        
        if type(tz) is pytz._FixedOffset: date_obj = tz.localize(date_obj)
        else: date_obj = pytz.timezone(tz).localize(date_obj)  
        
        return date_obj
    else:
        return print("No datetime found")
    
def sorting_annot_boxes(file, tz=None, date_begin=None, date_end=None, annotator=None, label=None) -> Tuple[int, int, list, list, pd.DataFrame]:
    # From an Aplose results csv, returns a DataFrame without the Aplose box annotations (weak annotations)

    df = pd.read_csv(file)
    max_freq = int(max(df['end_frequency']))
    max_time = int(max(df['end_time']))
    df = df.loc[(df['start_time'] == 0) & (df['end_time'] == max_time) & (df['end_frequency'] == max_freq)] #deletion of boxes
    df = df.sort_values('start_datetime') #sorting value according to datetime_start
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z')
    df['end_datetime'] = pd.to_datetime(df['end_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z')
    if tz is not None:
        df['start_datetime'] = [x.tz_convert(tz) for x in df['start_datetime']] #converting to desired tz
        df['end_datetime'] = [x.tz_convert(tz) for x in df['end_datetime']] #converting to desired tz
    if date_begin is not None and date_end is not None: 
        df = df[(df['start_datetime']>=date_begin) & (df['start_datetime']<=date_end)] #select data within [date_begin;date_end]
    df = df.reset_index(drop=True) #reset the indexes of row after sorting the df
    if annotator is not None:
        df = df.loc[(df['annotator'] == annotator)]
    if annotator is not None:
        df = df.loc[(df['annotation'] == label)]
    annotators = list(df['annotator'].drop_duplicates())
    labels = list(df['annotation'].drop_duplicates())
    
    # print(len(df), 'annotations')
    return (max_time, max_freq, annotators, labels, df)

# def t_rounder(t, resolution):
#     # Rounds to nearest 10-minute interval

#     minute = t.minute
#     minute = (minute + 5) // 10 * 10
#     if minute >= 60:
#             minute = 0
#             hour = t.hour + 1
#             if hour >= 24:
#                 hour = 0
#                 t += dt.timedelta(days=1)
#             t = t.replace(hour=hour, minute=minute, second=0, microsecond=0)
#     else: t = t.replace(minute=minute, second=0, microsecond=0)
#     return t

def t_rounder(t, res):
    """Rounds a Timestamp according to the user specified resolution : 10s / 1min / 10 min / 1h / 24h

    Parameter:
        t: Timestamp to round
        res: integer corresponding to the new resolution in seconds

    Returns:
        Rounded Timestamp"""
    
    if res == 600: #10min
        minute = t.minute
        minute = math.floor(minute/10)*10
        t = t.replace(minute=minute, second=0, microsecond=0)
    elif res == 10: #10s
        seconde = t.second
        seconde = math.floor(seconde/10)*10
        t = t.replace(second=seconde, microsecond=0)
    elif res == 60: #1min
        t = t.replace(second=0, microsecond=0)
    elif res == 3600: #1h
        t = t.replace(minute=0, second=0, microsecond=0)
    elif res == 86400: #24h
        t = t.replace(hour=0, minute=0, second=0, microsecond=0)
    return t

def from_str2ts(date):
    #from APLOSE date string to a timestamp
    
    return dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()

def from_str2dt(date):
    #from APLOSE date string to a datetime
    
    return dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f%z')

def oneday_per_month(time_vector_ts, time_vector_str, vec)-> Tuple[list, list, list, list]:
    #select a random day for each months in input datetimes list and returns all the datetimes of those randomly selected days
    
    time_vector = [dt.datetime.fromtimestamp(time_vector_ts[i]) for i in range(len(time_vector_ts))]
    
    datetimes_by_month = {}
    for i in range(len(time_vector)):
        key = (time_vector[i].year, time_vector[i].month)
        if key not in datetimes_by_month:
            datetimes_by_month[key] = []
        datetimes_by_month[key].append((time_vector[i], vec[i], time_vector_str[i]))

    # randomly select one day for each month
    selected_datetimes = []
    selected_vec = []
    selected_str = []
    for dt_by_month in datetimes_by_month.values():
        if len(dt_by_month) > 0:
            month_days = list(set(list_dt[0].day for list_dt in dt_by_month)) # get all unique days in the month
            selected_day = random.choice(month_days) # randomly select one day
            for i, PG, time_str in dt_by_month:
                if i.day == selected_day:
                    selected_datetimes.append(i)
                    selected_vec.append(PG)
                    selected_str.append(time_str)

    unique_dates = sorted(list(set(i.strftime('%d/%m/%Y') for i in selected_datetimes)), key=lambda x: dt.datetime.strptime(x, '%d/%m/%Y'))
    return [selected_datetimes[i].timestamp() for i in range(len(selected_datetimes))], [selected_vec[i] for i in range(len(selected_vec))], [selected_str[i] for i in range(len(selected_str))], unique_dates



def n_random_hour(time_vector_ts, time_vector_str, vec, n_hour, TZ, time_step)-> Tuple[list, list, list, list]:
    # randomly select n non-overlapping hours from the time vector
    if type(TZ) is not pytz._FixedOffset: TZ=pytz.timezone(TZ)
    
    if not isinstance(n_hour, int):
        print('n_hour is not an integer')
        return
    
    selected_time_vector_ts, selected_dates = [],[]
    while len(selected_dates) < n_hour:
        # choose a random datetime from the time vector
        rand_idx = random.randrange(len(time_vector_ts))
        rand_datetime = time_vector_ts[rand_idx]
        
        selected_dates.append(rand_datetime)
    
        # select all datetimes that fall within the hour following this datetime
        # possible_datetimes = [time for time in time_vector_ts if rand_datetime < time < rand_datetime + 3600]
        # possible_datetimes = [(idx, dt) for idx, dt in enumerate(time_vector_ts) if rand_datetime <= dt <= rand_datetime+3600]
        possible_datetimes = time_vector_ts[rand_idx:rand_idx+round(3600/time_step)+1]
        
        # check if any of the selected datetimes overlap with the previously selected datetimes
        overlap = False
        for i in selected_time_vector_ts:
            if any(i <= time < i+3600 for time in possible_datetimes):
                overlap = True
                break
    
        if overlap : continue
    
        # add the selected datetimes to the list
        selected_time_vector_ts.extend(possible_datetimes)
    
        # sort the selected datetimes in chronological order
        selected_time_vector_ts.sort()
        selected_dates.sort()
        
    # extract the corresponding vectors and time strings
    selected_vec = [vec[time_vector_ts.index(i)] for i in tqdm(selected_time_vector_ts, position=0, leave=True)]
    selected_time_vector_str = [time_vector_str[time_vector_ts.index(i)] for i in tqdm(selected_time_vector_ts, position=0, leave=True)]
    selected_dates = [dt.datetime.fromtimestamp(i, TZ).strftime('%d/%m/%Y %H:%M:%S') for i in selected_dates]

    return selected_time_vector_ts, selected_time_vector_str, selected_vec, selected_dates

def pick_datetimes(time_vector_ts, time_vector_str, vec, selected_dates, selected_durations, TZ)-> Tuple[list, list, list, list]:
    # user-selected datetimes from the time vector
    
    #
    selected_df_out = pd.DataFrame({'datetimes': selected_dates, 'durations': selected_durations})
    
    # format the datetimes and durations from strings to datetimes/timedeltas
    selected_dates = [dt.datetime.strptime(i, '%d/%m/%Y %H:%M:%S').timestamp() for i in selected_dates]
    timedeltas =[]
    for i in selected_durations:
        if i.endswith('h'):
            timedeltas.append(dt.timedelta(hours=int(i[:-1])).total_seconds())
        elif i.endswith('m'):
            timedeltas.append(dt.timedelta(minutes=int(i[:-1])).total_seconds())
        elif i.endswith('s'):
            timedeltas.append(dt.timedelta(seconds=int(i[:-1])).total_seconds())
        elif i.endswith('d'):
            timedeltas.append(dt.timedelta(days=int(i[:-1])).total_seconds())
        else:
            print('incorrect duration format')
            return
    selected_durations = timedeltas
    
    
    selected_time_vector_ts = []


    for i in range(len(selected_dates)):
        # select all datetimes that fall within the durations following this datetime
        possible_datetimes = [time for time in time_vector_ts if selected_dates[i] <= time <= selected_dates[i] + selected_durations[i] ]

        # add the selected datetimes to the list
        selected_time_vector_ts.extend(possible_datetimes)

    # sort the selected datetimes in chronological order
    selected_time_vector_ts.sort()
        
    # extract the corresponding vectors and time strings
    selected_vec = [vec[time_vector_ts.index(i)] for i in tqdm(selected_time_vector_ts, position=0, leave=True)]
    selected_time_vector_str = [time_vector_str[time_vector_ts.index(i)] for i in tqdm(selected_time_vector_ts, position=0, leave=True)]
    selected_dates = [dt.datetime.fromtimestamp(i, pytz.timezone(TZ)).strftime('%d/%m/%Y %H:%M:%S') for i in selected_dates]

    return selected_time_vector_ts, selected_time_vector_str, selected_vec, selected_df_out

def export2Raven(tuple_info, time_vec, time_str, bin_height, selection_vec=None):
    # Export a given vector to Raven formatted table
    #the functions requires several input arguments:
        #-time_vec : the vector to export
        #-time_str : the corresponding names of each timebin to be exported
        #-TZ : the time zone info
        #-files : the list of the paths of the correponding wav files
        #-dur : list of the wav files durations
        #-bin_height : the maximum frequency of the exported timebins
        #-selection_vec : if it is set to None, all the timebins are exported, else the selection_vec is used to selec the wanted timebins to export, for instance it corresponds to all the positives timebins, containing detections
    
    #TODO: gérer les dernieres timebin de chaque wav car elles peuvent être < à la timebin duration et donc elles débordent sur le wav suivant pour l'instant
    #TODO : créer un fichier readme avec des infos de la config
    
    file_list = tuple_info[0]
    file_datetimes = tuple_info[1]
    dur = tuple_info[2]
    if selection_vec is None: selection_vec= np.ones(len(time_vec)-1)
    
    offsets = [(file_datetimes[i]+dt.timedelta(seconds=dur[i])).timestamp() - (file_datetimes[i+1]).timestamp() for i in range(len(file_datetimes)-1)]
    offsets_cumsum=(list(np.cumsum([offsets[i] for i in range(len(offsets))])))
    offsets_cumsum.insert(0, 0)

    test_name = list(np.array([file.split('.wav')[0] for file in file_list])) # extract file names without extension
    idx_wav_Raven = [test_name.index(time_str[i].split('_+')[0]) for i in tqdm(range(len(time_vec)-1), position=0, leave=True, desc = '1/3')]
    start_datetime = [int(time_vec[i] - file_datetimes[0].timestamp()) + offsets_cumsum[idx_wav_Raven[i]] for i in tqdm(range(len(time_vec)-1), position=0, leave=True, desc = '2/3') if selection_vec[i] == 1]
    end_datetime =   [int(time_vec[i] - file_datetimes[0].timestamp())+(time_vec[i+1]-time_vec[i]) + offsets_cumsum[idx_wav_Raven[i]]  for i in tqdm(range(len(time_vec)-1), position=0, leave=True, desc = '3/3') if selection_vec[i] == 1]
    
    df_PG2Raven = pd.DataFrame()
    df_PG2Raven['Selection'] = np.arange(1,len(start_datetime)+1)
    df_PG2Raven['View'], df_PG2Raven['Channel'] = [1]*len(start_datetime), [1]*len(start_datetime)
    df_PG2Raven['Begin Time (s)'] = start_datetime     
    df_PG2Raven['End Time (s)'] = end_datetime     
    df_PG2Raven['Low Freq (Hz)'] = [0]*len(start_datetime)
    df_PG2Raven['High Freq (Hz)'] = [bin_height]*len(start_datetime)
    
    # durations = df_PG2Raven['End Time (s)']-df_PG2Raven['Begin Time (s)']
    # rows = [idx for idx, dur in tqdm(enumerate(durations), position=0, leave=True, desc = '4/3') if dur > stat.median(durations)]
    # df_PG2Raven = df_PG2Raven.drop(rows)
    # df_PG2Raven['Selection'] = np.arange(1,len(df_PG2Raven)+1)

    # Convert relevant columns to NumPy arrays
    begin_times = np.array(df_PG2Raven['Begin Time (s)'])
    end_times = np.array(df_PG2Raven['End Time (s)'])
    
    # Calculate durations using vectorized operations
    durations = end_times - begin_times
    
    # Filter rows based on duration using boolean indexing
    rows_to_keep = durations <= 10*np.median(durations)
    df_PG2Raven = df_PG2Raven[rows_to_keep]
    
    # Update the 'Selection' column with consecutive numbers
    df_PG2Raven['Selection'] = np.arange(1, len(df_PG2Raven) + 1)

    return df_PG2Raven

def get_season(ts):
    # "day of year" ranges for the northern hemisphere
    winter1 = range(1,80)
    spring = range(80, 172)
    summer = range(172, 264)
    autumn = range(264, 355)
    winter2 = range(355,367)
    # winter = everything else
    if ts.dayofyear in spring: season = 'spring'+ ' ' + str(ts.year)
    elif ts.dayofyear in summer: season = 'summer'+ ' ' + str(ts.year)
    elif ts.dayofyear in autumn: season = 'autumn'+ ' ' + str(ts.year)
    elif ts.dayofyear in winter1: season = 'winter'+ ' ' + str(ts.year-1)
    elif ts.dayofyear in winter2: season = 'winter'+ ' ' + str(ts.year)
    return season

def histo_detect(detections, lim, res_min, time_bin,  plot=False, hours_interval=4, label=None, annotator=None):
    #Get the detection histogram values and plot it if desired
    
    delta, start_vec, end_vec = dt.timedelta(seconds=60*res_min),  lim[0], lim[1]
    bins = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]
    n_annot_max = (res_min*60)/time_bin #max nb of annoted time_bin max per res_min slice
    tz_data = lim[0].tz

    fig,ax = plt.subplots(figsize=(20,9))
    y, x, _ = ax.hist(detections, bins); #histo
    if plot is False: plt.close()
    else:
    
        bars = range(0,110,10) #from 0 to 100 step 10
        y_pos = np.linspace(0,n_annot_max, num=len(bars))
        ax.set_yticks(y_pos, bars);
        ax.tick_params(axis='x', rotation= 60);
        ax.tick_params(labelsize=20)
        ax.set_ylabel("Positive detection rate [%] ("+str(res_min)+"min)", fontsize = 20)
        ax.tick_params(axis='y')
        if label is not None and annotator is not None:
            plt.suptitle('annotateur : ' + annotator +'\n'+ 'label : ' + label, fontsize = 24, y=0.98);
         
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=hours_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %Hh', tz=tz_data))
        ax.set_xlim(lim[0]-dt.timedelta(minutes=20), lim[-1]+dt.timedelta(minutes=20))
        ax.set_ylim(0, n_annot_max)
        plt.grid(color='k', linestyle='-', linewidth=0.2, axis='both')
    
    return y, x

def load_glider_nav():
    #Load the navigation data from glider output files
    #TODO : lire les data directement à partir des fichiers gz
    root = Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select master folder")
    
    all_rows = [] # Initialize an empty list to store the contents of all CSV files
    yo = []  # List to store the file numbers
    file = []
    
    first_file = True
    file_number = 1  # Initialize the file number
    
    for filename in os.listdir(directory):
        if filename.endswith('.gz'):
            file_path = os.path.join(directory, filename)

            with gzip.open(file_path, 'rt') as gz_file:
                delimiter = ';'  # Specify the desired delimiter
                gz_reader = pd.read_csv(gz_file, delimiter=delimiter)
                # If it's the first file, append the header row
                if first_file:
                    all_rows.append(gz_reader.columns.tolist())
                    first_file = False
                # Add the rows from the current CSV file to the all_rows list
                all_rows.extend(gz_reader.values.tolist())
                # Add yo number to the yo list
                yo.extend([filename.split('.')[-2]] * len(gz_reader))
                file.extend([filename] * len(gz_reader))
                file_number += 1  # Increment the file number for the next file
    
    # Create a DataFrame from the combined data
    df = pd.DataFrame(all_rows)
    df.columns = df.iloc[0]  # set 1st row as headers
    df = df.iloc[1:, 0:-1]  # delete last column and 1st row

    # Add the yo number to the DataFrame
    df['yo'] = [int(x) for x in yo]
    
    df['file'] = file
    df = df.sort_values(by=['Timestamp'])
    df = df.drop(df[(df['Lat'] == 0) & (df['Lon'] == 0)].index).reset_index(drop=True)
    df['Lat DD'] = [int(x) + (((x - int(x))/60)*100) for x in df['Lat']/100]
    df['Lon DD'] = [int(x) + (((x - int(x))/60)*100) for x in df['Lon']/100]
    df['Datetime'] = [dt.datetime.strptime(x, '%d/%m/%Y %H:%M:%S') for x in df['Timestamp']]
    df['Depth'] = -df['Depth']

    return df


def reshape_timebin(detections_file):
    """Changes the timebin (time resolution) of a detection file
    ex :    -from a raw PAMGuard detection file to a detection file with 10s timebin
            -from an 10s detection file to a 1min / 1h / 24h detection file

    Parameter:
        detection_file: Path to the detection file

    Returns:
        another dataframe with the new timebin and writes it to a csv"""
    
    t_detections = sorting_annot_boxes(detections_file)
    df_detections = t_detections[-1]
    timebin_orig = t_detections[0]
    fmax = t_detections[1]
    annotators = t_detections[2]
    labels = t_detections[3]
    tz_data = df_detections['start_datetime'][0].tz

    while True:
        timebin_new = easygui.buttonbox('Select a new time resolution for the detection file', 'Select new timebin', ['10s','1min', '10min', '1h', '24h'])
        if timebin_new == '10s':
            f= timebin_new
            timebin_new=10
        elif timebin_new == '1min':
            f= timebin_new
            timebin_new=60
        elif timebin_new == '10min':
            f= timebin_new
            timebin_new=600
        elif timebin_new == '1h':
            f= timebin_new
            timebin_new=3600
        elif timebin_new == '24h':
            f= timebin_new
            timebin_new=86400
        
        if timebin_new > timebin_orig: break
        else: easygui.msgbox('New time resolution is equal or smaller than the original one', 'Warning', 'Ok')
                    
    df_new = pd.DataFrame()
    for annotator in annotators:
        for label in labels:
            
            df_detect_prov = sorting_annot_boxes(file=detections_file, annotator = annotator, label = label)[-1]

            t = t_rounder(df_detect_prov['start_datetime'].iloc[0], timebin_new)
            t2 = t_rounder(df_detect_prov['start_datetime'].iloc[-1], timebin_new) + dt.timedelta(seconds=timebin_new)
            time_vector = [ts.timestamp() for ts in pd.date_range(start=t, end=t2, freq=f)]
            # time_vector_str = [ts.timestamp() for ts in pd.date_range(start=t, end=t2, freq=f)]
            
            times_detect_beg = [detect.timestamp() for detect in df_detect_prov['start_datetime']]
            times_detect_end = [detect.timestamp() for detect in df_detect_prov['end_datetime']]
                
            detect_vec, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
            for i in range(len(times_detect_beg)):
                for j in range(k, len(time_vector)-1):
                    if int(times_detect_beg[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)) or int(times_detect_end[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)):
                            ranks.append(j)
                            k=j
                            break
                    else: 
                        continue 
            ranks = sorted(list(set(ranks)))
            detect_vec[ranks] = 1
            detect_vec = list(detect_vec)
               
            
            start_datetime_str, end_datetime_str, filename = [],[],[]
            for i in range(len(time_vector)):
                if detect_vec[i] == 1:
                    start_datetime = pd.Timestamp(time_vector[i], unit='s', tz=tz_data)
                    start_datetime_str.append(start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[:-8]+ start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-5:-2] +':' + start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-2:])
                    end_datetime = pd.Timestamp(time_vector[i]+timebin_new, unit='s', tz=tz_data)
                    end_datetime_str.append(end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[:-8]+ end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-5:-2] +':' + end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-2:])
                    filename.append(str(pd.Timestamp(time_vector[i], unit='s', tz=tz_data)))
            
            
            df_new_prov = pd.DataFrame()
            dataset_str = list(set(df_detect_prov['dataset']))
            
            df_new_prov['dataset'] = dataset_str*len(start_datetime_str)
            df_new_prov['filename'] = filename
            df_new_prov['start_time'] = [0]*len(start_datetime_str)
            df_new_prov['end_time'] = [timebin_new]*len(start_datetime_str)
            df_new_prov['start_frequency'] = [0]*len(start_datetime_str)
            df_new_prov['end_frequency'] = [fmax]*len(start_datetime_str)
            
            df_new_prov['annotation'] = list(set(df_detect_prov['annotation']))*len(start_datetime_str)
            df_new_prov['annotator'] = list(set(df_detect_prov['annotator']))*len(start_datetime_str)
              
            df_new_prov['start_datetime'], df_new_prov['end_datetime'] = start_datetime_str, end_datetime_str
    
            df_new = pd.concat([df_new, df_new_prov])
            
        df_new = df_new.sort_values(by=['start_datetime'])
            
    return df_new

















