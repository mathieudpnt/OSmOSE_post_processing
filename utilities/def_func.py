import struct
from typing import Tuple, List, Dict
import pytz
import pandas as pd
import re
import datetime as dt
import random
import numpy as np
from tqdm import tqdm
import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import gzip
import math
import easygui
import glob
from typing import Union
import sys
import bisect
import csv


def get_detection_files(num_files: int) -> List[str]:
    """Opens a file dialog multiple times to get X APLOSE formatted detection files.

    Parameters :
        num_files: The number of detection files the user needs to select.

    Returns :
        List of file paths selected by the user.
    """
    root = Tk()
    root.withdraw()

    file_paths = []
    for _ in range(num_files):
        file_path = filedialog.askopenfilename(
            title=f"Select APLOSE formatted detection file ({len(file_paths) + 1}/{num_files})",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            break  # User cancelled or closed the file dialog
        file_paths.append(file_path)

    return file_paths

def sorting_detections(files: List[str], tz: pytz._FixedOffset = None, date_begin: dt.datetime = None, date_end: dt.datetime = None, annotator: str = None, label: str = None, box: bool = False, timebin_new:int = None) -> (pd.DataFrame,  pd.DataFrame):
    """ Filters an Aplose formatted detection file according to user specified filters
        
        Parameters :
            file : list of path(s) to the detection file(s), can be a str too
            tz : timezone info, to be specified if the user wants to change the TZ of the detections
            date_begin : datetime to be specified if the user wants to select detections after date_begin
            date_end : datetime to be specified if the user wants to select detections before date_end
            annotator : string to be specified if the user wants to select the detection of a particular annotator
            label : string to be specified if the user wants to select the detection of a particular label
            box : if set to True, keeps all the annotations, if False keeps only the absence/presence box (weak detection)
            timebin_new : integer to be specified if the user already know the new time resolution to set the detection file to
            
        Returns :
            max_time : spectrogram temporal length
            max_freq : sampling frequency *0.5
            annotators : list of annotators after filtering
            labels : list of labels after filtering
            result_df : dataFrame corresponding to the filters applied and containing all the detections
            info : DataFrame containing infos such as max_time/max_freq/annotators/labels corresponding to each detection file
    """
    
    if isinstance(files, str): files = [files]  # Convert the single string to a list with one element

    info, result_df = pd.DataFrame(), pd.DataFrame()
    for file in files:
        
        with open(file, 'r', newline='') as csv_file:
            try:
                temp_lines = csv_file.readline() + '\n' + csv_file.readline()
                dialect = csv.Sniffer().sniff(temp_lines, delimiters=',;')  
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ','
     
        df = pd.read_csv(file, sep=delimiter)
        
        max_freq = int(max(df['end_frequency']))

        if box is False:
            max_time = int(max(df['end_time']))
            df = df.loc[(df['start_time'] == 0) & (df['end_time'] == max_time) & (df['end_frequency'] == max_freq)]
            
            if len(df) == 0:
                if timebin_new is None :
                    df = reshape_timebin(file)
                    max_time = int(max(df['end_time']))
                else:
                    df = reshape_timebin(file, timebin_new=timebin_new)
                    max_time = timebin_new
        else:
            max_time = 0
            
            
        if timebin_new is not None and timebin_new!= max_time:
            df = reshape_timebin(file, timebin_new=timebin_new)
            max_time = timebin_new

        df = df.sort_values('start_datetime')
        df['start_datetime'] = pd.to_datetime(df['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z')
        df['end_datetime'] = pd.to_datetime(df['end_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z')
        if tz is not None:
            df['start_datetime'] = [x.tz_convert(tz) for x in df['start_datetime']]
            df['end_datetime'] = [x.tz_convert(tz) for x in df['end_datetime']]
        if date_begin is not None :
            df = df[df['start_datetime'] >= date_begin]      
        if date_end is not None :
            df = df[df['end_datetime'] <= date_end]      
        df = df.reset_index(drop=True)
        if annotator is not None:
            df = df.loc[(df['annotator'] == annotator)]
        if label is not None:
            df = df.loc[(df['annotation'] == label)]
        list_annotators = list(df['annotator'].drop_duplicates())    
        annotators = list_annotators if len(list_annotators)>1 else list_annotators[0]
        
        list_labels = list(df['annotation'].drop_duplicates())    
        labels = list_labels if len(list_labels)>1 else list_labels[0]

        result_df = pd.concat([result_df, df]).reset_index(drop=True)
        columns = ['file', 'max_time', 'max_freq', 'annotators', 'labels']
        info = pd.concat([info, pd.DataFrame([[file, int(max_time), max_freq, annotators, labels]], columns=columns) ]).reset_index(drop=True)

    return result_df, info

def reshape_timebin(detections_file: str, timebin_new:int=None) -> pd.DataFrame:
    """ Changes the timebin (time resolution) of a detection file
    ex :    -from a raw PAMGuard detection file to a detection file with 10s timebin
            -from an 10s detection file to a 1min / 1h / 24h detection file

    Parameter:
        detection_file: Path to the detection file
        timebin_new : Time resolution to base the detections on, if not provided it is asked to the user

    Returns:
        another dataframe with the new timebin
    """
    
    df_detections, t_detections = sorting_detections(files = detections_file, box=True)
    timebin_orig = t_detections.iloc[0]['max_time']
    fmax = t_detections.iloc[0]['max_freq']
    annotators = t_detections.iloc[0]['annotators']
    labels = t_detections.iloc[0]['labels']
    tz_data = df_detections['start_datetime'][0].tz
    if timebin_new is None: 
        while True:
            timebin_new = easygui.buttonbox('Select a new time resolution for the detection file', detections_file.split('/')[-1], ['10s','1min', '10min', '1h', '24h'])
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
    else: f=str(timebin_new)+'s'
    
    df_new = pd.DataFrame()
    if isinstance(annotators, str): annotators=[annotators]
    if isinstance(labels, str): labels=[labels]
    for annotator in annotators:
        for label in labels:
            
            df_detect_prov, _ = sorting_detections(files=detections_file, annotator = annotator, label = label, box=True)

            t = t_rounder(df_detect_prov['start_datetime'].iloc[0], timebin_new)
            t2 = t_rounder(df_detect_prov['start_datetime'].iloc[-1], timebin_new) + dt.timedelta(seconds=timebin_new)
            time_vector = [ts.timestamp() for ts in pd.date_range(start=t, end=t2, freq=f)]
            
            # #here test to find for each time vector value which filename corresponds
            filenames = sorted(list(set(df_detect_prov['filename'])))
            tz = df_detect_prov['start_datetime'][0].tz
            ts_filenames = [extract_datetime(filename, tz=tz).timestamp()for filename in filenames]
            
            filename_vector = []
            for ts in time_vector:
                index = bisect.bisect_left(ts_filenames, ts)
                if index == 0:
                    filename_vector.append(filenames[index])
                elif index == len(ts_filenames):
                    filename_vector.append(filenames[index - 1])
                else:
                    filename_vector.append(filenames[index - 1])
            
            
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
                    # filename.append(str(pd.Timestamp(time_vector[i], unit='s', tz=tz_data)))
                    # filename.append(df_detect_prov['filename'][i])
                    filename.append(filename_vector[i])
                    
            
            
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
    """ Reads header of a wav file to get info such as duration, samplerate etc...
    
    Parameter :
        file : path to the wav file
        
    Returns :
        sampwidth
        frames
        samplerate
        channels
        frames/samplerate
    """
    
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


# def get_wav_info(folder):
#     durations=[]
#     wav_files = glob.glob(os.path.join(folder, "**/*.wav"), recursive=True)
#     for file in tqdm(wav_files, 'Getting wav durations...', position=0, leave=True):
#         try:
#             with wave.open(file, 'r') as wav_files:
#                 frames = wav_files.getnframes()
#                 rate = wav_files.getframerate()
#                 durations.append(frames / float(rate))
#         except Exception as e:
#             print(f'An error occured while reading the file {file} : {e}') 
#     return durations


def extract_datetime(var:str, tz:pytz._FixedOffset, formats=None) -> Union[dt.datetime, str]:
    """ Extracts datetime from filename based on the date format
    
        Parameters :
            var : name of the wav file
            tz : timezone info
            formats : the date template in strftime format. For example, `2017/02/24` has the template `%Y/%m/%d`
                        For more information on strftime template, see https://strftime.org/
            
        Returns :
            date_obj : datetime corresponding to the datetime found in var
    """
    
    if formats is None:
        formats = [
                    r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}',
                    r'\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}',
                    r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}',
                    r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
                    ] #add more format if necessary
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
        elif f == r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}':
            dt_format = '%Y-%m-%d %H:%M:%S'
        date_obj = dt.datetime.strptime(dt_string, dt_format)
        
        if type(tz) is pytz._FixedOffset or tz is pytz.UTC : date_obj = tz.localize(date_obj)
        else: date_obj = pytz.timezone(tz).localize(date_obj)  
        
        return date_obj
    else:
        return print("No datetime found")

def t_rounder(t:dt.datetime, res:int):
    """ Rounds a Timestamp according to the user specified resolution : 10s / 1min / 10 min / 1h / 24h

    Parameter :
        t: Timestamp to round
        res: integer corresponding to the new resolution in seconds

    Returns :
        t: rounded Timestamp
    """
    
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

# def from_str2ts(date):
#     #from APLOSE date string to a timestamp
    
#     return dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()

# def from_str2dt(date):
#     #from APLOSE date string to a datetime
    
#     return dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f%z')

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

def export2Raven(tuple_info, time_vec, time_str, bin_height, selection_vec=None) -> pd.DataFrame:
    """ Export a given vector to Raven formatted table
            
        Parameter :

            time_vec : the vector to export
            time_str : the corresponding names of each timebin to be exported
            TZ : the time zone info
            files : the list of the paths of the correponding wav files
            dur : list of the wav files durations
            bin_height : the maximum frequency of the exported timebins
            selection_vec : if it is set to None, all the timebins are exported, else the selection_vec is used to selec the wanted timebins to export, for instance it corresponds to all the positives timebins, containing detections
    """
    
    #TODO: gérer les dernieres timebin de chaque wav car elles peuvent être < à la timebin duration et donc elles débordent sur le wav suivant pour l'instant
    
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

def get_season(ts: dt.datetime)-> str:
    """ "day of year" ranges for the northern hemisphere
    
        Parameter :
            ts : datetime
        
        Returns :
            season : string corresponding to the season and year of the datetime (ex : if datetime is 01/01/2023, returns 'winter 2022')
    """
    winter1 = range(1,80)
    spring = range(80, 172)
    summer = range(172, 264)
    autumn = range(264, 355)
    winter2 = range(355,367)

    if ts.dayofyear in spring: season = 'spring'+ ' ' + str(ts.year)
    elif ts.dayofyear in summer: season = 'summer'+ ' ' + str(ts.year)
    elif ts.dayofyear in autumn: season = 'autumn'+ ' ' + str(ts.year)
    elif ts.dayofyear in winter1: season = 'winter'+ ' ' + str(ts.year-1)
    elif ts.dayofyear in winter2: season = 'winter'+ ' ' + str(ts.year)
    
    return season


def load_glider_nav():
    """ Load the navigation data from glider output files
    
        Parameter :
            
        
        Returns :
            df : dataframe with glider navigation data
    """
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

__converter = {
    "%Y": r"[12][0-9]{3}",
    "%y": r"[0-9]{2}",
    "%m": r"(0[1-9]|1[0-2])",
    "%d": r"([0-2][0-9]|3[0-1])",
    "%H": r"([0-1][0-9]|2[0-4])",
    "%I": r"(0[1-9]|1[0-2])",
    "%p": r"(AM|PM)",
    "%M": r"[0-5][0-9]",
    "%S": r"[0-5][0-9]",
    "%f": r"[0-9]{6}",
}

def convert_template_to_re(date_template: str) -> str:
    """ Converts a template in strftime format to a matching regular expression

    Parameter :
        date_template: the template in strftime format

    Returns :
        The regular expression matching the template
    """

    res = ""
    i = 0
    while i < len(date_template):
        if date_template[i : i + 2] in __converter:
            res += __converter[date_template[i : i + 2]]
            i += 1
        else:
            res += date_template[i]
        i += 1

    return res


def get_timestamps(tz:str=None, f_type:str=None, n_dir:int=1, ext:str=None, choices:str=None, date_template:str=None, path_dir:str=None, msg:str=None)-> None:
    """  
    Read infos from APLOSE files timestamps.csv OR file_metadata.csv
    
    Parameters : 
        tz : str, optional, ex: tz='Etc/GMT-2'
            DESCRIPTION. The default is None.
        f_type : string, user specify to choose either a folder or a list of wav files
            f_type = 'dir' or f-type = 'file'
        ext : string, extension of the files
            ext='wav'
        choices : string, the user can specify the variable if the timestamps file is available or not
        date_template : string, the user can specify the variable if the date template of the wav file is known
        path_dir : string, the user can specify the path of the askfolder dialog to open

    Returns
        df_timestamps : TYPE
            DESCRIPTION.

    """
    if tz is not None:
        if type(tz) is pytz._FixedOffset or tz is not pytz.UTC: tz=pytz.timezone(tz)
        
    if choices not in ('Yes', 'No', None):
        raise ValueError('choices must be ''Yes'', ''No'', or None')

    if choices is None or (ext is None and f_type is None):
        msg_ch = 'Do you already have the timestamp.csv  ?'
        choices = ['Yes','No']
        reply = easygui.buttonbox(msg_ch, choices=choices)
    else : reply=choices
    
    if reply=='Yes':
        root = Tk()
        root.withdraw()
        timestampcsv_path = filedialog.askopenfilename(title='Select the timestamp csv file', filetypes=[("CSV files", "*.csv")]) # show an "Open" dialog box and return the path to the selected file
        root = Tk()
        root.withdraw()
        
        if os.path.basename(timestampcsv_path) == 'file_metadata.csv':
            df_timestamps = pd.read_csv(timestampcsv_path)
        elif os.path.basename(timestampcsv_path) == 'timestamp.csv':
            df_timestamps = pd.read_csv(timestampcsv_path, header=None)
            df_timestamps.columns=['filename', 'timestamp']
            
    elif reply=='No':
        if path_dir is None: list_wav_paths = find_files(f_type=f_type, ext=ext, n_dir=n_dir)
        else : list_wav_paths = find_files(f_type=f_type, ext=ext, path=path_dir, msg=msg)
        
        if date_template is None:
            date_template = easygui.enterbox('Enter your time template')
        
        list_audio_file = [os.path.basename(wav_path) for wav_path in list_wav_paths]
        
        timestamp = []
        filename_raw_audio = []

        converted = convert_template_to_re(date_template)
        for i, filename in enumerate(list_audio_file):
            date_extracted = re.search(converted, str(filename))[0]
            date_obj = dt.datetime.strptime(date_extracted, date_template)
            dates = dt.datetime.strftime(date_obj, "%Y-%m-%dT%H:%M:%S.%f")
    
            dates_final = dates[:-3] + "Z"
            timestamp.append(dates_final)
            filename_raw_audio.append(filename)
            
        df_timestamps = pd.DataFrame(
                {'filename': filename_raw_audio, 'timestamp': timestamp, 'path': list_wav_paths})
        df_timestamps.sort_values(by=["timestamp"], inplace=True)
    
    if tz is not None:
        df_timestamps['timestamp'] = [pd.Timestamp(tz.localize(pd.Timestamp(i.split('Z')[0]))) for i in df_timestamps['timestamp']]
    
    return df_timestamps

def find_files(f_type:str, ext:str, path:str=None, msg:str=None, n_dir:int=1)->list:
    """ Based on selection_type, ask the user a folder and yields all the wav files inside it or ask the user multiple wav files

    Parameters :
        f_type : str, either 'dir' or 'file'
        ext : str, ex: 'wav'
        path : string, the user can specify the path of the askfolder dialog to open
        msg : string, the user can specify a message to display on the askfolder dialog

    
    Returns :
        selected_files : list of the paths of the wav files

    """
    root = Tk()
    root.withdraw()

    # Define the file types to display in the dialog

    selected_files = []

    if f_type == 'dir':
        
        directory=[]
        if path is None:
            for i in range(n_dir):
                directory.append(filedialog.askdirectory(initialdir = path, title='Select {0} folder {1}'.format(ext, i+1)))

        else: 
            directory = os.path.join(path,'wav')
        
        if directory:
            [selected_files.extend(glob.glob(os.path.join(d, '**/*.{0}'.format(ext)), recursive=True)) for d in directory]
            
    elif f_type == 'file':
        # If the user wants to select multiple files, show the file dialog
        file_paths = filedialog.askopenfilenames(initialdir = path, title='Select {0} files {1}'.format(ext, msg), filetypes=[('{0} files'.format(ext), '*.{0}'.format(ext))])
        selected_files.extend(file_paths)

    return selected_files

def get_tz(file):
    
    tz=[]
    if isinstance(file, list):
        if len(file)==1 : [file] = file  # Convert the single string to a list with one element
        else:
            for i in file:
               dt = pd.to_datetime(pd.read_csv(i, usecols=['start_datetime'])['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z').tolist()
               tz.append(list(set([x.tz for x in dt]))[0])
            tz=list(set(tz))
            if len(tz)==1: tz=tz[0]
            else: sys.exit('tz error')
        return tz
    else:
        dt = pd.to_datetime(pd.read_csv(file, usecols=['start_datetime'])['start_datetime'].tolist(), format='%Y-%m-%dT%H:%M:%S.%f%z')
        tz = list(set([x.tz for x in dt]))
    
        if len(tz)==1:
            return tz[0]
        elif len(tz)>1:
            print('More than one timezone present in file')
        else:print('error tz')

def input_date(msg, tz_data):
    """ Based on selection_type, ask the user a folder and yields all the wav files inside it or ask the user multiple wav files

    Parameters :
        msg : Message to tell the user what date they have to enter (begin, end...)
        tz_data : UTC object of pytz module 

    
    Returns :
        date_dt : aware dataframe of the date entered by the user

    """    
    title = "Date"
    fieldNames = ["Year", "Month", "Day", "Hour", "Minute", "Second"]
    fieldValues = []  # we start with blanks for the values
    fieldValues = easygui.multenterbox(msg,title, fieldNames)
    
    # make sure that none of the fields was left blank
    while 1:
      if fieldValues == None: break
      errmsg = ""
      for i in range(len(fieldNames)):
        if fieldValues[i].strip() == "":
          errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
      if errmsg == "": break # no problems found
      fieldValues = easygui.multpasswordbox(errmsg, title, fieldNames, fieldValues)
    print("Reply was:", fieldValues) 
    date_dt = dt.datetime(*map(int, fieldValues),0 ,tz_data)
    
    
    return date_dt










