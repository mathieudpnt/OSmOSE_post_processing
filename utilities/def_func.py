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


def extract_datetime(var, formats=None, tz=None):
    #extract datetime from filename such as Apocado / Cetiroise or custom ones
    
    if tz is None : tz = 'Europe/Paris'
    if formats is None:
        formats = [r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', r'\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}'] #add more format if necessary
    match = None
    for f in formats:
        match = re.search(f, var)
        if match:
            break
    if match:
        dt_string = match.group()
        if f == r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}':
            dt_format = '%Y-%m-%d_%H-%M-%S'
        elif f == r'\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}':
            dt_format = '%y%m%d%H%M%S'
        elif f == r'\d{2}\d{2}\d{2}_\d{2}\d{2}\d{2}':
            dt_format = '%y%m%d_%H%M%S'
        date_obj = dt.datetime.strptime(dt_string, dt_format)
        date_obj = pytz.timezone(tz).localize(date_obj)
        return date_obj
    else:
        return print("No datetime found")
    
def sorting_annot_boxes(file, tz, date_begin=None, date_end=None) -> Tuple[int, int, list, list, pd.DataFrame]:
    # From an Aplose results csv, returns a DataFrame without the Aplose box annotations (weak annotations)

    df = pd.read_csv(file)
    max_freq = int(max(df['end_frequency']))
    max_time = int(max(df['end_time']))
    df = df.loc[(df['start_time'] == 0) & (df['end_time'] == max_time) & (df['end_frequency'] == max_freq)] #deletion of boxes
    df = df.sort_values('start_datetime') #sorting value according to datetime_start
    df = df.reset_index(drop=True) #reset the indexes of row after sorting the df
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%Z')
    df['end_datetime'] = pd.to_datetime(df['end_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%Z')
    df['start_datetime'] = [x.tz_convert(tz) for x in df['start_datetime']] #converting to desired tz
    df['end_datetime'] = [x.tz_convert(tz) for x in df['end_datetime']] #converting to desired tz
    if date_begin is not None and date_end is not None: 
        df = df[(df['start_datetime']>=date_begin) & (df['start_datetime']<=date_end)] #select data within [date_begin;date_end]

    annotators = list(df['annotator'].drop_duplicates())
    labels = list(df['annotation'].drop_duplicates())
    
    # print(len(df), 'annotations')
    return (max_time, max_freq, annotators, labels, df)

def t_rounder(t):
    # Rounds to nearest 10-minute interval

    minute = t.minute
    minute = (minute + 5) // 10 * 10
    if minute >= 60:
            minute = 0
            hour = t.hour + 1
            if hour >= 24:
                hour = 0
                t += dt.timedelta(days=1)
            t = t.replace(hour=hour, minute=minute, second=0, microsecond=0)
    else: t = t.replace(minute=minute, second=0, microsecond=0)
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



def n_random_hour(time_vector_ts, time_vector_str, vec, n_hour, TZ)-> Tuple[list, list, list, list]:
    # randomly select n non-overlapping hours from the time vector
    
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
        possible_datetimes = [time for time in time_vector_ts if rand_datetime < time < rand_datetime + 3600]
    
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
    selected_dates = [dt.datetime.fromtimestamp(i, pytz.timezone(TZ)).strftime('%d/%m/%Y %H:%M:%S') for i in selected_dates]

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
    end_datetime = [int(time_vec[i] - file_datetimes[0].timestamp())+(time_vec[i+1]-time_vec[i]) + offsets_cumsum[idx_wav_Raven[i]]  for i in tqdm(range(len(time_vec)-1), position=0, leave=True, desc = '3/3') if selection_vec[i] == 1]
    
    df_PG2Raven = pd.DataFrame()
    df_PG2Raven['Selection'] = np.arange(1,len(start_datetime)+1)
    df_PG2Raven['View'], df_PG2Raven['Channel'] = [1]*len(start_datetime), [1]*len(start_datetime)
    df_PG2Raven['Begin Time (s)'] = start_datetime     
    df_PG2Raven['End Time (s)'] = end_datetime     
    df_PG2Raven['Low Freq (Hz)'] = [0]*len(start_datetime)
    df_PG2Raven['High Freq (Hz)'] = [bin_height]*len(start_datetime)
    
    return df_PG2Raven























