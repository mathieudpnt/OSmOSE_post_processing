import struct
from typing import Tuple
import pytz
import pandas as pd
import re
import datetime as dt


def read_header(file:str) -> Tuple[int, int, int, int, int]:
    #reads header of a wav file to get info such as duration, samplerate etc...
    
    with open(file, "rb") as fh:
       _, size, _ = struct.unpack('<4sI4s', fh.read(12))

       chunk_header = fh.read(8)
       subchunkid, _ = struct.unpack('<4sI', chunk_header)

       if (subchunkid == b'fmt '):
           _, channels, samplerate, _, _, sampwidth = struct.unpack('HHIIHH', fh.read(16))

       sampwidth = (sampwidth + 7) // 8
       framesize = channels * sampwidth
       frames = size // framesize 
       return sampwidth, frames, samplerate, channels, frames/samplerate

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
    
def sorting_annot_boxes(file, tz, date_begin, date_end) -> Tuple[int, int, list, list, pd.DataFrame]:
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