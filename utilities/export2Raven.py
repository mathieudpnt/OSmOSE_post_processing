import pytz
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
from utilities.def_func import get_csv_file, sorting_detections, get_timestamps, extract_datetime, export2Raven

# Load data

file = get_csv_file(1)

df_detections, t_detections = sorting_detections(file=file[0], timebin_new=60, tz=pytz.FixedOffset(120))
timebin_detections = t_detections['max_time'][0]
fmax = t_detections['max_freq'][0]

timestamps_file = get_timestamps()
wav_names = timestamps_file['filename']
wav_datetimes = [extract_datetime(d, tz=pytz.FixedOffset(120)) for d in timestamps_file['timestamp']]
durations = timestamps_file['duration']
wav_tuple = (wav_names, wav_datetimes, durations)

# Export detections

df_Raven = export2Raven(tuple_info=wav_tuple, timestamps=timestamps_file, df=df_detections, timebin_new=timebin_detections, bin_height=0.8 * fmax, selection_vec=True)

PG2Raven_str = file[0].split('.csv')[0] + '_60s' + '.txt'
df_Raven.to_csv(PG2Raven_str, index=False, sep='\t')
