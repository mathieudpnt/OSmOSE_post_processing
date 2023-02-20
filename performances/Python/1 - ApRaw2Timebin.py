from tqdm import tqdm
import os
import datetime as dt
import pytz
import glob
from tkinter import filedialog
from tkinter import Tk
import pandas as pd
import sys
import numpy as np
import time
import easygui
sys.path.append('U:/Documents/Git/spyder_scripts')
from def_func import read_header, extract_datetime, from_str2dt, from_str2ts, t_rounder, get_wav_info


#%% LOAD DATA - User inputs

print('\nLoading data...', end='')

tz_data='Europe/Paris'
Ap=0

#PAMGuard raw detections
root = Tk()
root.withdraw()
pamguard_path = filedialog.askopenfilename(title="Select PAMGuard detection file", filetypes=[("CSV files", "*.csv")])
dfpamguard = pd.read_csv(pamguard_path).sort_values('start_datetime')

   
#WAV files 
root = Tk()
root.withdraw()
wavpath = filedialog.askdirectory(title = 'Select wav folder')
wav_files = glob.glob(os.path.join(wavpath, "**/*.wav"), recursive=True)
wav_list = [os.path.basename(file) for file in wav_files]
wav_folder = [os.path.dirname(file) for file in wav_files]
# durations = [read_header(file)[-1] for file in wav_files]
durations = get_wav_info(wavpath)
fmax = 0.5*read_header(wav_files[0])[2]

print('\tDone!', end='\n')
#%% FORMAT DATA
print('\nFormating data...', end='\n')

start = time.time()
time_bin_duration = easygui.integerbox('Enter time bin duration (s):', title = 'Timebin', lowerbound = 10, upperbound = 86400)

## Time vector
wav_datetimes = [extract_datetime(x) for x in wav_list] #datetime of wav files

first_date = from_str2dt(dfpamguard['start_datetime'][0]) #1st detection
last_date = from_str2dt(dfpamguard['start_datetime'].iloc[-1]) #last detection

#selection of waf files according to first and last dates
idx_wav_beg = 0 if all(wav_datetimes[i] >= first_date for i in range(len(wav_datetimes))) else [i for i, x in enumerate(wav_datetimes) if x < first_date][-1]
idx_wav_end = len(wav_list) if all(wav_datetimes[i] <= last_date for i in range(len(wav_datetimes))) else [i for i, x in enumerate(wav_datetimes) if x > last_date][0]
wav_datetimes, wav_list, wav_folder, wav_files, durations = wav_datetimes[idx_wav_beg:idx_wav_end], wav_list[idx_wav_beg:idx_wav_end], wav_folder[idx_wav_beg:idx_wav_end], wav_files[idx_wav_beg:idx_wav_end], durations[idx_wav_beg:idx_wav_end]
print('\n1st wav : ' + wav_list[0])
print('last wav : ' + wav_list[-1], end='\n\n')

time_vector = [elem for i in range(len(wav_list)) for elem in extract_datetime(wav_list[i]).timestamp() + np.arange(0, durations[i], time_bin_duration).astype(int)]
time_vector_str = [str(wav_list[i]).split('.wav')[0]+ '_+'  + str(elem) for i in range(len(wav_list)) for elem in np.arange(0, durations[i], time_bin_duration).astype(int)]


## Pamguard
times_PG_beg = [from_str2ts(x) for x in dfpamguard['start_datetime']]
times_PG_end = [from_str2ts(x) for x in dfpamguard['end_datetime']]

PG_vec, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
for i in tqdm(range(len(times_PG_beg)), 'Importing PAMGuard detections...'):
    for j in range(k, len(time_vector)-1):
        if int(times_PG_beg[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)) or int(times_PG_end[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)):
                ranks.append(j)
                k=j
                break
        else: 
            continue 
ranks = sorted(list(set(ranks)))
# for i in tqdm(range(len(time_vector)), 'Importing PAMGuard detections...'): PG_vec.append(1) if i in ranks else PG_vec.append(0)
PG_vec[ranks] = 1
PG_vec = list(PG_vec)
print('\tDone!', end='\n')

#%% EXPORT RESHAPPED DETECTIONS

start_datetime_str, end_datetime_str, filename = [],[],[]
for i in range(len(time_vector)):
    if PG_vec[i] == 1:
        start_datetime = pytz.timezone('UTC').localize(dt.datetime.utcfromtimestamp(time_vector[i])) #
        start_datetime = start_datetime.astimezone(pytz.timezone(tz_data))
        start_datetime_str.append(start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[:-8]+ start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-5:-2] +':' + start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-2:])
        end_datetime = pytz.timezone('UTC').localize(dt.datetime.utcfromtimestamp(time_vector[i]+10))
        end_datetime = end_datetime.astimezone(pytz.timezone(tz_data))
        end_datetime_str.append(end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[:-8]+ end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-5:-2] +':' + end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-2:])
        filename.append(time_vector_str[i])

## APLOSE

df_PG2Aplose = pd.DataFrame()
dataset_str = list(set(dfpamguard['dataset']))

df_PG2Aplose['dataset'] = dataset_str*len(start_datetime_str)
df_PG2Aplose['filename'] = filename
df_PG2Aplose['start_time'] = [0]*len(start_datetime_str)
df_PG2Aplose['end_time'] = [time_bin_duration]*len(start_datetime_str)
df_PG2Aplose['start_frequency'] = [0]*len(start_datetime_str)
df_PG2Aplose['end_frequency'] = [fmax]*len(start_datetime_str)

df_PG2Aplose['annotation'] = list(set(dfpamguard['annotation']))*len(start_datetime_str)
df_PG2Aplose['annotator'] = list(set(dfpamguard['annotator']))*len(start_datetime_str)
  
df_PG2Aplose['start_datetime'], df_PG2Aplose['end_datetime'] = start_datetime_str, end_datetime_str

PG2Ap_str = "/PG_formatteddata_" + t_rounder(wav_datetimes[0]).strftime('%y%m%d') + '_' + t_rounder(wav_datetimes[-1]).strftime('%y%m%d') +'_'+ str(time_bin_duration) + 's'+ '.csv'

df_PG2Aplose.to_csv(os.path.dirname(pamguard_path) + PG2Ap_str, index=False)  
print('\n\nAplose formatted data file exported to '+ os.path.dirname(pamguard_path))

## RAVEN
df_PG2Raven = pd.DataFrame()

df_PG2Raven['Selection'] = np.arange(1,len(start_datetime_str)+1)
df_PG2Raven['View'], df_PG2Raven['Channel'] = [1]*len(start_datetime_str), [1]*len(start_datetime_str)

datetime_endfiles, datetime_begfiles=[],[]
for i in range(len(wav_list)):
    datetime_endfiles.append((wav_datetimes[i]+dt.timedelta(seconds=durations[i])).strftime('%Y-%m-%d %H:%M:%S.%f'))
    datetime_begfiles.append((wav_datetimes[i]).strftime('%Y-%m-%d %H:%M:%S.%f'))
    # datetime_begfiles.append(dt.datetime.utcfromtimestamp(time_vector[idx_beg[i]]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]) #datetime beginning of next file according to title

# df_Raven = pd.DataFrame({'datetime begin': datetime_begfiles, 'datetime begin + duration': datetime_endfiles})

offsets =[]
for i in range(len(datetime_endfiles)-1):
    offsets.append(((wav_datetimes[i]+dt.timedelta(seconds=durations[i])).timestamp() - (wav_datetimes[i+1]).timestamp()))
    offsets_cumsum=(list(np.cumsum([offsets[i] for i in range(len(offsets))])))
    offsets_cumsum.insert(0, 0)

test3 = [wav_list[i].split('.wav')[0] for i in range(len(wav_list))] #names of the waves without extension
start_datetime, end_datetime = [],[] 
for i in range(len(time_vector)):
    if PG_vec[i] == 1:
        test4 = [time_vector_str[i].split('_+')[0] == test3[j] for j in range(len(wav_list))] #finding which wav the detection is belonging to
        idx_wav_Raven = [i for i, x in enumerate(test4) if x][0] #index of the wav the detection i is belonging to
        start_datetime.append(int(time_vector[i] - wav_datetimes[0].timestamp())      + offsets_cumsum[idx_wav_Raven] )
        end_datetime.append(int(time_vector[i] - wav_datetimes[0].timestamp())+10     + offsets_cumsum[idx_wav_Raven] )

df_PG2Raven['Begin Time (s)'] = start_datetime     
df_PG2Raven['End Time (s)'] = end_datetime     

df_PG2Raven['Low Freq (Hz)'] = [0]*len(start_datetime_str)
df_PG2Raven['High Freq (Hz)'] = [0.8*fmax]*len(start_datetime_str)

PG2Raven_str = "/PG_formatteddata_" + t_rounder(wav_datetimes[0]).strftime('%y%m%d') + '_' + t_rounder(wav_datetimes[-1]).strftime('%y%m%d') + '_'+ str(time_bin_duration) + 's' + '.txt'

df_PG2Raven.to_csv(os.path.dirname(pamguard_path) + PG2Raven_str, index=False, sep='\t')  
print('\n\nRaven formatted data file exported to '+ os.path.dirname(pamguard_path))




