from tqdm import tqdm
import os
import glob
from tkinter import filedialog
from tkinter import Tk
import sys
import numpy as np
import time
import datetime as dt
import pandas as pd
import pytz
from utilities.def_func import input_date, get_tz, read_header, get_timestamps, sorting_detections, get_csv_file, extract_datetime, t_rounder, pick_datetimes, export2Raven, n_random_hour

#%% LOAD DATA - User inputs
# /!\ warning : the file_metadata.csv OR the wav files are necessary for this scipt /!\

#PAMGuard detections
pamguard_path = get_csv_file(1)[0]
df_pamguard, t_pamguard = sorting_detections(files=pamguard_path, timebin_new=60, tz=pytz.FixedOffset(120))

time_bin = t_pamguard['max_time'][0]
fmax = t_pamguard['max_freq'][0]
annotators = t_pamguard['annotators'][0]
labels = t_pamguard['labels'][0]
tz_data = df_pamguard['start_datetime'][0].tz

#WAV files 
# Chose your mode :
    # input : you will fill a dialog box with the start and end date of the Figure you want to make
    # auto : the script automatically extract the timestamp from the file_metadata.csv file or from the selected wav files of the Figure you want to make

dt_mode = 'auto'

if dt_mode == 'auto':
    filemetadata = get_timestamps()
    begin_deploy = extract_datetime(filemetadata['filename'].iloc[0], tz_data)
    end_deploy = extract_datetime(filemetadata['filename'].iloc[-1], tz_data) + dt.timedelta(seconds=filemetadata['duration'].iloc[-1])  
    durations = filemetadata['duration']
        
elif dt_mode == 'manual' :
    timestamps_file = get_timestamps()
    filemetadata = get_timestamps()
    begin_deploy = extract_datetime(filemetadata['filename'].iloc[0], tz_data)
    wav_path = timestamps_file['path']
    durations = [read_header(i)[-1] for i in tqdm(wav_path)]
    end_deploy = extract_datetime(timestamps_file['filename'].iloc[-1], tz_data) + dt.timedelta(seconds=durations[-1])  


wav_names = filemetadata['filename']
wav_datetimes = [extract_datetime(i, tz=tz_data) for i in filemetadata['timestamp']]

#%% FORMAT DATA

time_vector = [elem for i in range(len(filemetadata)) for elem in wav_datetimes[i].timestamp() + np.arange(0, durations[i], time_bin).astype(int)]
time_vector_str = [str(wav_names[i]).split('.wav')[0]+ '_+'  + str(elem) for i in range(len(wav_names)) for elem in np.arange(0, durations[i], time_bin).astype(int)]


## Pamguard
times_PG_beg = [df_pamguard['start_datetime'][i].timestamp() for i in range(len(df_pamguard))]
times_PG_end = [df_pamguard['end_datetime'][i].timestamp() for i in range(len(df_pamguard))]

PG_vec, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
for i in range(len(times_PG_beg)):
    for j in range(k, len(time_vector)-1):
        if int(times_PG_beg[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)) or int(times_PG_end[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)):
                ranks.append(j)
                k=j
                break
        else: continue 
ranks = sorted(list(set(ranks)))
PG_vec[np.isin(range(len(time_vector)), ranks)] = 1

##  DETECTIONS
print('\n\nDetections : ', sum(PG_vec))
print('Label : ', labels)
    
#%% DOUBLE CHECK a proportion of the detection

# Create a new list to hold the selected timestamps
selected_time_vector = []

# tv_year = ([dt.datetime.fromtimestamp(time_vector[i]).year for i in range(len(time_vector))])
# tv_month = ([dt.datetime.fromtimestamp(time_vector[i]).month for i in range(len(time_vector))])
# tv_day = ([dt.datetime.fromtimestamp(time_vector[i]).day for i in range(len(time_vector))])
# tv_hour = ([dt.datetime.fromtimestamp(time_vector[i]).hour for i in range(len(time_vector))])

while True:
#     # selected_time_vector = [time_vector[i] for i in range(len(time_vector)) if tv_hour[i]%2 == 0] #select even hours
#     # selected_time_vector, selected_PG_vec, selected_time_vector_str, selected_dates = oneday_per_month(time_vector, time_vector_str, PG_vec) #select randomly one day per month
    selected_time_vector, selected_time_vector_str, selected_PG_vec, selected_dates = n_random_hour(time_vector_ts=time_vector, time_vector_str=time_vector_str, vec=PG_vec, n_hour=3, tz=tz_data, time_step=time_bin) #select randomly n hour in the dataset
    if round(sum(selected_PG_vec)/len(selected_time_vector),3) > 0.75*round(sum(PG_vec)/len(time_vector),3) and round(sum(selected_PG_vec)/len(selected_time_vector),3) < 1.25*round(sum(PG_vec)/len(time_vector),3):
        break

# selected_datetimes,  selected_durations = ['07/07/2022 00:00:00'], ['1d']
# selected_time_vector, selected_time_vector_str, selected_PG_vec, selected_dates = pick_datetimes(time_vector, time_vector_str, PG_vec, selected_datetimes, selected_durations, tz_data)


print('\n\nselected dates : {0}'.format(selected_dates))
print('\n%%% Positive detection rate %%%', end='\n')
print('\t original :', round(sum(PG_vec)/len(time_vector),3)) #proportion de positifs dans les éléments
print('\tselection :', round(sum(selected_PG_vec)/len(selected_time_vector),3))  #proportion de positifs dans les éléments selectionnés randomly



#%% EXPORT TO RAVEN FORMAT

double_check_dir = os.path.join(os.path.dirname(pamguard_path), 'Double_check')
if not os.path.exists(double_check_dir): os.mkdir(double_check_dir)
double_check_path = double_check_dir

result_dir = os.path.join(double_check_path, dt.datetime.now().strftime('%Y-%m-%dT%H_%M_%S'))
if not os.path.exists(result_dir): os.mkdir(result_dir)
result_path = result_dir

with open(os.path.join(result_path, "configuration.txt"), "w+") as f:
    f.write("Selected dates :\n")
    # [f.write(selected_dates['datetimes'][i] +"\t"+ selected_dates['durations'][i] + "\n") for i in range(len(selected_dates))]
    [f.write(selected_dates[i]+"\n") for i in range(len(selected_dates))]


wav_tuple = (wav_names, wav_datetimes, durations)

df0_PG2Raven =  export2Raven(wav_tuple, time_vector, time_vector_str, 0.9*fmax, selection_vec=None)
PG2Raven_str0 = "PG_double_check0_all_" + t_rounder(wav_datetimes[0], res=3600).strftime('%y%m%d') + '_' + t_rounder(wav_datetimes[-1], res=3600).strftime('%y%m%d') + '_'+ str(time_bin) + 's' + '.txt'
df0_PG2Raven.to_csv(os.path.join(result_path, PG2Raven_str0), index=False, sep='\t')  

df1_PG2Raven =  export2Raven(wav_tuple, time_vector, time_vector_str, 0.8*fmax, selection_vec=PG_vec)
PG2Raven_str1 = "PG_double_check1_all_positives_" + t_rounder(wav_datetimes[0]).strftime('%y%m%d') + '_' + t_rounder(wav_datetimes[-1]).strftime('%y%m%d') + '_'+ str(time_bin) + 's' + '.txt'
df1_PG2Raven.to_csv(os.path.join(result_path, PG2Raven_str1), index=False, sep='\t')  

df2_PG2Raven =  export2Raven(wav_tuple, selected_time_vector, selected_time_vector_str, 0.6*fmax, selection_vec=None)
PG2Raven_str2 = "PG_double_check2_selected_" + t_rounder(wav_datetimes[0]).strftime('%y%m%d') + '_' + t_rounder(wav_datetimes[-1]).strftime('%y%m%d') + '_'+ str(time_bin) + 's' + '.txt'
df2_PG2Raven.to_csv(os.path.join(result_path, PG2Raven_str2), index=False, sep='\t')  

df3_PG2Raven =  export2Raven(wav_tuple, selected_time_vector, selected_time_vector_str, 0.4*fmax, selection_vec=selected_PG_vec)
PG2Raven_str3 = "PG_double_check3_selected_positives_" + t_rounder(wav_datetimes[0]).strftime('%y%m%d') + '_' + t_rounder(wav_datetimes[-1]).strftime('%y%m%d') + '_'+ str(time_bin) + 's' + '.txt'
df3_PG2Raven.to_csv(os.path.join(result_path, PG2Raven_str3), index=False, sep='\t')  

print('\n\nRaven double check files exported to:\n'+ os.path.dirname(pamguard_path))






