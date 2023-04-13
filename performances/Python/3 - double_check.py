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
from post_processing_detections.utilities.def_func import read_header, extract_datetime, from_str2dt, from_str2ts, t_rounder, get_wav_info, sorting_annot_boxes, pick_datetimes, export2Raven

#%% LOAD DATA - User inputs

print('\n\nLoading data...', end='')

#PAMGuard detections
root = Tk()
root.withdraw()
pamguard_path = filedialog.askopenfilename(title="Select PAMGuard detection file", filetypes=[("CSV files", "*.csv")])
tuple_pamguard = sorting_annot_boxes(pamguard_path)
time_bin = tuple_pamguard[0]
fmax = tuple_pamguard[1]
annotators = tuple_pamguard[2]
labels = tuple_pamguard[3]
dfpamguard = tuple_pamguard[-1]
tz_data = dfpamguard['start_datetime'][0].tz


#WAV files 
root = Tk()
root.withdraw()
wavpath = filedialog.askdirectory(title = 'Select wav folder')
wav_files = glob.glob(os.path.join(wavpath, "**/*.wav"), recursive=True)
wav_list = [os.path.basename(file) for file in wav_files]
wav_folder = [os.path.dirname(file) for file in wav_files]
wav_datetimes = [extract_datetime(file, tz=tz_data) for file in wav_list] #datetime of wav files

durations = get_wav_info(wavpath)
wav_tuple = (wav_list, wav_datetimes, durations)

print('\tDone!', end='\n')

#%% FORMAT DATA
print('\nFormating data...', end='\n')

start = time.time()

first_date = dfpamguard['start_datetime'][0] #1st detection
last_date = dfpamguard['start_datetime'].iloc[-1] #last detection

## Time vector

#selection of waf files according to first and last dates
idx_wav_beg = 0 if all(wav_datetimes[i] >= first_date for i in range(len(wav_datetimes))) else [i for i, x in enumerate(wav_datetimes) if x < first_date][-1]
idx_wav_end = len(wav_list) if all(wav_datetimes[i] <= last_date for i in range(len(wav_datetimes))) else [i for i, x in enumerate(wav_datetimes) if x > last_date][0]
wav_datetimes, wav_list, wav_folder, wav_files, durations = wav_datetimes[idx_wav_beg:idx_wav_end], wav_list[idx_wav_beg:idx_wav_end], wav_folder[idx_wav_beg:idx_wav_end], wav_files[idx_wav_beg:idx_wav_end], durations[idx_wav_beg:idx_wav_end]
print('\n1st wav : ' + wav_list[0])
print('last wav : ' + wav_list[-1], end='\n\n')

time_vector_test = [elem for i in range(len(wav_list)) for elem in np.arange(0, durations[i], time_bin).astype(int)]
time_vector = [elem for i in range(len(wav_list)) for elem in extract_datetime(wav_list[i], tz_data).timestamp() + np.arange(0, durations[i], time_bin).astype(int)]
time_vector_str = [str(wav_list[i]).split('.wav')[0]+ '_+'  + str(elem) for i in range(len(wav_list)) for elem in np.arange(0, durations[i], time_bin).astype(int)]


## Pamguard
times_PG_beg = [dfpamguard['start_datetime'][i].timestamp() for i in range(len(dfpamguard))]
times_PG_end = [dfpamguard['end_datetime'][i].timestamp() for i in range(len(dfpamguard))]

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
# PG_vec = [1 if i in ranks else 0 for i in tqdm(range(len(time_vector)), 'Importing PAMGuard detections...')] #takes too long
PG_vec[np.isin(range(len(time_vector)), ranks)] = 1


##  DETECTIONS
print('\n\nDetections : ', sum(PG_vec))
print('Label : ', labels[0])
    
end = time.time()
print('\nElapsed time : ', round(end-start,2), 's')  

#%% DOUBLE CHECK

# Create a new list to hold the selected timestamps
selected_time_vector = []

# tv_year = ([dt.datetime.fromtimestamp(time_vector[i]).year for i in range(len(time_vector))])
# tv_month = ([dt.datetime.fromtimestamp(time_vector[i]).month for i in range(len(time_vector))])
# tv_day = ([dt.datetime.fromtimestamp(time_vector[i]).day for i in range(len(time_vector))])
# tv_hour = ([dt.datetime.fromtimestamp(time_vector[i]).hour for i in range(len(time_vector))])

# while True:
#     # selected_time_vector = [time_vector[i] for i in range(len(time_vector)) if tv_hour[i]%2 == 0] #select even hours
#     # selected_time_vector, selected_PG_vec, selected_time_vector_str, selected_dates = oneday_per_month(time_vector, time_vector_str, PG_vec) #select randomly one day per month
#     # selected_time_vector, selected_time_vector_str, selected_PG_vec, selected_dates = n_random_hour(time_vector, time_vector_str, PG_vec, 2, tz_data)
#     if round(sum(selected_PG_vec)/len(selected_time_vector),3) > 0.75*round(sum(PG_vec)/len(time_vector),3) and round(sum(selected_PG_vec)/len(selected_time_vector),3) < 1.25*round(sum(PG_vec)/len(time_vector),3):
#         break

selected_datetimes,  selected_durations = ['07/07/2022 00:00:00'], ['1d']
selected_time_vector, selected_time_vector_str, selected_PG_vec, selected_dates = pick_datetimes(time_vector, time_vector_str, PG_vec, selected_datetimes, selected_durations, tz_data)


print('\n', selected_dates)
print('Taux de détection positives :', end='\n')
print('selection :', round(sum(selected_PG_vec)/len(selected_time_vector),3))  #proportion de positifs dans les éléments selectionnés randomly
print('original :', round(sum(PG_vec)/len(time_vector),3)) #proportion de positifs dans les éléments

#%% EXPORT TO RAVEN FORMAT

double_check_dir = os.path.join(os.path.dirname(pamguard_path), 'Double_check')
if not os.path.exists(double_check_dir): os.mkdir(double_check_dir)
double_check_path = double_check_dir

result_dir = os.path.join(double_check_path, dt.datetime.now().strftime('%Y-%m-%dT%H_%M_%S'))
if not os.path.exists(result_dir): os.mkdir(result_dir)
result_path = result_dir

with open(os.path.join(result_path, "configuration.txt"), "w+") as f:
    f.write("Selected dates :\n")
    [f.write(selected_dates['datetimes'][i] +"\t"+ selected_dates['durations'][i] + "\n") for i in range(len(selected_dates))]


wav_tuple = (wav_list, wav_datetimes, durations)

df0_PG2Raven =  export2Raven(wav_tuple, time_vector, time_vector_str, 0.9*fmax, selection_vec=None)
PG2Raven_str0 = "PG_double_check0_all_" + t_rounder(wav_datetimes[0]).strftime('%y%m%d') + '_' + t_rounder(wav_datetimes[-1]).strftime('%y%m%d') + '_'+ str(time_bin) + 's' + '.txt'
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





