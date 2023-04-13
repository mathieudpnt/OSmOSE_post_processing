from tqdm import tqdm
import os
import datetime as dt
import pytz
import glob
from tkinter import filedialog
from tkinter import Tk
import pandas as pd
import numpy as np
import time
import easygui
from post_processing_detections.utilities.def_func import read_header, extract_datetime, from_str2dt, from_str2ts, t_rounder, get_wav_info, sorting_annot_boxes


#%% LOAD DATA - User inputs

print('\n\nLoading data...', end='')

# tz_data='Europe/Paris'
# tz_data ='Etc/GMT-2' # UTC+2
# tz_data ='Etc/GMT-1' #UTC+1

#PAMGuard detections
root = Tk()
root.withdraw()
pamguard_path = filedialog.askopenfilename(title="Select PAMGuard detection file", filetypes=[("CSV files", "*.csv")])
tuple_pamguard = sorting_annot_boxes(pamguard_path)
dfpamguard = tuple_pamguard[-1]

#APLOSE annotations
root = Tk()
root.withdraw()
aplose_path = filedialog.askopenfilename(title="Select APLOSE annotation file", filetypes=[("CSV files", "*.csv")])

    
#WAV files 
root = Tk()
root.withdraw()
wavpath = filedialog.askdirectory(title = 'Select wav folder')
wav_files = glob.glob(os.path.join(wavpath, "**/*.wav"), recursive=True)
wav_list = [os.path.basename(file) for file in wav_files]
wav_folder = [os.path.dirname(file) for file in wav_files]
durations = [read_header(file)[-1] for file in wav_files]

print('\tDone!', end='\n')

#%% FORMAT DATA
print('\nFormating data...', end='\n')

start = time.time()

first_date = dfpamguard['start_datetime'][0] #1st detection
last_date = dfpamguard['start_datetime'].iloc[-1] #last detection

tuple_aplose = sorting_annot_boxes(aplose_path, tz_data, first_date, last_date)
time_bin = tuple_aplose[0]
fmax = tuple_aplose[1]
annotators = tuple_aplose[2]
labels = tuple_aplose[3]
df_aplose = tuple_aplose[-1]

## Time vector
wav_datetimes = [extract_datetime(x, tz=tz_data) for x in wav_list] #datetime of wav files

#selection of waf files according to first and last dates
idx_wav_beg = 0 if all(wav_datetimes[i] >= first_date for i in range(len(wav_datetimes))) else [i for i, x in enumerate(wav_datetimes) if x < first_date][-1]
idx_wav_end = len(wav_list) if all(wav_datetimes[i] <= last_date for i in range(len(wav_datetimes))) else [i for i, x in enumerate(wav_datetimes) if x > last_date][0]
wav_datetimes, wav_list, wav_folder, wav_files, durations = wav_datetimes[idx_wav_beg:idx_wav_end], wav_list[idx_wav_beg:idx_wav_end], wav_folder[idx_wav_beg:idx_wav_end], wav_files[idx_wav_beg:idx_wav_end], durations[idx_wav_beg:idx_wav_end]
print('\n1st wav : ' + wav_list[0])
print('last wav : ' + wav_list[-1], end='\n\n')

time_vector = [elem for i in range(len(wav_list)) for elem in extract_datetime(wav_list[i], tz=tz_data).timestamp() + np.arange(0, durations[i], time_bin).astype(int)]
time_vector_str = [str(wav_list[i]).split('.wav')[0]+ '_+'  + str(elem) for i in range(len(wav_list)) for elem in np.arange(0, durations[i], time_bin).astype(int)]


## Aplose
selected_label = ''.join(easygui.buttonbox('Select a label', 'Single plot', labels) if len(labels)>1 else labels)
selected_annotations = df_aplose.loc[(df_aplose['annotation'] == selected_label)]
    
times_Ap_beg = sorted(list(set(x.timestamp() for x in selected_annotations['start_datetime'])) )
times_Ap_end = sorted(list(set(y.timestamp() for y in selected_annotations['end_datetime']))) #set -> Remove recurrent elements ie annotator common annotations in the list, returns a set with random order -> list + sorting

Aplose_vec, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
for i in range(len(times_Ap_beg)):
    for j in range(k, len(time_vector)-1):
        if int(times_Ap_beg[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)) or int(times_Ap_end[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)):
                ranks.append(j)
                k=j
                break
        else: 
            continue 
ranks = sorted(list(set(ranks)))
Aplose_vec[np.isin(range(len(time_vector)), ranks)] = 1
    

## Pamguard
times_PG_beg = [i.timestamp() for i in dfpamguard['start_datetime']]
times_PG_end = [i.timestamp() for i in dfpamguard['end_datetime']]

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


##  DETECTION PERFORMANCES
print('\n\nDetection results :', end='\n')
true_pos, false_pos, true_neg, false_neg, error = 0,0,0,0,0
for i in range(len(time_vector)):
    if Aplose_vec[i] == 0 and PG_vec[i] == 0:
        true_neg+=1
    elif Aplose_vec[i] == 1 and PG_vec[i] == 1:
        true_pos+=1
    elif Aplose_vec[i] == 0 and PG_vec[i] == 1:
        false_pos+=1   
    elif Aplose_vec[i] == 1 and PG_vec[i] == 0:
        false_neg+=1
    else:error+=1
        
if error == 0:
    print('\tTrue positive : ', true_pos)
    print('\tTrue negative : ', true_neg)
    print('\tFalse positive : ', false_pos)
    print('\tFalse negative : ', false_neg)   
    
    print('\nPRECISION : ', round(true_pos/(true_pos+false_pos),3))
    print('RECALL : ', round(true_pos/(false_neg+true_pos) ,3    ), end='\n\n')
    print('Label : ', selected_label)
else: print('Error : ', error)
    
end = time.time()
print('\nElapsed time : ', round(end-start,2), 's')  



