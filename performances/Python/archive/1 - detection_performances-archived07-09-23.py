# This script is used in order to compute the detection performances of a detection file
# the computed metrics are so far precision and recall
# it takes as an input 2 APLOSE formatted detection/annotation files
# the user has to select one of the 2 files as the reference/"ground truth" to calcultate the performance of the second file

import numpy as np
import easygui
from post_processing_detections.utilities.def_func import get_detection_files, extract_datetime, sorting_detections, get_timestamps, get_tz
import pytz
import pandas as pd
#%% LOAD DATA - User inputs

files_list = get_detection_files(2)

choice_ref = easygui.buttonbox('Select the reference', '{0}'.format(files_list[0].split('/')[-1]), [file.split('/')[-1] for file in files_list]) 
if files_list[0].split('/')[-1] != choice_ref:
    files_list[0],  files_list[1] = files_list[1],  files_list[0] 

timestamps_file = get_timestamps(tz=get_tz(files_list), f_type='dir', ext= 'wav')

df_detections, t_detections = sorting_detections(files=files_list, timebin_new=60)
timebin_detections = int(list(set(t_detections['max_time']))[0])
labels_detections = list(set(t_detections['labels'].explode()))
annotators_detections = list(set(t_detections['annotators'].explode()))

first_date = pd.Timestamp('2022-08-28 00:00:00+0200', tz=pytz.FixedOffset(120))
last_date = pd.Timestamp('2022-08-28 23:59:00+0200', tz=pytz.FixedOffset(120))

tz_data = timestamps_file['timestamp'][0].tz

wav_files = timestamps_file['filename']
wav_datetimes = timestamps_file['timestamp']

annotator1 = easygui.buttonbox('Select annotator 1 (reference)', 'file 1 : {0}'.format(files_list[0].split('/')[-1]), annotators_detections) if len(annotators_detections)>1 else annotators_detections[0]
annotator2 = easygui.buttonbox('Select annotator 2', 'file 2 : {0}'.format(files_list[1].split('/')[-1]), annotators_detections) if len(annotators_detections)>1 else annotators_detections[0]

df1, t1 = sorting_detections(files=files_list[0], annotator=annotator1, timebin_new=timebin_detections, tz=tz_data, date_begin=first_date, date_end=last_date)
df2, t2 = sorting_detections(files=files_list[1], annotator=annotator2, timebin_new=timebin_detections, tz=tz_data, date_begin=first_date, date_end=last_date)

labels1 = t1['labels'][0]
labels2 = [t2['labels'][0]]
# %% FORMAT DATA

# selection of waf files according to first and last dates
idx_wav_beg = 0 if all(wav_datetimes[i] >= first_date for i in range(len(wav_datetimes))) else [i for i, x in enumerate(wav_datetimes) if x < first_date][-1]
idx_wav_end = len(wav_files) if all(wav_datetimes[i] <= last_date for i in range(len(wav_datetimes))) else [i for i, x in enumerate(wav_datetimes) if x > last_date][0]
wav_datetimes, wav_files = wav_datetimes[idx_wav_beg:idx_wav_end], wav_files[idx_wav_beg:idx_wav_end]
print('\n1st wav : ' + wav_files.iloc[0])
print('last wav : ' + wav_files.iloc[-1], end='\n\n')

time_vector = [elem for i in range(len(wav_files)) for elem in [extract_datetime(wav_files.iloc[i], tz=tz_data).timestamp()]]
time_vector_str = [wav_files.iloc[i].split('.wav')[0] for i in range(len(wav_files))]

# # df1 - REFERENCE
selected_label1 = easygui.buttonbox('Select a label', 'file 1 : {0}'.format(files_list[0].split('/')[-1]), labels1) if len(labels1)>1 else labels1[0]
selected_annotations1, _ = sorting_detections(files=files_list[0], timebin_new=timebin_detections, annotator=annotator1, label=selected_label1, date_begin=first_date, date_end=last_date)
    
times1_beg = sorted(list(set(x.timestamp() for x in selected_annotations1['start_datetime'])))
times1_end = sorted(list(set(y.timestamp() for y in selected_annotations1['end_datetime'])))

vec1, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
for i in range(len(times1_beg)):
    for j in range(k, len(time_vector)-1):
        if int(times1_beg[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)) or int(times1_end[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)):
                ranks.append(j)
                k=j
                break
        else: 
            continue 
ranks = sorted(list(set(ranks)))
vec1[np.isin(range(len(time_vector)), ranks)] = 1
    

# df2
selected_label2 = easygui.buttonbox('Select a label', '{0}'.format(files_list[1].split('/')[-1]), labels2) if len(labels2)>1 else labels2[0]
selected_annotations2, _ = sorting_detections(files=files_list[1], timebin_new=timebin_detections, annotator = annotator2, label=selected_label2, date_begin=first_date, date_end=last_date)

times2_beg = [i.timestamp() for i in selected_annotations2['start_datetime']]
times2_end = [i.timestamp() for i in selected_annotations2['end_datetime']]

vec2, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
for i in range(len(times2_beg)):
    for j in range(k, len(time_vector)-1):
        if int(times2_beg[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)) or int(times2_end[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)):
                ranks.append(j)
                k=j
                break
        else: 
            continue 
ranks = sorted(list(set(ranks)))
vec2[np.isin(range(len(time_vector)), ranks)] = 1


##  DETECTION PERFORMANCES
true_pos, false_pos, true_neg, false_neg, error = 0,0,0,0,0
for i in range(len(time_vector)):
    if vec1[i] == 0 and vec2[i] == 0:
        true_neg+=1
    elif vec1[i] == 1 and vec2[i] == 1:
        true_pos+=1
    elif vec1[i] == 0 and vec2[i] == 1:
        false_pos+=1   
    elif vec1[i] == 1 and vec2[i] == 0:
        false_neg+=1
    else:error+=1

print('\n\nDetection results :', end='\n') 
if error == 0:
    print('\tTrue positive : {0}'.format(true_pos))
    print('\tTrue negative : {0}'.format(true_neg))
    print('\tFalse positive : {0}'.format(false_pos))
    print('\tFalse negative : {0}'.format(false_neg))
    
    print('\nPRECISION : {0:.2f}'.format(true_pos/(true_pos+false_pos)))
    print('RECALL : {0:.2f}'.format(true_pos/(false_neg+true_pos)), end='\n\n')
    
    print('Label 1 : {0}\nLabel 2 : {1}\n'.format(selected_label1, selected_label2))
else: print('Error : ', error)
    



