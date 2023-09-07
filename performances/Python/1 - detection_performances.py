# This script is used in order to compute the detection performances of an APLOSE formatted detection file
# the computed metrics are so far precision and recall
# it takes as an input 2 APLOSE formatted detection/annotation files
# the user has to select one of the 2 files as the reference/"ground truth" to calcultate the performance of the second file

import pandas as pd
import numpy as np
import easygui
from post_processing_detections.utilities.def_func import get_detection_files, sorting_detections, input_date

# %% Load data - user inputs

# get the path of the 2 detections files
files_list = get_detection_files(2)


# choose which file is used as the reference or "ground truth"
choice_ref = easygui.buttonbox('Select the reference', '{0}'.format(files_list[0].split('/')[-1]), [file.split('/')[-1] for file in files_list])
if files_list[0].split('/')[-1] != choice_ref:
    files_list[0], files_list[1] = files_list[1], files_list[0]

# import detections, reference timebin, labels and annotators for each file
df_detections, t_detections = sorting_detections(files=files_list, timebin_new=60, user_sel='all')
timebin_detections = int(list(set(t_detections['max_time']))[0])
labels_detections = list(set(t_detections['labels'].explode()))
annotators_detections = list(set(t_detections['annotators'].explode()))

# choose the date interval on which the performances will be computed
# if mode is input, a pop-up window ask the user the dates to work with
# if mode is fixed, the date interval is hard coded bu the user
mode = 'fixed'

if mode == 'input':
    begin_date = input_date('Enter begin datetime')
    end_date = input_date('Enter end datetime')
elif mode == 'fixed':
    begin_date = pd.Timestamp('2022-07-07 00:00:00 +0200')
    end_date = pd.Timestamp('2022-07-08 00:00:00 +0200')

# annotators
annotator1 = easygui.buttonbox('Select annotator 1 (reference)', 'file 1 : {0}'.format(files_list[0].split('/')[-1]), t_detections['annotators'][0]) if len(t_detections['annotators'][0]) > 1 else t_detections['annotators'][0][0]
annotator2 = easygui.buttonbox('Select annotator 2', 'file 2 : {0}'.format(files_list[1].split('/')[-1]), t_detections['annotators'][1]) if len(t_detections['annotators'][1]) > 1 else t_detections['annotators'][1][0]

# labels
labels1 = t_detections.iloc[0]['labels']
labels2 = t_detections.iloc[1]['labels']

# data recap
print('\n### Detections ###')
print('Timebin: {0}s'.format(timebin_detections))
print('Begin date: {0}'.format(begin_date))
print('End date: {0}'.format(end_date))

print('\n### Reference file ###')
print('detection file: {0}'.format(files_list[0].split('/')[-1]))
print('labels: {0}'.format(labels1))
print('annotator: {0}'.format(annotator1))

print('\n### file 2 ###')
print('detection file: {0}'.format(files_list[1].split('/')[-1]))
print('labels: {0}'.format(labels2))
print('annotator: {0}'.format(annotator2))

# %% FORMAT DATA

time_vector = [i.timestamp() for i in pd.date_range(start=begin_date, end=end_date, freq=str(timebin_detections) + 's')]

# df1 - REFERENCE
selected_label1 = easygui.buttonbox('Select a label', 'file 1 : {0}'.format(files_list[0].split('/')[-1]), labels1) if len(labels1) > 1 else labels1[0]
selected_annotations1 = df_detections[(df_detections['annotator'] == annotator1) & (df_detections['annotation'] == selected_label1) & (df_detections['start_datetime'] >= begin_date) & (df_detections['end_datetime'] <= end_date)]

times1_beg = sorted(list(set(x.timestamp() for x in selected_annotations1['start_datetime'])))
times1_end = sorted(list(set(y.timestamp() for y in selected_annotations1['end_datetime'])))

vec1, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
for i in range(len(times1_beg)):
    for j in range(k, len(time_vector) - 1):
        if int(times1_beg[i] * 1000) in range(int(time_vector[j] * 1000), int(time_vector[j + 1] * 1000)) or int(times1_end[i] * 1000) in range(int(time_vector[j] * 1000), int(time_vector[j + 1] * 1000)):
            ranks.append(j)
            k = j
            break
        else:
            continue
ranks = sorted(list(set(ranks)))
vec1[np.isin(range(len(time_vector)), ranks)] = 1

# df2
selected_label2 = easygui.buttonbox('Select a label', '{0}'.format(files_list[1].split('/')[-1]), labels2) if len(labels2) > 1 else labels2[0]
selected_annotations2 = df_detections[(df_detections['annotator'] == annotator2) & (df_detections['annotation'] == selected_label2) & (df_detections['start_datetime'] >= begin_date) & (df_detections['end_datetime'] <= end_date)]

times2_beg = [i.timestamp() for i in selected_annotations2['start_datetime']]
times2_end = [i.timestamp() for i in selected_annotations2['end_datetime']]

vec2, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
for i in range(len(times2_beg)):
    for j in range(k, len(time_vector) - 1):
        if int(times2_beg[i] * 1000) in range(int(time_vector[j] * 1000), int(time_vector[j + 1] * 1000)) or int(times2_end[i] * 1000) in range(int(time_vector[j] * 1000), int(time_vector[j + 1] * 1000)):
            ranks.append(j)
            k = j
            break
        else:
            continue
ranks = sorted(list(set(ranks)))
vec2[np.isin(range(len(time_vector)), ranks)] = 1


# DETECTION PERFORMANCES
true_pos, false_pos, true_neg, false_neg, error = 0, 0, 0, 0, 0
for i in range(len(time_vector)):
    if vec1[i] == 0 and vec2[i] == 0:
        true_neg += 1
    elif vec1[i] == 1 and vec2[i] == 1:
        true_pos += 1
    elif vec1[i] == 0 and vec2[i] == 1:
        false_pos += 1
    elif vec1[i] == 1 and vec2[i] == 0:
        false_neg += 1
    else: error += 1

print('\n\n### Detection results ###', end='\n')
if error == 0:
    print('True positive : {0}'.format(true_pos))
    print('True negative : {0}'.format(true_neg))
    print('False positive : {0}'.format(false_pos))
    print('False negative : {0}'.format(false_neg))

    print('\nPRECISION : {0:.2f}'.format(true_pos / (true_pos + false_pos)))
    print('RECALL : {0:.2f}'.format(true_pos / (false_neg + true_pos)), end='\n\n')

    print('Label 1 : {0}\nLabel 2 : {1}\n'.format(selected_label1, selected_label2))
    print('Annotator 1 : {0}\nAnnotator 2 : {1}\n'.format(annotator1, annotator2))

else: print('Error : ', error)
