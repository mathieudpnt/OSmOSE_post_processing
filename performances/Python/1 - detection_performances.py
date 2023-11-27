'''
This script is used in order to compute the detection performances of an APLOSE formatted detection file
the computed metrics are precision and recall and F-score
it takes as an input 2 APLOSE formatted detection/annotation files and a corresponding parameters file
the user has to select one of the 2 files as the reference/"ground truth" to calculate the performance of the second file
'''

import pandas as pd
import numpy as np
import easygui
from scipy import stats
import os
from utilities.def_func import get_csv_file, sorting_detections, input_date, t_rounder, task_status_selection, read_param

# %% Load data - user inputs

# Load parameters from the YAML file
yaml_file_path = os.path.join(os.getcwd(), 'performances', 'Python', 'detection_performance_parameters.yaml')
parameters = read_param(file=yaml_file_path)


# import detections, reference timebin, labels and annotators for each file
df_detections, info = pd.DataFrame(), pd.DataFrame()
for args in parameters:
    df_detections_file, info_file = sorting_detections(**args)
    df_detections = pd.concat([df_detections, df_detections_file], ignore_index=True)
    info = pd.concat([info, info_file], ignore_index=True)

timebin_detections = list(set(info['max_time'].explode()))
annotators_detections = list(set(info['annotators'].explode()))
labels_detections = list(set(info['labels'].explode()))


# choose which file is used as the reference or "ground truth"
if len(parameters) == 2:
    choice_ref = easygui.buttonbox('Select the reference', '{0}'.format('Loading data'), [os.path.basename(parameters[i]['file']) for i in range(len(parameters))])
    if os.path.basename(parameters[0]['file']) != choice_ref:
        info.iloc[0], info.iloc[1] = info.iloc[1], info.iloc[0]
else: raise Exception("Passed sets of parameters different than 2")


# select only detections/annotations of certain annotators
# status_list = get_csv_file(1)
# df_detections = task_status_selection(files=status_list, df_detections=df_detections, user=['jbeesa', 'bcolon'])


# choose the date interval on which the performances will be computed
# if mode is input, a pop-up window ask the user the dates to work with
# if mode is fixed, the date interval is hard coded bu the user
mode = 'fixed'

if mode == 'input':
    begin_date = input_date('Enter begin datetime')
    end_date = input_date('Enter end datetime')
elif mode == 'fixed':
    begin_date = pd.Timestamp('2023-02-11 12:15:00 +0100')
    end_date = pd.Timestamp('2023-02-12 09:00:00 +0100')

# annotators
annotator1 = easygui.buttonbox('Select annotator 1 (reference)', 'file 1 : {0}'.format(os.path.basename(parameters[0]['file'])), info['annotators'][0]) if len(info['annotators'][0]) > 1 else info['annotators'][0][0]
annotator2 = easygui.buttonbox('Select annotator 2', 'file 2 : {0}'.format(os.path.basename(parameters[1]['file'])), info['annotators'][1]) if len(info['annotators'][1]) > 1 else info['annotators'][1][0]

# labels
labels1 = info.iloc[0]['labels']
labels2 = info.iloc[1]['labels']

# files list
files_list = [parameters[i]['file'] for i in range(len(parameters))]

# data recap
print('\n### Detections ###')
print('Timebin: {0}s'.format(timebin_detections))
print('Begin date: {0}'.format(begin_date))
print('End date: {0}'.format(end_date))

print('\n### Reference file ###')
print('detection file: {0}'.format(os.path.basename(parameters[0]['file'])))
print('labels: {0}'.format(labels1))
print('annotator: {0}'.format(annotator1))

print('\n### file 2 ###')
print('detection file: {0}'.format(os.path.basename(parameters[1]['file'])))
print('labels: {0}'.format(labels2))
print('annotator: {0}'.format(annotator2))

# %% FORMAT DATA
'''
For each file, a dataframe is created, df1 (reference) and df2.
For each dataframe, whithin each time_vector timestamp is checked if
at least one timestamp of the detection file is present.
A binary vector is then created for each df (vec1 and vec2),
its length is the same of time_vector and is composed of 0 and 1
corresponding to the presence/absence of detections at the corresponding datetime frame
'''

# creation of a time vector that goes from the start date to the end date with every timebin
time_vector = [i.timestamp() for i in pd.date_range(start=begin_date, end=end_date, freq=str(timebin_detections[0]) + 's')]

# df1 - REFERENCE
selected_label1 = easygui.buttonbox('Select a label', 'file 1 : {0}'.format(files_list[0].split('/')[-1]), labels1) if len(labels1) > 1 else labels1[0]
selected_annotations1 = df_detections[(df_detections['annotator'] == annotator1) & (df_detections['annotation'] == selected_label1) & (df_detections['start_datetime'] >= begin_date) & (df_detections['end_datetime'] <= end_date)]

times1_beg = sorted(list(set(x.timestamp() for x in selected_annotations1['start_datetime'])))
times1_end = sorted(list(set(y.timestamp() for y in selected_annotations1['end_datetime'])))

vec1, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
for i in range(len(times1_beg)):
    for j in range(k, len(time_vector) - 1):
        if int(times1_beg[i] * 1e7) in range(int(time_vector[j] * 1e7), int(time_vector[j + 1] * 1e7)) or int(times1_end[i] * 1e7) in range(int(time_vector[j] * 1e7), int(time_vector[j + 1] * 1e7)):
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
        if int(times2_beg[i] * 1e7) in range(int(time_vector[j] * 1e7), int(time_vector[j + 1] * 1e7)) or int(times2_end[i] * 1e7) in range(int(time_vector[j] * 1e7), int(time_vector[j + 1] * 1e7)):
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
    print('RECALL : {0:.2f}'.format(true_pos / (false_neg + true_pos)))

    # f-score : 2*(precision*recall)/(precision+recall)
    print('F-SCORE : {0:.2f}'.format(2 * ((true_pos / (true_pos + false_pos)) * (true_pos / (false_neg + true_pos))) / ((true_pos / (true_pos + false_pos)) + (true_pos / (false_neg + true_pos)))), end='\n\n')

    print('File 1 : {0}/{1}\nFile 2 : {2}/{3}\n'.format(annotator1, selected_label1, annotator2, selected_label2))

else: print('Error : ', error)


# %% Compute Pearson corelation coefficient between the two subsets

annot_ref = annotator1
label_ref = selected_label1

df_detections1 = df_detections[(df_detections['annotator'] == annotator1) & (df_detections['annotation'] == label_ref)]
df_detections2 = df_detections[(df_detections['annotator'] == annotator2) & (df_detections['annotation'] == label_ref)]

time_bin_ref = int(info[info['annotators'].apply(lambda x: annot_ref in x)]['max_time'].iloc[0])
file_ref = info[info['annotators'].apply(lambda x: annot_ref in x)]['file']
tz_data = df_detections['start_datetime'][0].tz

'''
Useless here or should we plot something ?

# Ask user if their resolution_bin is in minutes or in months or in seasons
resolution_bin = easygui.buttonbox(msg='Do you want to chose your resolution bin in minutes or in months', choices=('Minutes', 'Days', 'Weeks', 'Months'))
if resolution_bin == 'Minutes':
    res_min = easygui.integerbox('Enter the bin size (min)', 'Time resolution', default=10, lowerbound=1, upperbound=86400)
    n_annot_max = (res_min * 60) / time_bin_ref  # max nb of annoted time_bin max per res_min slice
    delta, start_vec, end_vec = dt.timedelta(seconds=60 * res_min), t_rounder(begin_date, res=600), t_rounder(end_date + dt.timedelta(seconds=time_bin_ref), res=600)
    time_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]
    y_label_txt = 'Number of detections\n({0} min)'.format(res_min)

elif resolution_bin == 'Days':
    time_vector_ts = pd.date_range(begin_date, end_date, freq='D', tz=tz_data)
    time_vector = [timestamp.date() for timestamp in time_vector_ts]
    n_annot_max = (24 * 60 * 60) / time_bin_ref
    y_label_txt = 'Number of detections per day'
elif resolution_bin == 'Weeks':
    time_vector_ts = pd.date_range(begin_date, end_date, freq='W-MON', tz=tz_data)
    time_vector = [timestamp.date() for timestamp in time_vector_ts]
    n_annot_max = (24 * 60 * 60 * 7) / time_bin_ref
    y_label_txt = 'Number of detections per week (starting every Monday)'
else:
    # Compute the time_vector for a monthly resolution
    time_vector_ts = pd.date_range(begin_date, end_date, freq='MS', tz=tz_data)
    time_vector = [timestamp.date() for timestamp in time_vector_ts]
    n_annot_max = (31 * 24 * 60 * 60) / time_bin_ref
    y_label_txt = 'Number of detections per month'
'''

# Compute histograms
hist1 = np.histogram(df_detections1['start_datetime'], bins=time_vector)
hist2 = np.histogram(df_detections2['start_datetime'], bins=time_vector)

# Compute the Pearson correlation coefficient
res = stats.pearsonr(hist1[0], hist2[0])

print('Pearson correlation coefficient : {0:.3f}\np-value : {1:.3e}\n'.format(res[0], res[1]))

























