# This script is used in order to compute the detection performances of an APLOSE formatted detection file
# the computed metrics are so far precision and recall
# it takes as an input 2 APLOSE formatted detection/annotation files
# the user has to select one of the 2 files as the reference/"ground truth" to calcultate the performance of the second file

import pandas as pd
import numpy as np
import easygui
import datetime as dt
from scipy import stats
from utilities.def_func import get_detection_files, sorting_detections, input_date,  t_rounder

# %% Load data - user inputs

# get the path of the 2 detections files
files_list = get_detection_files(2)


# choose which file is used as the reference or "ground truth"
choice_ref = easygui.buttonbox('Select the reference', '{0}'.format(files_list[0].split('/')[-1]), [file.split('/')[-1] for file in files_list])
if files_list[0].split('/')[-1] != choice_ref:
    files_list[0], files_list[1] = files_list[1], files_list[0]

# import detections, reference timebin, labels and annotators for each file
df_detections, t_detections = sorting_detections(files=files_list, timebin_new=10, user_sel='all')
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
    begin_date = pd.Timestamp('2022-07-17 00:25:46 +0200')
    end_date = pd.Timestamp('2022-07-18 00:27:49 +0200')

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

# creation of a time vector that goes from the start date to the end date with every timebin
time_vector = [i.timestamp() for i in pd.date_range(start=begin_date, end=end_date, freq=str(timebin_detections) + 's')]


# For each file, a dataframe is created, df1 (reference) and df2.
# For each dataframe, whithin each time_vector timestamp is checked if
# at least one timestamp of the detection file is present.
# A binary vector is then created for each df (vec1 and vec2),
# its length is the same of time_vector and is composed of 0 and 1
# corresponding to the presence/absence of detections at the corresponding datetime frame

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
    print('RECALL : {0:.2f}'.format(true_pos / (false_neg + true_pos)))

    # f-score : 2*(precision*recall)/(precision+recall)
    print('F-SCORE : {0:.2f}'.format(2 * ((true_pos / (true_pos + false_pos)) * (true_pos / (false_neg + true_pos))) / ((true_pos / (true_pos + false_pos)) + (true_pos / (false_neg + true_pos)))), end='\n\n')

    print('Label 1 : {0}\nLabel 2 : {1}\n'.format(selected_label1, selected_label2))
    print('Annotator 1 : {0}\nAnnotator 2 : {1}\n'.format(annotator1, annotator2))

else: print('Error : ', error)


#%% Compute Pearson corelation coefficient between the two subsets

df_detections1, _ = sorting_detections(files=files_list[0], timebin_new=10, user_sel='all', annotator = annotator1)
df_detections2, _ = sorting_detections(files=files_list[1], timebin_new=10, user_sel='all', annotator = annotator2)


annot_ref = annotator1
label_ref = selected_label1
time_bin_ref = int(t_detections[t_detections['annotators'].apply(lambda x: annot_ref in x)]['max_time'].iloc[0])
file_ref = t_detections[t_detections['annotators'].apply(lambda x: annot_ref in x)]['file']
tz_data = df_detections['start_datetime'][0].tz

# Ask user if their resolution_bin is in minutes or in months or in seasons
resolution_bin = easygui.buttonbox(msg='Do you want to chose your resolution bin in minutes or in months', choices=('Minutes', 'Days', 'Weeks', 'Months'))
if resolution_bin == 'Minutes':
    res_min = easygui.integerbox('Enter the bin size (min)', 'Time resolution', default=10, lowerbound=1, upperbound=86400)
    n_annot_max = (res_min * 60) / time_bin_ref  # max nb of annoted time_bin max per res_min slice
    # Est-ce que c'est utile de garder start_vec et end_vec sachant qu'ils sont égaux à begin_date et end_date non ?
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

# Compute histograms
hist1 = np.histogram(df_detections1['start_datetime'], bins=time_vector)
hist2 = np.histogram(df_detections2['start_datetime'], bins=time_vector)

# Compute the Pearson correlation coefficient
res = stats.pearsonr(hist1[0], hist2[0])

print('Pearson correlation coefficient : {0}\np-value : {1}\n'.format(res[0], res[1]))

























