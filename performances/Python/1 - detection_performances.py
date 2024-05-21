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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cycler import cycler
import pytz
import warnings

# os.chdir(r'U:/Documents_U/Git/post_processing_detections')
os.chdir(r'C:\Users\dupontma2\Desktop\data_local\post_processing_detections-main_17052024')
from utilities.def_func import get_csv_file, sorting_detections, input_date, t_rounder, task_status_selection, read_param, t_rounder

mpl.style.use('seaborn-v0_8-paper')
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams["axes.prop_cycle"] = cycler('color', ['#4590d3', 'darkorange', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

# %% Load data - user inputs

# Load parameters from the YAML file
yaml_file_path = os.path.join(os.getcwd(), 'performances', 'Python', 'detection_performance_parameters.yaml')
parameters = read_param(file=yaml_file_path)

# import detections, reference timebin, labels and annotators for each file
df_detections, info = pd.DataFrame(), pd.DataFrame()
for param in parameters:
    df_detections_file, info_file = sorting_detections(**param)
    df_detections = pd.concat([df_detections, df_detections_file], ignore_index=True)
    info = pd.concat([info, info_file], ignore_index=True)

annotators_detections = info['annotators'].explode().unique()[0] if len(info['annotators'].explode().unique()) == 1 else list(set(info['annotators'].explode()))
labels_detections = info['labels'].explode().unique()[0] if len(info['labels'].explode().unique()) == 1 else list(set(info['labels'].explode()))
if len(info['max_time'].explode().unique()) == 1:
    timebin_detections = int(info['max_time'].unique()[0])
else: raise Exception("More than one timebin passed")
if len(info['timestamp_file'].explode().unique()) == 1:
    timestamp_file = info['timestamp_file'].explode().unique()[0]
else: raise Exception("More than one timestamp file passed")
if len(info['tz_data'].explode().unique()) == 1:
    tz_data = info['tz_data'].explode().unique()[0]
else:
    tz_data = pytz.FixedOffset(0)
    warnings.warn("More than one timezone passed, UTC offset is chosen by default")

# choose which file is used as the reference or "ground truth"
if len(parameters) == 2:
    choice_ref = easygui.buttonbox('Select the reference', '{0}'.format('Loading data'), [os.path.basename(parameters[i]['file']) for i in range(len(parameters))])
    if os.path.basename(parameters[0]['file']) != choice_ref:
        info.iloc[0], info.iloc[1] = info.iloc[1], info.iloc[0]
else: raise Exception("Passed sets of parameters different than 2")

'''
choose the date interval on which the performances will be computed
if mode is 'timestamps', begin and end date are selected from the timestamp file
if mode is 'custom', the date interval is hard coded by the user
'''
mode = 'custom'

if mode == 'timestamps':
    begin_date = pd.read_csv(timestamp_file, parse_dates=['timestamp'])['timestamp'].iloc[0]
    end_date = pd.read_csv(timestamp_file, parse_dates=['timestamp'])['timestamp'].iloc[-1] + pd.Timedelta(timebin_detections, unit='second')
elif mode == 'custom':
    # begin_date = pd.Timestamp('2022-07-06 23:59:47 +0200')
    # end_date = pd.Timestamp('2022-07-08 01:59:28 +0200')
    begin_date = pd.Timestamp('2022-07-07 09:00:00 +0200')
    end_date = pd.Timestamp('2022-07-08 00:00:00 +0200')
    # begin_date = pd.Timestamp('2023-02-11 12:00:00 +0100')
    # end_date = pd.Timestamp('2023-02-12 00:00:00 +0100')

# annotators
annotators1 = info.iloc[0]['annotators']
annotators2 = info.iloc[1]['annotators']

# labels
labels1 = info.iloc[0]['labels']
labels2 = info.iloc[1]['labels']

# files list
files_list = [parameters[i]['file'] for i in range(len(parameters))]

# data recap
print('\n### Detections ###')
print(f'Timebin: {timebin_detections}s')
print(f'Begin date: {begin_date}')
print(f'End date: {end_date}')

print('\n### Reference file ###')
print(f'detection file: {os.path.basename(parameters[0]["file"])}')
print(f'labels: {labels1}')
print(f'annotator: {annotators1}')

print('\n### file 2 ###')
print(f'detection file: {os.path.basename(parameters[1]["file"])}')
print(f'labels: {labels2}')
print(f'annotator: {annotators2}')

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
'''
timestamp_csv = pd.read_csv(timestamp_file, parse_dates=['timestamp'])
timestamp_csv2 = timestamp_csv[0::6]
timestamp_range = timestamp_csv2[(timestamp_csv2['timestamp'] >= begin_date) & (timestamp_csv2['timestamp'] <= end_date)]['timestamp'].to_list()
# timestamp_range = timestamp_csv[(timestamp_csv['timestamp'] >= begin_date) & (timestamp_csv['timestamp'] <= end_date)]['timestamp'].to_list()
timestamp_range.append(timestamp_range[-1] + pd.Timedelta(timebin_detections, unit='second'))

# time_vector = [i.timestamp() for i in pd.date_range(start=begin_date, end=end_date, freq=str(timebin_detections) + 's')]
time_vector = [ts.timestamp() for ts in timestamp_range]
'''
timestamp_csv = pd.read_csv(timestamp_file, parse_dates=['timestamp'])
timestamp_range = timestamp_csv['timestamp'].to_list()
t = t_rounder(t=max(timestamp_range[0], begin_date), res=timebin_detections)
t2 = t_rounder(min(timestamp_range[-1], end_date), res=timebin_detections) + pd.Timedelta(seconds=timebin_detections)
time_vector = [ts.timestamp() for ts in pd.date_range(start=t, end=t2, freq=str(timebin_detections) + 's')]


# df1 - REFERENCE
selected_annotator1 = easygui.buttonbox('Select annotator 1 (reference)', 'file 1 : {0}'.format(files_list[0].split('/')[-1]), annotators1) if len(annotators1) > 1 else annotators1[0]
selected_label1 = easygui.buttonbox('Select a label', 'file 1 : {0}'.format(files_list[0].split('/')[-1]), labels1) if len(labels1) > 1 else labels1[0]
selected_annotations1 = df_detections[(df_detections['annotator'] == selected_annotator1) & (df_detections['annotation'] == selected_label1) & (df_detections['start_datetime'] >= begin_date) & (df_detections['end_datetime'] <= end_date)]

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
selected_annotator2 = easygui.buttonbox('Select annotator 1 (reference)', 'file 1 : {0}'.format(files_list[0].split('/')[-1]), annotators2) if len(annotators2) > 1 else annotators2[0]
selected_label2 = easygui.buttonbox('Select a label', '{0}'.format(files_list[1].split('/')[-1]), labels2) if len(labels2) > 1 else labels2[0]
selected_annotations2 = df_detections[(df_detections['annotator'] == selected_annotator2) & (df_detections['annotation'] == selected_label2) & (df_detections['start_datetime'] >= begin_date) & (df_detections['end_datetime'] <= end_date)]

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
    print(f'True positive : {true_pos}')
    print(f'True negative : {true_neg}')
    print(f'False positive : {false_pos}')
    print(f'False negative : {false_neg}')

    print(f'\nPRECISION : {true_pos / (true_pos + false_pos):.2f}')
    print(f'RECALL : {true_pos / (false_neg + true_pos):.2f}')

    # f-score : 2*(precision*recall)/(precision+recall)
    f_score = 2 * ((true_pos / (true_pos + false_pos)) * (true_pos / (false_neg + true_pos))) / ((true_pos / (true_pos + false_pos)) + (true_pos / (false_neg + true_pos)))
    print(f'F-SCORE : {f_score:.2f}')

    print(f'File 1 : {selected_annotator1}/{selected_label1} \nFile 2 : {selected_annotator2}/{selected_label2}')

else: print(f'Error : {error}')

# plot
res_min = easygui.integerbox('Enter the bin size (min) ', 'Time resolution', default=10, lowerbound=1, upperbound=86400)
delta, start_vec, end_vec = pd.Timedelta(seconds=60 * res_min), t_rounder(begin_date, res=600), t_rounder(end_date + pd.Timedelta(seconds=timebin_detections), res=600)
bin_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]
n_annot_max = (res_min * 60) / timebin_detections  # max nb of annoted time_bin max per res_min slice

fig, ax = plt.subplots(1, 1, dpi=200, figsize=(10, 4))

hist_plot = ax.hist([selected_annotations1['start_datetime'], selected_annotations2['start_datetime']], bins=bin_vector, label=[selected_label1, selected_label2], edgecolor='black')
ax.legend(loc='upper right')

bars = range(0, 101, 10)  # from 0 to 100 step 10
y_pos = np.linspace(0, n_annot_max, num=len(bars))
ax.set_yticks(y_pos, bars)
ax.tick_params(axis='x', rotation=60)
ax.set_ylabel('positive detection rate\n({0} min)'.format(res_min))
ax.tick_params(axis='y')
fig.suptitle('[{0}/{1}] VS [{2}/{3}]'.format(selected_annotator1, selected_label1, selected_annotator2, selected_label2), y=1.02)

ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz_data))
ax.set_xlim(bin_vector[0], bin_vector[-1])
ax.grid(linestyle='-', linewidth=0.2, axis='both')

plt.tight_layout()
plt.show()
























