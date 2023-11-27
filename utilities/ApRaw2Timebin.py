import os
import pytz
from utilities.def_func import get_csv_file, sorting_detections, t_rounder

# LOAD DATA - User inputs

file = get_csv_file(1)

parameters = {'file': file[0],
              'timebin_new': 10,
              'annotator': 'mdupon',
              'annotation': 'Odontocete whistle',
              'tz': pytz.FixedOffset(60)}

df_detections, t_detections = sorting_detections(**parameters)


# EXPORT RESHAPPED DETECTIONS
dataset_name = '/APOCADO_C2D1_07072022_results'
# PG2Ap_str = dataset_name + t_rounder(df_detections['start_datetime'][0], res=600).strftime('%y%m%d') + '_' + t_rounder(df_detections['start_datetime'].iloc[-1], res=600).strftime('%y%m%d') + '_' + str(t_detections['max_time'][0]) + 's' + '.csv'
PG2Ap_str = dataset_name + '_' + str(t_detections['max_time'][0]) + 's' + '.csv'
df_detections.to_csv(os.path.dirname(file[0]) + PG2Ap_str, index=False)
print('\n\nAplose formatted data file exported to ' + os.path.dirname(file[0]))

# %%
import os
import pandas as pd
import pytz
import numpy as np
from utilities.def_func import get_csv_file, sorting_detections, t_rounder, extract_datetime
import bisect


file = get_csv_file(2)
tz_data = pytz.FixedOffset(60)
tb = 10
tb_new = 3
f = str(tb_new) + 's'

task_status = pd.read_csv(file[1])
t = extract_datetime(var=task_status['filename'].iloc[0], tz=tz_data)
t2 = extract_datetime(var=task_status['filename'].iloc[-1], tz=tz_data) + pd.to_timedelta(tb, unit='s')



parameters = {'file': file[0],
              'box': True,
              'annotator': 'mdupon',
              'annotation': 'Odontocete whistle',
              'tz': pytz.FixedOffset(60)}

df_detections, _ = sorting_detections(**parameters)
df_no_box = df_detections[df_detections['is_box'] == 1]




tv = pd.date_range(start=t, end=t2, freq=f)
time_vector = [ts.timestamp() for ts in pd.date_range(start=t, end=t2, freq=f)]

# #here test to find for each time vector value which filename corresponds
filenames = sorted(list(set(df_no_box['filename'])))
ts_filenames = [extract_datetime(filename, tz=tz_data).timestamp()for filename in filenames]

filename_vector = []
for ts in time_vector:
    index = bisect.bisect_left(ts_filenames, ts)
    if index == 0:
        # filename_vector.append(filenames[index])
        filename_vector.append(0)
    else:
        filename_vector.append(filenames[index - 1])


times_detect_beg = [detect.timestamp() for detect in df_no_box['start_datetime']]
times_detect_end = [detect.timestamp() for detect in df_no_box['end_datetime']]

detect_vec, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
for i in range(len(times_detect_beg)):
    for j in range(k, len(time_vector) - 1):
        if int(times_detect_beg[i] * 1e7) in range(int(time_vector[j] * 1e7), int(time_vector[j + 1] * 1e7)) or int(times_detect_end[i] * 1e7) in range(int(time_vector[j] * 1e7), int(time_vector[j + 1] * 1e7)):
            ranks.append(j)
            k = j
            break
        else:
            continue

ranks = sorted(list(set(ranks)))
detect_vec[ranks] = 1
detect_vec = list(detect_vec)


start_datetime_str, end_datetime_str, filename = [], [], []
for i in range(len(time_vector)):
    if detect_vec[i] == 1:
        start_datetime = pd.Timestamp(time_vector[i], unit='s', tz=tz_data)
        start_datetime_str.append(start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[:-8] + start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-5:-2] + ':' + start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-2:])
        end_datetime = pd.Timestamp(time_vector[i] + tb_new, unit='s', tz=tz_data)
        end_datetime_str.append(end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[:-8] + end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-5:-2] + ':' + end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-2:])
        filename.append(task_status['filename'].iloc[i].split('.wav')[0])

df_new_prov = pd.DataFrame()
dataset_str = list(set(df_detect_prov['dataset']))

df_new_prov['dataset'] = dataset_str * len(start_datetime_str)
df_new_prov['filename'] = filename
df_new_prov['start_time'] = [0] * len(start_datetime_str)
df_new_prov['end_time'] = [timebin_new] * len(start_datetime_str)
df_new_prov['start_frequency'] = [0] * len(start_datetime_str)
df_new_prov['end_frequency'] = [max_freq] * len(start_datetime_str)
df_new_prov['annotation'] = list(set(df_detect_prov['annotation'])) * len(start_datetime_str)
df_new_prov['annotator'] = list(set(df_detect_prov['annotator'])) * len(start_datetime_str)
df_new_prov['start_datetime'], df_new_prov['end_datetime'] = start_datetime_str, end_datetime_str

df_new = pd.concat([df_new, df_new_prov])

df_new['start_datetime'] = [pd.to_datetime(d, format='%Y-%m-%dT%H:%M:%S.%f%z') for d in df_new['start_datetime']]
# df_new['start_datetime'] = pd.to_datetime(df_new['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z')
df_new['end_datetime'] = [pd.to_datetime(d, format='%Y-%m-%dT%H:%M:%S.%f%z') for d in df_new['end_datetime']]
# df_new['end_datetime'] = pd.to_datetime(df_new['end_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z')
df_new = df_new.sort_values(by=['start_datetime'])



















