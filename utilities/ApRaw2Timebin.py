import os
import pandas as pd
import pytz
from utilities.def_func import get_csv_file, sorting_detections

#%% LOAD DATA - User inputs

files_list = get_csv_file(1)
arguments_list = [
    {
        'file': files_list[0],
        'timebin_new': 60,
        'tz': pytz.FixedOffset(0),
        #'fmin_filter': 10000
    },
    # {
    #     'file': files_list[1],
    #     'timebin_new': 60,
    #     'tz': pytz.FixedOffset(120),
    #     #'fmin_filter': 10000
    # },
   ] 
    
df_detections, info = pd.DataFrame(), pd.DataFrame()
for args in arguments_list:
    df_detections_file, info_file = sorting_detections(**args)
    df_detections = pd.concat([df_detections, df_detections_file], ignore_index=True)
    info = pd.concat([info, info_file], ignore_index=True)


time_bin = list(set(info['max_time'].explode()))
fmax = list(set(info['max_freq'].explode()))
annotators = list(set(info['annotators'].explode()))
labels = list(set(info['labels'].explode()))
tz_data = list(set(info['tz_data'].explode()))
if len(tz_data) == 1:
    [tz_data] = tz_data
else:
    raise Exception('More than one timezone in the detections')

f = os.path.splitext(files_list[0])[0]
new_fn = f + '_' + str(arguments_list[0]['timebin_new']) + 's.csv'

df_detections.to_csv(new_fn,index=False, sep=',')
