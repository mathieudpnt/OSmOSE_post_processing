import pandas as pd
import numpy as np
from utilities.def_func import extract_datetime
from csv import writer
import pytz
import os

file = r'C:\Users\dupontma2\Downloads\APOCADO_C5D1_ST7181_results.csv'
df = pd.read_csv(file, sep=',', parse_dates=['start_datetime', 'end_datetime']).sort_values('start_datetime').reset_index(drop=True)
df = df.dropna(subset=['annotation'])  # Drop the lines with only comments
list_annotators = list(df['annotator'].drop_duplicates())
max_freq = int(max(df['end_frequency']))
max_time = int(max(df['end_time']))
dataset_ID = df['dataset'][0]

df['start_datetime'] = [dt.tz_convert(pytz.FixedOffset(60)) for dt in df['start_datetime']]
df['end_datetime'] = [dt.tz_convert(pytz.FixedOffset(60)) for dt in df['start_datetime']]

tz = df['start_datetime'][0].tz


for annot in list_annotators:
    list_labels = list(df[df['annotator'] == annot]['annotation'].drop_duplicates())
    for label in list_labels:
        filename = list(df[df['annotator'] == annot]['filename'].drop_duplicates())
        for f in filename:
            test = df[(df['filename'] == f) & (df['annotation'] == label)]['is_box']
            if not any(test == 0) and len(test) > 0:
                start_datetime = extract_datetime(var=f, tz=tz)
                end_datetime = start_datetime + pd.Timedelta(max_time, unit='s')
                # start_datetime_str = start_datetime.strftime("%Y-%m-%dT%H:%M:%S.000") + '+01:00'
                # end_datetime_str = end_datetime.strftime("%Y-%m-%dT%H:%M:%S.000") + '+01:00'
                new_line = [dataset_ID, f, 0, max_time, 0, max_freq, label, annot, start_datetime, end_datetime, 0, np.nan, np.nan, np.nan]
                df.loc[len(df)] = new_line

df = df.sort_values('start_datetime').reset_index(drop=True)

df['start_datetime'] = [x.strftime("%Y-%m-%dT%H:%M:%S.000") + '+01:00' for x in df['start_datetime']]
df['end_datetime'] = [x.strftime("%Y-%m-%dT%H:%M:%S.000") + '+01:00' for x in df['end_datetime']]

file_out = os.path.join(os.path.dirname(file), os.path.basename(file).split('.csv')[0] + '_WD_added.csv')

df.to_csv(file_out, index=False)
