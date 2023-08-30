# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:06:42 2023

@author: torterma

Clean PamGuard  whistle and moan detector first detection of each wav file (might be very specific to Sylence data).
This is because the first detection on each wav files corresponds to the detection of a electronic buzz made by the recorder

"""

import pandas as pd
import pytz
from post_processing_detections.utilities.def_func import get_detection_files
import datetime as dt

# Read csv file
detection_file = get_detection_files(1)
df_detections = pd.read_csv(detection_file[0])
#df_detections['start_datetime'] = pd.to_datetime(df_detections['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z')
a=[]
a= pd.to_datetime(df_detections['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z')

tz_data = a[0].tz

# 1- Read the filename date of the detection
filename = df_detections['filename']
filename_d = [x[9:28] for x in filename]
date_file = [dt.datetime.strptime(x, "%Y-%m-%d_%H-%M-%S") for x in filename_d]
date_file = [ d.replace(tzinfo=tz_data) for d in date_file]
# 2 - Read detection date in start_datetime
start = [d.to_pydatetime() for d in a]

# 3- Compare date of filename detection and date of detection 
idx_FA = []
for i in range (0, len(start)):
    d = (start[i]-date_file[i]).total_seconds()
    if d < 5:
        idx_FA.append(i)
# 4 - Delete all lines for which the detection happens in the 5 first seconds of the file
print('result cleaned')
results_cleaned = df_detections.drop(labels=idx_FA, axis=0)

# Write the new UTC dataframe in csv

new_filename = detection_file[0][:-4] +'_clean.csv'
results_cleaned.to_csv(new_filename, index=None)
