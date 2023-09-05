# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:50:38 2023

@author: torterma
"""

import datetime as dt
import pandas as pd
import numpy as np
import easygui
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
import seaborn as sns
from scipy import stats
import sys
import pytz
from astral.sun import sun
import astral
from collections import OrderedDict
from post_processing_detections.utilities.def_func import get_detection_files, extract_datetime, sorting_detections, t_rounder, get_timestamps, input_date, suntime_hour


#%% Read and format detection file

files_list = get_detection_files(1)
df_detections, t_detections = sorting_detections(files_list, timebin_new=60)

time_bin = list(set(t_detections['max_time']))
fmax = list(set(t_detections['max_freq']))
annotators = list(set(t_detections['annotators'].explode()))
labels = list(set(t_detections['labels'].explode()))
tz_data = df_detections['start_datetime'][0].tz

dt_mode = 'fixed'


if dt_mode == 'fixed' :
    # if you work with wav names
    #begin_deploy = extract_datetime('335556632.220501000000.wav', tz_data)
    #end_deploy = extract_datetime('335556632.230228235959.wav', tz_data)
    # or if you work with a fixed date
    begin_deploy = dt.datetime(2022, 5, 1, 0, 0, 0, 0, tz_data)
    end_deploy = dt.datetime(2022, 8, 24, 0, 0, 0, 0, tz_data)
elif dt_mode == 'auto':
    timestamps_file = get_timestamps()
    wav_names = timestamps_file['filename']
    begin_deploy = extract_datetime(wav_names.iloc[0], tz_data)
    end_deploy = extract_datetime(wav_names.iloc[-1], tz_data)
elif dt_mode == 'input' :
    msg='Enter begin date of Figure'
    begin_deploy=input_date(msg, tz_data)
    msg='Enter end date of Figure'
    end_deploy=input_date(msg, tz_data)

print("\ntime_bin : ", str(time_bin), "s", end='')
print("\nfmax : ", str(fmax), "Hz", end='')
print('\nannotators :',str(annotators), end='')
print('\nlabels :', str(labels), end='\n')


#%%

# User input : gps coordinates in Decimal Degrees
title = "Coordinates en degree° minute' "
msg="Latitudes (N/S) and longitudes (E/W)"
fieldNames = ["Lat Decimal Degree", "Lon Decimal Degree "]
fieldValues = []  # we start with blanks for the values
fieldValues = easygui.multenterbox(msg,title, fieldNames)

# make sure that none of the fields was left blank
while 1:
  if fieldValues == None: break
  errmsg = ""
  for i in range(len(fieldNames)):
    if fieldValues[i].strip() == "":
      errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
  if errmsg == "": break # no problems found
  fieldValues = easygui.multpasswordbox(errmsg, title, fieldNames, fieldValues)
print("Reply was:", fieldValues) 

lat = fieldValues[0] 
lon = fieldValues[1] 
# Compute sunrise and sunet decimal hour at the dataset location
[_, _, dt_dusk, dt_dawn, dt_day, dt_night] = suntime_hour(begin_deploy, end_deploy, tz_data, lat,lon)

# List of days in the dataset
list_days = [dt.date(d.year, d.month, d.day) for d in dt_day]
# Compute dusk_duration, dawn_duration, day_duration, night_duration
dawn_duration = [b-a for a,b in zip(dt_dawn, dt_day)]
day_duration = [b-a for a,b in zip(dt_day, dt_night)]
dusk_duration = [b-a for a,b in zip(dt_night, dt_dusk)]
night_duration = [dt.timedelta(hours=24) - dawn - day - dusk for dawn, day, dusk in zip(dawn_duration, day_duration, dusk_duration)]

# Assign a light regime to each detection
# Count the number of detection per light regime
# 
