# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:50:38 2023

@author: torterma
"""

import datetime as dt
import pylab
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

files_list = get_detection_files(13)
df_detections, t_detections = sorting_detections(files_list, tz = pytz.UTC, timebin_new=10)

time_bin = list(set(t_detections['max_time']))
fmax = list(set(t_detections['max_freq']))
annotators = list(set(t_detections['annotators'].explode()))
labels = list(set(t_detections['labels'].explode()))
tz_data = df_detections['start_datetime'][0].tz

dt_mode = 'input'


if dt_mode == 'fixed' :
    # if you work with wav names
    #begin_deploy = extract_datetime('335556632.220501000000.wav', tz_data)
    #end_deploy = extract_datetime('335556632.230228235959.wav', tz_data)
    # or if you work with a fixed date
    begin_deploy = dt.datetime(2022, 5, 1, 0, 0, 0, 0, tz_data)
    end_deploy = dt.datetime(2023, 4, 24, 0, 0, 0, 0, tz_data)
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
# Seems to only work with UTC data ?
[_, _, dt_dusk, dt_dawn, dt_day, dt_night] = suntime_hour(begin_deploy, end_deploy, tz_data, lat,lon)

# List of days in the dataset
list_days = [dt.date(d.year, d.month, d.day) for d in dt_day]
# Compute dusk_duration, dawn_duration, day_duration, night_duration
dawn_duration = [b-a for a,b in zip(dt_dawn, dt_day)]
day_duration = [b-a for a,b in zip(dt_day, dt_night)]
dusk_duration = [b-a for a,b in zip(dt_night, dt_dusk)]
night_duration = [dt.timedelta(hours=24) - dawn - day - dusk for dawn, day, dusk in zip(dawn_duration, day_duration, dusk_duration)]
# Convert to decimal
dawn_duration_dec = [dawn_d.total_seconds()/3600 for dawn_d in dawn_duration]
day_duration_dec = [day_d.total_seconds()/3600 for day_d in day_duration]
dusk_duration_dec = [dusk_d.total_seconds()/3600 for dusk_d in dusk_duration]
night_duration_dec = [night_d.total_seconds()/3600 for night_d in night_duration]



# Assign a light regime to each detection
# : 1 = night ; 2 = dawn ; 3 = day ; 4 = dusk
day_det = [start_datetime.date() for start_datetime in df_detections['start_datetime']]
light_regime = []
for idx_day, day in enumerate(list_days):
    for idx_det, d in enumerate(day_det):
        # If the detection occured during 'day'
        if d == day :
            if df_detections['start_datetime'][idx_det] > dt_dawn[idx_day] and df_detections['start_datetime'][idx_det] < dt_day[idx_day] :
                l=2
                light_regime.append(l)
            elif df_detections['start_datetime'][idx_det] > dt_day[idx_day] and df_detections['start_datetime'][idx_det] < dt_night[idx_day] :
                l=3
                light_regime.append(l)
            elif df_detections['start_datetime'][idx_det] > dt_night[idx_day] and df_detections['start_datetime'][idx_det] < dt_dusk[idx_day] :
                l=4
                light_regime.append(l)
            else:
                l=1
                light_regime.append(l)
        
                 

# For each day, count the number of detection per light regime
nb_det_night = []
nb_det_dawn = []
nb_det_day = []
nb_det_dusk = []
for idx_day, day in enumerate(list_days) :
    # Find index of detections that occured during 'day'
    idx_det = [idx for idx, det in enumerate(day_det) if det == day]
    if idx_det == []:
        l=0
        nb_det_night.append(l)
        nb_det_dawn.append(l)
        nb_det_day.append(l)
        nb_det_dusk.append(l)
    else :
        nb_det_night.append(light_regime[idx_det[0]:idx_det[-1]].count(1))
        nb_det_dawn.append(light_regime[idx_det[0]:idx_det[-1]].count(2))
        nb_det_day.append(light_regime[idx_det[0]:idx_det[-1]].count(3))
        nb_det_dusk.append(light_regime[idx_det[0]:idx_det[-1]].count(4))
    
# For each day :  compute number of detection per light regime corrected by ligh regime duration
 
nb_det_night_corr = [(nb/d) for nb,d in zip(nb_det_night, night_duration_dec)]    
nb_det_dawn_corr = [(nb/d) for nb,d in zip(nb_det_dawn, dawn_duration_dec)]   
nb_det_day_corr = [(nb/d) for nb,d in zip(nb_det_day, day_duration_dec)]   
nb_det_dusk_corr = [(nb/d) for nb,d in zip(nb_det_dusk, dusk_duration_dec)]      

# Normalize by daily average number of detection per hour
av_daily_nbdet = []
nb_det_night_corr_norm = []
nb_det_dawn_corr_norm = []
nb_det_day_corr_norm = []
nb_det_dusk_corr_norm = []

for idx_day, day in enumerate(list_days) :
    # Find index of detections that occured during 'day'
    idx_det = [idx for idx, det in enumerate(day_det) if det == day]
    # Compute daily average number of detections per hour
    a = len(idx_det)/24
    av_daily_nbdet.append(a)
    if a == 0:
        nb_det_night_corr_norm.append(0)
        nb_det_dawn_corr_norm.append(0)
        nb_det_day_corr_norm.append(0)
        nb_det_dusk_corr_norm.append(0)
    else : 
        nb_det_night_corr_norm.append(nb_det_night_corr[idx_day]-a)
        nb_det_dawn_corr_norm.append(nb_det_dawn_corr[idx_day]-a)
        nb_det_day_corr_norm.append(nb_det_day_corr[idx_day]-a)
        nb_det_dusk_corr_norm.append(nb_det_dusk_corr[idx_day]-a)
        

LIGHTR = [nb_det_night_corr_norm, nb_det_dawn_corr_norm, nb_det_day_corr_norm, nb_det_dusk_corr_norm]
BoxName = ['Night', 'Dawn', 'Day', 'Dusk']

fig, ax = plt.subplots()
ax.boxplot(LIGHTR, showfliers=False) 
plt.ylim(-20,20)
pylab.xticks([1,2,3,4], BoxName)

#%%
def diel_plot(df_detections, begin_deploy, end_deploy, tz_data, lat, lon):
    
    if lat is None or lon is None:
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
    # Seems to only work with UTC data ?
    [_, _, dt_dusk, dt_dawn, dt_day, dt_night] = suntime_hour(begin_deploy, end_deploy, tz_data, lat,lon)
    
    # List of days in the dataset
    list_days = [dt.date(d.year, d.month, d.day) for d in dt_day]
    # Compute dusk_duration, dawn_duration, day_duration, night_duration
    dawn_duration = [b-a for a,b in zip(dt_dawn, dt_day)]
    day_duration = [b-a for a,b in zip(dt_day, dt_night)]
    dusk_duration = [b-a for a,b in zip(dt_night, dt_dusk)]
    night_duration = [dt.timedelta(hours=24) - dawn - day - dusk for dawn, day, dusk in zip(dawn_duration, day_duration, dusk_duration)]
    # Convert to decimal
    dawn_duration_dec = [dawn_d.total_seconds()/3600 for dawn_d in dawn_duration]
    day_duration_dec = [day_d.total_seconds()/3600 for day_d in day_duration]
    dusk_duration_dec = [dusk_d.total_seconds()/3600 for dusk_d in dusk_duration]
    night_duration_dec = [night_d.total_seconds()/3600 for night_d in night_duration]
    
    
    
    # Assign a light regime to each detection
    # : 1 = night ; 2 = dawn ; 3 = day ; 4 = dusk
    day_det = [start_datetime.date() for start_datetime in df_detections['start_datetime']]
    light_regime = []
    for idx_day, day in enumerate(list_days):
        for idx_det, d in enumerate(day_det):
            # If the detection occured during 'day'
            if d == day :
                if df_detections['start_datetime'][idx_det] > dt_dawn[idx_day] and df_detections['start_datetime'][idx_det] < dt_day[idx_day] :
                    l=2
                    light_regime.append(l)
                elif df_detections['start_datetime'][idx_det] > dt_day[idx_day] and df_detections['start_datetime'][idx_det] < dt_night[idx_day] :
                    l=3
                    light_regime.append(l)
                elif df_detections['start_datetime'][idx_det] > dt_night[idx_day] and df_detections['start_datetime'][idx_det] < dt_dusk[idx_day] :
                    l=4
                    light_regime.append(l)
                else:
                    l=1
                    light_regime.append(l)
            
                     
    
    # For each day, count the number of detection per light regime
    nb_det_night = []
    nb_det_dawn = []
    nb_det_day = []
    nb_det_dusk = []
    for idx_day, day in enumerate(list_days) :
        # Find index of detections that occured during 'day'
        idx_det = [idx for idx, det in enumerate(day_det) if det == day]
        if idx_det == []:
            l=0
            nb_det_night.append(l)
            nb_det_dawn.append(l)
            nb_det_day.append(l)
            nb_det_dusk.append(l)
        else :
            nb_det_night.append(light_regime[idx_det[0]:idx_det[-1]].count(1))
            nb_det_dawn.append(light_regime[idx_det[0]:idx_det[-1]].count(2))
            nb_det_day.append(light_regime[idx_det[0]:idx_det[-1]].count(3))
            nb_det_dusk.append(light_regime[idx_det[0]:idx_det[-1]].count(4))
        
    # For each day :  compute number of detection per light regime corrected by ligh regime duration
     
    nb_det_night_corr = [(nb/d) for nb,d in zip(nb_det_night, night_duration_dec)]    
    nb_det_dawn_corr = [(nb/d) for nb,d in zip(nb_det_dawn, dawn_duration_dec)]   
    nb_det_day_corr = [(nb/d) for nb,d in zip(nb_det_day, day_duration_dec)]   
    nb_det_dusk_corr = [(nb/d) for nb,d in zip(nb_det_dusk, dusk_duration_dec)]      
    
    # Normalize by daily average number of detection per hour
    av_daily_nbdet = []
    nb_det_night_corr_norm = []
    nb_det_dawn_corr_norm = []
    nb_det_day_corr_norm = []
    nb_det_dusk_corr_norm = []
    
    for idx_day, day in enumerate(list_days) :
        # Find index of detections that occured during 'day'
        idx_det = [idx for idx, det in enumerate(day_det) if det == day]
        # Compute daily average number of detections per hour
        a = len(idx_det)/24
        av_daily_nbdet.append(a)
        if a == 0:
            nb_det_night_corr_norm.append(0)
            nb_det_dawn_corr_norm.append(0)
            nb_det_day_corr_norm.append(0)
            nb_det_dusk_corr_norm.append(0)
        else : 
            nb_det_night_corr_norm.append(nb_det_night_corr[idx_day]-a)
            nb_det_dawn_corr_norm.append(nb_det_dawn_corr[idx_day]-a)
            nb_det_day_corr_norm.append(nb_det_day_corr[idx_day]-a)
            nb_det_dusk_corr_norm.append(nb_det_dusk_corr[idx_day]-a)
            
    
    LIGHTR = [nb_det_night_corr_norm, nb_det_dawn_corr_norm, nb_det_day_corr_norm, nb_det_dusk_corr_norm]
    BoxName = ['Night', 'Dawn', 'Day', 'Dusk']
    
    fig, ax = plt.subplots()
    ax.boxplot(LIGHTR, showfliers=False) 
    plt.ylim(-20,20)
    pylab.xticks([1,2,3,4], BoxName)
    
    return

diel_plot(df_detections, begin_deploy, end_deploy, tz_data, lat=48.165, lon=-4.419)
