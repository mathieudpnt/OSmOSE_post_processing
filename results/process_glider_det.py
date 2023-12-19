# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:21:07 2023

@author: torterma
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:07:20 2023

@author: torterma
"""

import datetime as dt
import pandas as pd
import numpy as np
import easygui
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from collections import Counter
import seaborn as sns
from scipy import stats
import sys
import pytz
from utilities.trajectoryFda import TrajectoryFda
import gpxpy
import time

from utilities.def_func import get_csv_file, sorting_detections, t_rounder, get_timestamps, input_date, suntime_hour
# %% User inputs

files_list = get_csv_file(1)

arguments_list = [
    {
        'file': files_list[0],
        #'timebin_new': 60,
        'tz': pytz.FixedOffset(120),
        #'fmin_filter': 10000
    },
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


#%% Import gpx

gpx_filename = 'C:/Users/torterma/Documents/Projets_GLIDER/TAAF/Delgost/deployment_3.gpx'
#gpx_filename = 'L:/acoustock/Bioacoustique/DATASETS/GLIDER/GLIDER SEA034/MISSION_46_DELGOST/ANALYSES/carto/output_glider3.gpx'
gpx_file = open(gpx_filename, 'r')

gpx = gpxpy.parse(gpx_file)

# Compute lists of lat lon and time
latitude = []
longitude = []
time_dt = []
depth=[]
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            latitude.append(point.latitude)
            longitude.append(point.longitude)
            time_dt.append(point.time)
            depth.append(point.elevation)
            

time_unix = [time.mktime(t.timetuple()) for t in time_dt]
depth = np.array([0.]*len(time_unix))
track_data = np.column_stack((np.array(time_unix),np.array(longitude), np.array(latitude), depth))


#%%
dict_mmsi={}
key_mmsi=dict_mmsi.keys()
# ix : index de la position
# row : ligne de la position (time, lat, lon, depth)
for ix,row in enumerate(track_data): 
    # on commence par chercher si le navire existe déjà dans le flux
    if 0 not in key_mmsi:
        dict_mmsi[0]=TrajectoryFda(0,0.001,3) 

    dict_mmsi[0].setNewData(row[0], row[2], row[1])


ts_min=time_unix[0]
ts_max=time_unix[-1]

# Create array with unix time of detections
time_det = df_detections['start_datetime']
time_det_unix = [time.mktime(t.timetuple()) for t in time_det]

res=[]
for ts in time_det_unix:
    if ts_min<ts <ts_max:
        lat,lon=dict_mmsi[0].getPosition(ts) 

        if len(lon)>0:
            res.append([ts,lon[0][0],lat[0][0]])
            
df_detections['longitude'] = [res[i][1] for i in list(range(0,len(res)))]
df_detections['latitude'] = [res[i][2] for i in list(range(0,len(res)))]


#%% Save the csv file in the same folder as the FPOD results csv file
df_detections.to_csv(files_list[0] + '_coordinates.csv', index=False)

















