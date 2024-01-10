# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:09:20 2023

@author: torterma
"""

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
from utilities.trajectoryFda import TrajectoryFda
import gpxpy
from tkinter import filedialog
import time
from tkinter import Tk

from utilities.def_func import get_csv_file, sorting_detections



def get_gpx(num_files):
    root = Tk()
    root.withdraw()

    file_paths = []
    for _ in range(num_files):
        file_path = filedialog.askopenfilename(
            title=f'Select gpx ({len(file_paths) + 1}/{num_files})',
            filetypes=[('GPX files', '*.gpx')],
            parent=None
        )
        if not file_path:
            break  # User cancelled or closed the file dialog
        file_paths.append(file_path)
    return file_paths

def get_track_data(gpx_path):
    gpx_file = open(gpx_path, 'r')
    gpx = gpxpy.parse(gpx_file)

    # Compute lists of lat lon depth and time
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
    track_data = np.column_stack((np.array(time_unix),np.array(longitude), np.array(latitude), depth))
    
    return track_data


def compute_loc_from_time(track_data, time_unix):
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
    res=[]
    for ts in time_unix:
        if ts_min<=ts <=ts_max:
            lat,lon=dict_mmsi[0].getPosition(ts) 

            if len(lon)>0:
                res.append([ts,lon[0][0],lat[0][0]])
                
    return res






#%%┴Figure 'planning'
#Select all csv files with detections/annotations that will appear in the following Figures
files_list = get_csv_file(1)

arguments_list = [
    {
        'file': files_list[0],
        #'timebin_new': 60,
        'tz': pytz.FixedOffset(120),
        #'fmin_filter': 10000
    },
    {
        'file': files_list[1],
        #'timebin_new': 60,
        'tz': pytz.FixedOffset(120),
        #'fmin_filter': 10000
    },
    {
        'file': files_list[2],
        #'timebin_new': 60,
        'tz': pytz.FixedOffset(120),
        #'fmin_filter': 10000
    },
    {
        'file': files_list[3],
        #'timebin_new': 60,
        'tz': pytz.FixedOffset(120),
        #'fmin_filter': 10000
    },
    {
        'file': files_list[4],
        #'timebin_new': 60,
        'tz': pytz.FixedOffset(120),
        #'fmin_filter': 10000
    },
    {
        'file': files_list[5],
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


det = df_detections.drop_duplicates(subset=['annotation', 'start_datetime'])

list_color = { 'label' : ['Odontocete whistles', 'Sperm whale clics','Odontocete clics','UnidentifiedCalls','Odontocete buzz','Blackfish whistles', 'Fin whale 40 Hz', 'Fin whale 20 Hz'], 'color' : ['#7fd779', '#e8718d', '#e77148', '#1c4a64', '#b7484b', '#72450a', 'black', 'gray']}

list_labels = list(det['annotation'].unique())

df_color = pd.DataFrame(data=list_color)
fig, ax = plt.subplots(figsize=(20,8))
for i, label in enumerate(list_labels):
    det_label = df_detections[(df_detections['annotation'] == label)]
    time_det = det_label['start_datetime']
    time_det_unix = [time.mktime(t.timetuple()) for t in time_det]  
    mpl_time_det = mdates.epoch2num(time_det_unix)


    l_data = len(mpl_time_det)
    x=np.ones((l_data,1), int)*i
    c=df_color.loc[df_color['label'] == label,'color'].values[0]
    
    plt.scatter(mpl_time_det,x, s=38, color = c)
    print(i)
    print(label)
    
    #locator = mdates.HourLocator(interval=24)
    #formatter = mdates.DateFormatter('%d/%m - %H:%M')
    locator = mdates.DayLocator(interval=1)
    formatter = mdates.DateFormatter('%d-%m')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    plt.grid(color='k', linestyle='-', linewidth=0.2)

    #ticks labels
    #plt.yticks(np.arange(0, 6, 1.0))
    plt.ylim(-0.5, len(list_labels) - 0.5) 
    ax.set_yticks(np.arange(0, len(list_labels), 1.0))
    ax.set_yticklabels(list_labels)
    
    ax.set_ylabel('Label', fontsize=25)
    ax.set_xlabel('Jour', fontsize=25)
    # ax.tick_params(labelsize=20)
    # plt.xlabel('Date (dd.mm)', fontsize=22)


#%% Compute acoustic diversity
# filename_audioF = 'C:/Users/torterma/Documents/Projets_GLIDER/Delgost/DELGOST2_D2 HF_task_status.csv'
filename_audioF = get_csv_file(2, 'Select task status csv')
# Download task results
for i, f in filename_audioF:
    
    l = pd.read_csv(f, delimiter=',')
    if i==0:
        list_audioF = l
    else : list_audioF = pd.concat([list_audioF, l])
# Put a coordinate on each audio file
# Download track data
gpx_paths = get_gpx(2)


for i,p in enumerate(gpx_paths):
    td = get_track_data(p)
    if i==0:
        track_data = td
    else : track_data = np.concatenate((track_data,td), axis = 0)
# Sort track data     
track_data = track_data[np.argsort(track_data[:, 0])]

# Create array with unix time of files
name_audioF = list_audioF['filename']
audioF_dt = [dt.datetime.strptime(t, "%Y_%m_%d_%H_%M_%S.wav") for t in name_audioF]
time_unix = [time.mktime(t.timetuple()) for t in audioF_dt]

# Compute localisation of each audio file            
res=compute_loc_from_time(track_data, time_unix)

#%% Read detections
files_list = get_csv_file(2)

arguments_list = [
    {
        'file': files_list[0],
        #'timebin_new': 60,
        'tz': pytz.FixedOffset(120),
        #'fmin_filter': 10000
    },
    {
        'file': files_list[1],
        #'timebin_new': 60,
        'tz': pytz.FixedOffset(120),
        #'fmin_filter': 10000
    }]

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

det = df_detections.drop_duplicates(subset=['annotation', 'start_datetime'])
# DElete unkown detections
det.drop(det[det['annotation'] == 'UnidentifiedCalls'].index, inplace = True)

# Create array with unix time of detections
time_det = det['start_datetime']
time_det_unix = [time.mktime(t.timetuple()) for t in time_det]


AD = np.zeros(len(time_unix))
list_det = []

for i, file in enumerate(time_unix):
    for d in time_det_unix:
        if d == file:
            AD[i]+=1
            
     
    list_det.append([file,res[i][1], res[i][2],AD[i]])
            
            
np.savetxt('C:/Users/torterma/Documents/Projets_GLIDER/TAAF/Delgost/Acoustic_Diversity_D2.csv', [p for p in list_det], delimiter=',', fmt = '%i,%f,%f,%i')
             
            
            
            
            
            
            
            
            
            
            
            
            
            
            