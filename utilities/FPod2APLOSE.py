# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:27:51 2022
This script transforms a FPOD .xls result files to a APLOSE csv results file
@author: torterma
"""
import pandas as pd
import pytz
import easygui
import os
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from datetime import timedelta
from datetime import datetime

# Usually FPOD are in UTC, but think to change the timezone if different
tz_data = 'UTC'

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
FPOD_file_path = askopenfilename(title="Open a FPOD csv file containing the detection results") # show an "Open" dialog box and return the path to the selected file

dataset_name = easygui.enterbox("Dataset name (enter a string): ")
species = easygui.enterbox("Label (enter a species and a call type): ")
det_bin_size = int(easygui.enterbox("Size of the detection bin (in sec)"))


#%%

# Read detection begin time
df_FPOD = pd.read_csv(FPOD_file_path)
# Number of detections
nb_det = len(df_FPOD)


# Transform start detection format from string to absolute datatime (with time zone info)
df_FPOD_start_dt = sorted([pytz.timezone(tz_data).localize(pd.to_datetime(x, format="%d/%m/%Y %H:%M")) for x in df_FPOD['Date heure']])
# Compute the absolute end date time of detection
df_FPOD_end_dt = sorted([x + timedelta(seconds = det_bin_size) for x in df_FPOD_start_dt])

# Change datetime format to match with APLOSE format
df_FPOD_start_AP = [datetime.strftime(dt, ("%Y-%m-%dT%H:%M:%S.%f"))[:-3] + datetime.strftime(dt, ("%Y-%m-%dT%H:%M:%S.%f%z"))[26:29] + ':' + datetime.strftime(dt, ("%Y-%m-%dT%H:%M:%S.%f%z"))[29:] for dt in df_FPOD_start_dt]
df_FPOD_end_AP = [datetime.strftime(dt, ("%Y-%m-%dT%H:%M:%S.%f"))[:-3] + datetime.strftime(dt, ("%Y-%m-%dT%H:%M:%S.%f%z"))[26:29] + ':' + datetime.strftime(dt, ("%Y-%m-%dT%H:%M:%S.%f%z"))[29:] for dt in df_FPOD_end_dt]


# Build the dataframe
data = {'dataset' : [dataset_name]*nb_det, 'filename' : ['']*nb_det, 'start_time' : [0]*nb_det, 'end_time' : [det_bin_size]*nb_det, 'start_frequency' :  [0]*nb_det, 'end_frequency' :  [0]*nb_det,'annotation' : [species]*nb_det, 'annotator' : ['FPOD']*nb_det, 'start_datetime' : df_FPOD_start_AP,  'end_datetime' : df_FPOD_end_AP}
df_APLOSE = pd.DataFrame(data)  
#▒ Save the csv file in the same folder as the FPOD results csv file
df_APLOSE.to_csv(FPOD_file_path+'_APLOSE.csv', index=False)







