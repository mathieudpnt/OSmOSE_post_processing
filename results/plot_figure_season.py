# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:46:38 2023

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
from tkinter import Tk
from tkinter import filedialog

from utilities.def_func import get_detection_files, sorting_detections, t_rounder, get_timestamps, input_date, suntime_hour

# %% User inputs

files_list = get_detection_files(1)
df_detections, t_detections = sorting_detections(files_list, timebin_new = 3600, tz = pytz.FixedOffset(120))

time_bin = list(set(t_detections['max_time']))
fmax = list(set(t_detections['max_freq']))
annotators = list(set(t_detections['annotators'].explode()))
labels = list(set(t_detections['labels'].explode()))
tz_data = df_detections['start_datetime'][0].tz

# Chose your mode :
# fixed : hard coded date interval
# auto : the script automatically extract the timestamp from the timestamp file
# input : you will fill a dialog box with the start and end date

dt_mode = 'fixed'

if dt_mode == 'fixed':
    begin_date = pd.Timestamp('2022-04-29T00:00:00.000000+0200')
    end_date = pd.Timestamp('2023-07-01T00:00:00.000000+0200')
elif dt_mode == 'auto':
    timestamps_file = get_timestamps()
    begin_date = pd.to_datetime(timestamps_file['timestamp'].iloc[0], format='%Y-%m-%dT%H:%M:%S.%f%z')
    end_date = pd.to_datetime(timestamps_file['timestamp'].iloc[-1], format='%Y-%m-%dT%H:%M:%S.%f%z') + dt.timedelta(seconds=time_bin[0])
elif dt_mode == 'input':
    begin_date = input_date('Enter begin date')
    end_date = input_date('Enter end date')

print("\ntime_bin: ", str(time_bin), "s", end='')
print("\nfmax: ", str(fmax), "Hz", end='')
print('\nannotators: ', str(annotators), end='')
print('\nlabels: ', str(labels), end='')
print('\nBegin date: {0}'.format(begin_date))
print('\nEnd date: {0}'.format(end_date))


# %%

def get_season(date): # winter : december -> february, spring : march -> may, summer : june to august, autumn : september to november

  m = date.month
  x = m%12 // 3 + 1
  if x == 1:
    season = "Winter"
  if x == 2:
    season = "Spring"
  if x == 3:
    season = "Summer"
  if x == 4:
    season = "Autumn"
  return season

# Ask the file with the number of days per season

root = Tk()
root.withdraw()

file_seasons = []

file_path = filedialog.askopenfilename(
    title='Select Sylence_seasons.csv or FPOD_seasons.csv',
    filetypes=[('CSV files', '*.csv')]
)

file_seasons.append(file_path)
    
seasons = pd.read_csv(file_seasons[0], delimiter = ';', encoding='latin-1', index_col = 0)
# Ask which site 
site = easygui.buttonbox(msg='On which CETIROISE site are you working ?', choices=('A', 'B', 'C', 'D', 'G', 'E', 'F'))
# Number of days per season at the study site
seasons_site = seasons[site]

season_det = [get_season(x) for x in df_detections['start_datetime']]

nb_det_spring = season_det.count("Spring")
nb_det_summer = season_det.count("Summer")
nb_det_autumn = season_det.count("Autumn")
nb_det_winter = season_det.count("Winter")

prop_det_spring = nb_det_spring/seasons_site['Printemps']
prop_det_summer = nb_det_summer/seasons_site['Ete']
prop_det_autumn = nb_det_autumn/seasons_site['Automne']
prop_det_winter = nb_det_winter/seasons_site['Hiver']

hist_seasons = [prop_det_spring, prop_det_summer, prop_det_autumn, prop_det_winter]
#hist_seasons = pd.DataFrame([prop_det_spring, prop_det_summer, prop_det_autumn, prop_det_winter], index = )

fig, ax = plt.subplots(figsize=(20, 10))
ax.bar(['Printemps', 'Eté', 'Automne', 'Hiver'], hist_seasons)
ax2 = ax.twinx()
ax2.scatter([0,1,2,3], seasons_site, s = 500, color = 'orange')

# Labels
ax.set_ylabel('Nombre d heure', fontsize=25, color = 'blue')
ax2.set_ylabel('Nombre de jour échantillonnés', fontsize=25, color = 'orange')
ax.set_xlabel('Saison', fontsize=25, rotation=0)

# ticks
ax.tick_params(axis='both', rotation=0, labelsize=20)


print("Spring\nNumber of recording days : {}".format(seasons_site['Printemps']) + "\nAverage daily number of hour with detections : {}".format(prop_det_spring))
print("Summer\nNumber of recording days : {}".format(seasons_site['Ete']) + "\nAverage daily number of hour with detections : {}".format(prop_det_summer))
print("Autumn\nNumber of recording days : {}".format(seasons_site['Automne']) + "\nAverage daily number of hour with detections : {}".format(prop_det_autumn))
print("Winter\nNumber of recording days : {}".format(seasons_site['Hiver']) + "\nAverage daily number of hour with detections : {}".format(prop_det_winter))


#%% Proportion of days per season positive to detection

day_det = [dt.datetime.strftime(x, '%y-%m-%d') for x in df_detections['start_datetime']]
unique_dd = set(day_det)

unique_dd_dt = [dt.datetime.strptime(x, '%y-%m-%d') for x in unique_dd]

season_dd = [get_season(x) for x in unique_dd_dt]

nb_dd_spring = season_dd.count("Spring")
nb_dd_summer = season_dd.count("Summer")
nb_dd_autumn = season_dd.count("Autumn")
nb_dd_winter = season_dd.count("Winter")

prop_dd_spring = nb_dd_spring/seasons_site['Printemps']
prop_dd_summer = nb_dd_summer/seasons_site['Ete']
prop_dd_autumn = nb_dd_autumn/seasons_site['Automne']
prop_dd_winter = nb_dd_winter/seasons_site['Hiver']


hist_seasons_dd = [prop_dd_spring, prop_dd_summer, prop_dd_autumn, prop_dd_winter]
#hist_seasons = pd.DataFrame([prop_det_spring, prop_det_summer, prop_det_autumn, prop_det_winter], index = )

fig, ax = plt.subplots(figsize=(20, 10))
ax.bar(['Printemps', 'Eté', 'Automne', 'Hiver'], hist_seasons_dd)
ax2 = ax.twinx()
ax2.scatter([0,1,2,3], seasons_site, s = 500, color = 'orange')

# Labels
ax.set_ylabel('Proportion de jours', fontsize=25, color = 'tab:blue')
ax2.set_ylabel('Nombre de jour échantillonnés', fontsize=25, color = 'orange')
ax.set_xlabel('Saison', fontsize=25, rotation=0)

# ticks
ax.tick_params(axis='both', rotation=0, labelsize=20)


print("Spring\nNumber of recording days : {}".format(seasons_site['Printemps']) + "\nProportion of days with detection: {}".format(prop_dd_spring))
print("Summer\nNumber of recording days : {}".format(seasons_site['Ete']) + "\nProportion of days with detection : {}".format(prop_dd_summer))
print("Autumn\nNumber of recording days : {}".format(seasons_site['Automne']) + "\nProportion of days with detection : {}".format(prop_dd_autumn))
print("Winter\nNumber of recording days : {}".format(seasons_site['Hiver']) + "\nProportion of days with detection : {}".format(prop_dd_winter))








































