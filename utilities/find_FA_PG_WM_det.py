# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:55:32 2023

@author: torterma
"""
from post_processing_detections.utilities.def_func import get_detection_files, sorting_detections, input_date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

file = get_detection_files(1)

delimiter = ','
df = pd.read_csv(file[0], sep=delimiter)
list_annotators = list(df['annotator'].drop_duplicates())
list_labels = list(df['annotation'].drop_duplicates())
max_freq = int(max(df['end_frequency']))


#%%
startF = df['start_frequency']
endF = df['end_frequency']
# datetime det
dt_det = df['start_datetime']
dt_det = pd.to_datetime(df['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z')
ts_det = [datetime.timestamp(d) for d in dt_det]

fig, ax = plt.subplots(figsize=(20, 9))
plt.hist(startF, 100)
plt.hist(endF, 100)


#%% Look for positions of detections with a small bandwidth 
bandwidth = [end-start for end, start in zip(endF, startF)]
pos = [x for x in range(len(bandwidth)) if bandwidth[x] < 200]
small_bw = [bandwidth[x] for x in pos] 

startF_sb = [startF[x] for x in pos]
ts_det_sb = [ts_det[x] for x in pos]

fig, ax = plt.subplots(figsize=(20, 9))
plt.scatter(ts_det_sb, startF_sb)

#%% Look for positions of detections with a small startf 

pos = [x for x in range(len(startF)) if startF[x] < 10000]

bandwidth_sf = [bandwidth[x] for x in pos]
startF_sf = [startF[x] for x in pos]
ts_det_sf = [ts_det[x] for x in pos]

fig, ax = plt.subplots(figsize=(20, 9))
plt.scatter(ts_det_sf[5000:6000], startF_sf[5000:6000])





#%% All freq

fig, ax = plt.subplots(figsize=(20, 9))
plt.scatter(ts_det[4000:5000], startF[4000:5000])
plt.ylim(5000, 10000)








