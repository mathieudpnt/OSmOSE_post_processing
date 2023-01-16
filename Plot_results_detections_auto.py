# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:44:38 2022

@author: torterma
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:58:22 2022

@author: torterma
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from collections import Counter
import matplotlib.dates as mdates
import calendar
from scipy.signal import savgol_filter
import easygui
from datetime import datetime, timedelta
import math
from pytz import timezone
import pytz



# Return a Datafram with the annotations of 1 annotator and 1 label
def df_annot_label(df, annotator, label):
    df_OneAnnot=df.loc[df['annotator'] == annotator]
    df_OneAnnot_OneLabel =df_OneAnnot.loc[df_OneAnnot['annotation'] == label]
    return df_OneAnnot_OneLabel

#%% Functions used in the following code

# Returns 2 lists containing the start and end datetime of each annotation 
def CreatVec_datetime_det(df_results, annotator, label):

    # Create a DataFrame containing the lines corresponding to the 'label' annotations of 'annot_ref'
    det_annot_ref = df_results.loc[df_results['annotator'].isin([annotator])]  
    det_annot_ref_label = det_annot_ref.loc[det_annot_ref['annotation'].isin([label])]
    # print("Number of annotations of reference annotator: ", len(det_annot_ref_label))

    # Read the start and end timestamp of each detection of 'annot_ref' 
    beg_det_ref = np.sort([calendar.timegm(dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f%z").timetuple()) for x in det_annot_ref_label['start_datetime']])
    end_det_ref = np.sort([calendar.timegm(dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f%z").timetuple()) for x in det_annot_ref_label['end_datetime']])

    return beg_det_ref, end_det_ref

#plot tides
def plot_tides(data_path, date_begin_tide, date_end_tide):
    df_maree = pd.read_csv(data_path, skiprows=13, delimiter=';')
    hauteur = df_maree['Valeur']
    hauteur_f = savgol_filter(hauteur, 301, 4) # window size 51, polynomial order 3
    h_max = max(hauteur_f)
    dt_maree = pd.to_datetime(df_maree['# Date'], format='%d/%m/%Y %H:%M:%S' , utc=True)
    #dt_maree = pd.to_datetime(df_maree['# Date'], format='%d/%m/%Y %H:%M:%S')
    dt_maree2 = dt_maree[(dt_maree <= date_end_tide) & (dt_maree >= date_begin_tide)]
    h2 = hauteur_f[(dt_maree <= date_end_tide) & (dt_maree >= date_begin_tide)]
    return (dt_maree2, h2, h_max)

def round_dt(dt, delta):
    dt_min = datetime.min
    dt_min = dt_min.replace(tzinfo=pytz.timezone('Europe/Paris'))

    return dt_min + math.floor((dt - dt_min) / delta) * delta


# %% Enter filename of APLOSE detection/annotation csv

fileName_Results = 'C:/Users/torterma/Documents/Projets_OFB/CETIROISE/Analyses/Annotation/PG_formatteddata_220505_220823.csv'


# tmin and tmax for the plot  
tmin = dt.datetime(2022,5,10,0,0,0)
tmax = dt.datetime(2022,8,21,0,0,0)

# Read result csv
results = pd.read_csv(fileName_Results)

# Read and print label
list_labels = np.unique(results['annotation'])
list_labels = np.ndarray.tolist(list_labels)
list_annotators = np.unique(results['annotator'])
annotators = np.ndarray.tolist(list_annotators)
print("List of label (detectors) : ", list_labels)

# User input : desired time bin duration for the Figures
bin_size = int(easygui.enterbox("Size of the desired time bin duration for the following Figures (in min)"))
#%%

start = results['start_datetime']
dt_start = [dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f%z") for x in start]
    
# Eg with minute positive to detection
# 1- round every datetime to the minute

delta = timedelta(minutes=bin_size)
a= dt_start[0]
b = round_dt(a,delta)
#new_dt_start = [round_dt(dt,delta) for dt in dt_start]
# 2 - only keep one occurence of each dataframe





#%%
column_to_monitor = 'annotation'
counter_annotator = results.groupby('annotator')[column_to_monitor].apply(Counter).unstack(fill_value=0)

 # Plot annotations of each annotator
fig, ax = plt.subplots(figsize=(30,10))
ax = counter_annotator.plot(kind='bar', ax=ax)
plt.ylabel('Number of annotated calls')
plt.xlabel('Annotators')
plt.xticks(rotation=0)
plt.title('Number of detections of each annotator')
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.tick_params(labelsize=20)
ax.set_title('Number of detections of each annotator', fontsize = 40)



#%% Compute a dataframe containing the number of annotations per annotator and per label
counter_label = results.groupby('annotation')['annotator'].apply(Counter).unstack(fill_value=0)

 # Plot annotations of each annotator
fig, ax = plt.subplots(figsize=(30,10))
ax = counter_label.plot(kind='bar', ax=ax)
plt.ylabel('Number of annotated calls', fontsize=30 )
plt.xlabel('Labels', fontsize=30 )
plt.xticks(rotation=0)
plt.title('Number of detections of each annotator, classified by label type', fontsize=40 )
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.tick_params(labelsize=20)
plt.legend(loc=2, prop={'size': 20});


#%% Histogram of detections


annotator = 'PAMGuard'
fig, ax = plt.subplots(nrows = 2, figsize=(20,20))
bars = ('0', '10', '20','30', '40', '50','60', '70', '80', '90', '100')
#y_pos = np.linspace(0,24*2, num=11)

for i, L in enumerate(list_labels):    
    ax[i].set_xlim([tmin, tmax])
    annotation = df_annot_label(results, annotator, L)  
    df_timestamp_beg = annotation['start_datetime']
    #t_dt=pd.to_datetime(df_timestamp_beg, format="%Y-%m-%dT%H:%M:%S.%f+00:00") 
    t_dt=pd.to_datetime(df_timestamp_beg, format="%Y-%m-%dT%H:%M:%S") 
    ax[i].hist(t_dt, bins=111)
    ax[i].grid(color='k', linestyle='-', linewidth=0.2)
    locator = mdates.MonthLocator(interval=1)
    ax[i].xaxis.set_major_locator(locator)
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax[i].set_title(L, fontsize = 20)
    ax[i].tick_params(labelsize=20)
    #ax[i].set_yticks(y_pos)
    ax[i].set_yticklabels(bars)
    #ax[i].set_xlabel('Time (hh.mm)') 
    #ax[i].set_ylabel('Percentage of 10 sec positive to detections for 10 min time bins') 
#%%

annotator = 'PAMGuard'
label = 'Odontocete whistles'
fig, ax = plt.subplots(figsize=(20,10))
bars = ('0', '10', '20','30', '40', '50','60', '70', '80', '90', '100')
y_pos = np.linspace(0,60, num=11)

   
#ax.set_xlim([tmin, tmax])
annotation = df_annot_label(results, annotator, list_labels[0])  
df_timestamp_beg = annotation['start_datetime']
#t_dt=pd.to_datetime(df_timestamp_beg, format="%Y-%m-%dT%H:%M:%S.%f+00:00") 
t_dt=pd.to_datetime(df_timestamp_beg, format="%Y-%m-%dT%H:%M:%S") 
ax.hist(t_dt, bins=24*6)
ax.grid(color='k', linestyle='-', linewidth=0.2)
locator = mdates.HourLocator(interval=2)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_title(L, fontsize = 20)
ax.tick_params(labelsize=20)
ax.set_yticks(y_pos)
ax.set_yticklabels(bars)
#ax[i].set_xlabel('Time (hh.mm)') 
#ax[i].set_ylabel('Percentage of 10 sec positive to detections for 10 min time bins') 

#%%
counter_label.plot.pie(subplots=True,figsize=(30, 10), autopct='%.1f')

#fig, ax = plt.subplots(figsize=(30,10))
#ax = plt.pie(x)

#%% 

fig, ax = plt.subplots(nrows = 2, figsize=(20,20))
plt.grid(color='k', linestyle='-', linewidth=0.2)
y_pos = np.linspace(0,60, num=11)

for i, L in enumerate(annotators):
    annot_whistlesM = df_annot_label(results, L, 'Odontocete Buzzs')  
    df_timestamp_beg = annot_whistlesM['start_datetime']
    t=df_timestamp_beg
    t_dt=pd.to_datetime(t, format="%Y-%m-%dT%H:%M:%S.%f+00:00")
    ax[i].grid(color='k', linestyle='-', linewidth=0.2)
    ax[i].hist(t_dt, bins=24*6) 
    locator = mdates.HourLocator(interval=2)
    ax[i].xaxis.set_major_locator(locator)
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax[i].set_title(L, fontsize = 20)
    ax[i].tick_params(labelsize=20)
    ax[i].set_yticks(y_pos)
    ax[i].set_yticklabels(bars)
    
    #%%

annot_ref = 'jbeesa'
label = 'Odontocete whistles'

True_det = []
Detect_rate = []
False_alarm = []
Missed_det = []   

# Read the start and end timestamp of each detection of 'annot_ref' 
beg_det_ref, end_det_ref = CreatVec_datetime_det(results, annot_ref, label)

# Browse every other annotator (other than 'annot_ref')
for num_annot, x in enumerate(annotators):
   
    # Read the start and end timestamp of each detection of annotator 'x'
    beg_det, end_det = CreatVec_datetime_det(results, x, label)
    # Initialize the metrics 
    nDetect = len(beg_det) # Number of detections of annotator 'x'
    L = len(beg_det_ref) # Number of detections of annot_ref
    Annot_detect = np.zeros((1,nDetect)) # Array of 0 and 1 with length = number of detections of annotator 'x' -> 1 if the annotation matches with one annot_ref annotation
    Ref_detect = np.zeros((1, L)) # Array of 0 and 1 with length = number of detections of annotator annot_ref -> 1 if the annotation matches with one annotatator 'x' annotation
    idx_common_det= np.zeros((1,nDetect))
    # Initialize loop for each annotator    
    b = 0 
        
    for j in range(0,L):
    
        for i in range(b,nDetect):
            
            Rinterv_ref = range(beg_det_ref[j], end_det_ref[j])
            RintervA = range(beg_det[i], end_det[i])
            
            interv_ref = set(Rinterv_ref)
            intervA = set(RintervA)
            len_intervA = len(intervA)            
            overlapInter = interv_ref.intersection(RintervA) 
            
            if len(overlapInter) > 0.5* len_intervA:
                Annot_detect[0,i] = 1 
                idx_common_det[0,i] = 1
                Ref_detect[0,j] = 1                
                b=i+1
                break   
    
    True_det.append(np.sum(Annot_detect))
    False_alarm.append(nDetect-np.sum(Annot_detect))    
    Missed_det.append(L - np.sum(Ref_detect))
    
    print("Comparaison entre l'annotateur", annot_ref, "et l'annotateur", x)
    print('Nombre de détections en commun :', np.sum(Annot_detect))
    print("Nombre de détections manquées par l'autre annotateur :", Missed_det[num_annot])
    print("Nombre de détections que l'autre annotateur a, mais pas vous :", False_alarm[num_annot])
    















