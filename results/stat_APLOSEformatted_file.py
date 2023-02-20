# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:32:40 2023
This script allows to perform some simple statistics regarding annotated data
e.g. % annotations for each label/annotator, % agreement between two annotators
@author: torterma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import calendar
from scipy.signal import savgol_filter

from datetime import datetime, timedelta
import math
import pytz
from dateutil.tz import gettz # pip install python-dateutil
from astral.sun import sun
import astral
from collections import Counter

#%% Functions used in the following code
# Return a Datafram with the annotations of 1 annotator and 1 label

def df_annot_label(df, annotator, label):
    df_OneAnnot=df.loc[df['annotator'] == annotator]
    df_OneAnnot_OneLabel =df_OneAnnot.loc[df_OneAnnot['annotation'] == label]
    return df_OneAnnot_OneLabel


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

def round_dt(dt, delta, tz_info):
    dt_min = datetime.min

    #dt_min_aware = timezone('Europe/Paris').localize(dt_min)
    dt_min = dt_min.replace(tzinfo=pytz.UTC)
    #dt_min = dt_min.replace(tzinfo=pytz.FR)
    dt_round = dt_min + math.floor((dt - dt_min) / delta) * delta
    dt_round_TZ = dt_round.astimezone(gettz(tz_info))
    
    
    return dt_round_TZ 

# Fonction qui permet d'obtenir l'heure de lever et de coucher du soleil selon la position GPS
def suntime_hour(date_beg, date_end, timeZ, lat,lon):
    # Infos sur la localisation
    gps = astral.LocationInfo( timezone=timeZ,latitude=lat, longitude=lon)
    # List of days during when the data were recorded
    list_time = pd.date_range(date_beg, date_end)
    h_sunrise = []
    h_sunset = []
    # For each day : find time of sunset, sun rise, begin dawn and dusk
    for day in list_time:
        suntime = sun(gps.observer,date=day, dawn_dusk_depression = astral.Depression, tzinfo = timeZ)
        
        dawn_dt=(suntime['dawn'])
        
        dusk_dt=(suntime['dusk'])
        
        day_dt=(suntime['sunrise'])
        
        night_dt=(suntime['sunset'])
        
        day_hour = day_dt.hour+day_dt.minute/60
        night_hour = night_dt.hour+night_dt.minute/60
        h_sunrise.append(day_hour)
        h_sunset.append(night_hour)
        hour_sunrise = h_sunrise[0:len(h_sunrise)-1]
        hour_sunset = h_sunset[0:len(h_sunset)-1]
    return hour_sunrise, hour_sunset

def clean_PG_FA_fct(results, tz_info):
    
    # 1- lire la date dans le nom du fichier 
    filename = results['filename']
    filename_d = [x[9:28] for x in filename]
    date_file = [dt.datetime.strptime(x, "%Y-%m-%d_%H-%M-%S").astimezone(gettz(tz_info)) for x in filename_d]

    # 2 - lire la date dans start_datetime
    start = results['start_datetime']
    dt_start = [dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f%z") for x in start]
    # 3- les comparer et si < 10 sec, on retient le numéro de la ligne
    idx_FA = []
    for i in range (0, len(dt_start)):
        d = (dt_start[i]-date_file[i]).total_seconds()
        if d < 10:
            idx_FA.append(i)
    # 4 - on delete toutes les lignes concernées
    print('result cleaned')
    results_cleaned = results.drop(labels=idx_FA, axis=0)
    
    return results_cleaned


# =============================================================================
# =============================================================================
# =============================================================================
# # %% User input
#fileName_Results = 'C:/Users/torterma/Documents/Projets_OFB/CETIROISE/Analyses/Annotation/HF_220725_POINT_E/CETIROISE_POINT_E_20220725_results.csv'
#fileName_Results = 'L:/acoustock/Bioacoustique/DATASETS/CETIROISE/ANALYSE/FPOD/PHASE_1_2/POINT_G_Marsouins_minute_positive_PHASE_1_et_2.csv_APLOSE.csv'
#fileName_Results ='C:/Users/torterma/Documents/Projets_OFB/CETIROISE/Analyses/Annotation/HF_220717_POINT_B/CETIROISE_HF 17072022_Results_final.csv'
fileName_Results = 'C:/Users/torterma/Documents/Projets_OFB/CETIROISE/Analyses/Annotation/HF_220717_POINT_B/CETIROISE_HF 17072022_Results_final.csv'
# Time zone of the plot
tz_info = 'Europe/Paris'
# tmin and tmax for the plot  
tmin = dt.datetime(2022,7,17,0,0,0).astimezone(gettz(tz_info))
tmax = dt.datetime(2022,7,18,0,0,0).astimezone(gettz(tz_info))
# Group all annotators ?
#group_annot = True



# Size of the desired time bin duration for the following Figures (in sec)"
bin_size_det = 60
# Size of the representation time bin duration for the following Figures 
# 1 if days and 24 if hours 24*6 if 10 minutes
bin_size_rep = 1
# Number max of bin_size_det in bin_size_rep
max_y = 1440
# Set clean_PG_FA to True if you want to remove the FA that occur at the begining of each file
clean_PG_FA = False


# =============================================================================
# =============================================================================
# =============================================================================

results = pd.read_csv(fileName_Results)

# Read and print label
list_labels = np.unique(results['annotation'])
list_labels = np.ndarray.tolist(list_labels)
list_annotators = np.unique(results['annotator'])
annotators = np.ndarray.tolist(list_annotators)
print("List of label (detectors) : ", list_labels)
print("List of annotators : ", list_annotators)



# Plot annotations of each annotator
counter_label = results.groupby('annotation')['annotator'].apply(Counter).unstack(fill_value=0)
print(counter_label)

fig, ax = plt.subplots(figsize=(30,15))
ax = counter_label.plot(kind='bar', ax=ax)
plt.ylabel('Number of annotated calls', fontsize=30 )
plt.xlabel('Labels', fontsize=30 )
plt.xticks(rotation=0)
plt.title('Number of detections of each annotator, classified by label type', fontsize=40 )
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.tick_params(labelsize=25)
plt.legend(loc=2, prop={'size': 30});


#%% Comparaison de plusieurs annotateurs
annot_ref = 'mtorte'
label_ref = 'Odontocete whistles'
True_det = []
#Detect_rate = []
False_alarm = []
Missed_det = []
# Read the start and end timestamp of each detection of 'annot_ref'
beg_det_ref, end_det_ref = CreatVec_datetime_det(results, annot_ref, label_ref)

# Browse every other annotator (other than 'annot_ref')
for num_annot, x in enumerate(annotators):
    # Read the start and end timestamp of each detection of annotator 'x'
    beg_det, end_det = CreatVec_datetime_det(results, x, label_ref)
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
    print("\nComparaison entre l'annotateur", annot_ref, "et l'annotateur", x)
    print('Nombre de détections en commun :', np.sum(Annot_detect))
    print("Nombre de détections manquées par l'autre annotateur :", Missed_det[num_annot])
    print("Nombre de détections que l'autre annotateur a, mais pas vous :", False_alarm[num_annot])



#%% Comparaison avec un detecteur automatique

fileName_Detector = 'C:/Users/torterma/Documents/Projets_OFB/CETIROISE/Analyses/PAMGuard/POINT_B_PHASE_1/PG_formatteddata_2022-07-17.csv'
detector_csv = pd.read_csv(fileName_Detector)
Detector_name = 'PAMGuard'
label = 'Odontocete whistles'
label2 = 'Whistle and moan detector'

True_det = []
Detect_rate = []
False_alarm = []
Missed_det = []   
FA = []

# Read the start and end timestamp of each detection of 'annot_ref' 
beg_det_detector, end_det_detector = CreatVec_datetime_det(detector_csv, Detector_name, label2)



# Browse every other annotator (other than 'annot_ref')
for num_annot, x in enumerate(annotators):
   
    # Read the start and end timestamp of each detection of annotator 'x'
    beg_det, end_det = CreatVec_datetime_det(results, x, label)
    # Initialize the metrics 
    nDetect = len(beg_det) # Number of detections of annotator 'x'
    L = len(beg_det_detector) # Number of detections of annot_ref
    Annot_detect = np.zeros((1,nDetect)) # Array of 0 and 1 with length = number of detections of annotator 'x' -> 1 if the annotation matches with one annot_ref annotation
    Ref_detect = np.zeros((1, L)) # Array of 0 and 1 with length = number of detections of annotator annot_ref -> 1 if the annotation matches with one annotatator 'x' annotation
    # Initialize loop for each annotator    
    b = 0 
      
    
    for j in range(0,L):
    
        for i in range(0,nDetect):
            if beg_det_detector[j] == end_det_detector[j]:
                end_det_detector[j] = end_det_detector[j]+1
            Rinterv_ref = range(beg_det_detector[j],end_det_detector[j])
            RintervA = range(beg_det[i], end_det[i])
            
            interv_ref = set(Rinterv_ref)
            intervA = set(RintervA)
            len_intervA = len(intervA)            
            overlapInter = interv_ref.intersection(RintervA) 
           
            if len(overlapInter) > 0:#0.1* len_intervA:
                Annot_detect[0,i] = 1               
                Ref_detect[0,j] = 1 
                FA.append(beg_det_detector[j])
                b=i+1
                break   

    True_det.append(np.sum(Ref_detect))
    Detect_rate.append(np.sum(Ref_detect)/L)
    False_alarm.append(L-np.sum(Ref_detect))    
    Missed_det.append(nDetect - np.sum(Annot_detect))
    
    Recall = np.sum(Annot_detect)/(np.sum(Annot_detect)+Missed_det[num_annot])
    Precision = np.sum(Annot_detect)/(np.sum(Annot_detect) + False_alarm[num_annot])
    print("Comparaison entre le détecteur et l'annotateur", x)
    print("Nombre de détections en commun (VRAIS POSITIFS):", np.sum(Annot_detect))
    print("Nombre de détections manquées par le détecteur (FAUX NEGATIFS):", Missed_det[num_annot])
    print("Nombre de fausses détections faites par le détecteur (FAUX POSITIFS)  :", False_alarm[num_annot])
    print("Rappel (recall)", Recall ) 
    print("Précision", Precision)



















