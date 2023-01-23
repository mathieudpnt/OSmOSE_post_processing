# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:04:21 2022

@author: torterma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

import matplotlib.dates as mdates

from pytz import all_timezones
from astral.sun import sun
import astral


#%%

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
    beg_det_struc_time = [(dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f+00:00").timetuple()) for x in det_annot_ref_label['start_datetime']]
    beg_det_datetime = [(dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f+00:00")) for x in det_annot_ref_label['start_datetime']]
    #end_det_ref = [(dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f+00:00")) for x in det_annot_ref_label['end_datetime']]

    return beg_det_struc_time, beg_det_datetime

# Fonction qui permet d'obtenir l'heure de lever et de coucher du soleil selon la position GPS
def suntime_hour(date_beg, date_end, timeZ, lat,lon,tz):
    # Infos sur la localisation
    gps = astral.LocationInfo( timezone=timeZ,latitude=lat, longitude=lon)
    # List of days during when the data were recorded
    list_time = pd.date_range(date_beg, date_end)
    h_sunrise = []
    h_sunset = []
    # For each day : find time of sunset, sun rise, begin dawn and dusk
    for day in list_time:
        suntime = sun(gps.observer,date=day, dawn_dusk_depression = astral.Depression)
        
        dawn_dt=(suntime['dawn'])
        
        dusk_dt=(suntime['dusk'])
        
        day_dt=(suntime['sunrise'])
        
        night_dt=(suntime['sunset'])
        
        day_hour = tz+day_dt.hour+day_dt.minute/60
        night_hour = tz+night_dt.hour+night_dt.minute/60
        h_sunrise.append(day_hour)
        h_sunset.append(night_hour)
        hour_sunrise = h_sunrise[0:len(h_sunrise)-1]
        hour_sunset = h_sunset[0:len(h_sunset)-1]
    return hour_sunrise, hour_sunset



# =============================================================================
# #%% User input

# =============================================================================

fileName_FPOD = 'C://Users/torterma/Documents/Projets_OFB/CETIROISE/Analyses/Annotation/POINT_G_Delphinidés_minute_positive.csv'
raw_results = pd.read_csv(fileName_FPOD)
# tmin and tmax for the plot  
tmin = dt.datetime(2022,5,9,0,0,0)
tmax = dt.datetime(2022,8,27,0,0,0)




t_detections=pd.to_datetime(raw_results['Date heure'], format="%d/%m/%Y %H:%M") 
t_detections = t_detections +pd.Timedelta(hours=2)

#♦ Figure to represent the number of 'n' (n is the precision of the manual detection, here it is minute) with detection within hour, day, week, month...
fig, ax = plt.subplots(figsize=(20,10))
bars = ('0', '10', '20','30', '40', '50','60', '70', '80', '90', '100')
# To display results in % : change the second number with the maximum number of detection within your timeframe bin (eg. 60 minutes in one hour)
y_pos = np.linspace(0,60*24, num=11)

a=ax.hist(t_detections, bins=111)
ax.grid(color='k', linestyle='-', linewidth=0.2)
# = mdates.HourLocator(interval=2)
#ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
#ax.set_title(L, fontsize = 20)
ax.tick_params(labelsize=20)
ax.set_yticks(y_pos)
ax.set_yticklabels(bars)
ax.set_xlim([tmin, tmax])


#%%

# fileName_FPOD = 'C://Users/torterma/Documents/Projets_OFB/CETIROISE/Analyses/Annotation/POINT_G_Delphinidés_minute_positive.csv'
# raw_results = pd.read_csv(fileName_FPOD)
# # tmin and tmax for the plot  
# tmin = dt.datetime(2022,5,9,0,0,0)
# tmax = dt.datetime(2022,8,27,0,0,0)


# t_detections=raw_results['Date heure']
# t_detections_dt = [(dt.datetime.strptime(x, "%d/%m/%Y %H:%M")+ dt.timedelta(hours=2)) for x in t_detections]

# #♦ Figure to represent the number of 'n' (n is the precision of the manual detection, here it is minute) with detection within hour, day, week, month...
# fig, ax = plt.subplots(figsize=(20,10))
# bars = ('0', '10', '20','30', '40', '50','60', '70', '80', '90', '100')
# # To display results in % : change the second number with the maximum number of detection within your timeframe bin (eg. 60 minutes in one hour)
# y_pos = np.linspace(0,60, num=11)

# a=ax.hist(t_detections, bins=111)
# ax.grid(color='k', linestyle='-', linewidth=0.2)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
# #ax.set_title(L, fontsize = 20)
# ax.tick_params(labelsize=20)
# ax.set_yticks(y_pos)
# ax.set_yticklabels(bars)
# ax.xaxis_date()
# ax.set_xlim([tmin, tmax])


#%% Sun
date_beg = '2022-05-17'
date_end = '2022-08-28'

x_data = np.arange(date_beg,date_end,dtype="M8[D]")


t_detections_dt = [(dt.datetime.strptime(x, "%d/%m/%Y %H:%M")) for x in raw_results['Date heure']]
t_detections_struc_time = [(dt.datetime.strptime(x, "%d/%m/%Y %H:%M").timetuple()) for x in raw_results['Date heure']]

Day_det = t_detections_dt
Hour_det = [x.tm_hour + x.tm_min/60 for x in t_detections_struc_time] 

#x_data = [dt.datetime.strptime(np.array2string(x_data[i]), "'%Y-%m-%d'") for i in range(0,len(x_data))]

timeZ = 'UTC'
# A MODIFIER : décalage horaire entre UTC et heure locale
tz = 0
# A MODIFIER : coordonnées géographiques
lat = "48°31′N"
lon = "5°7'W"
# A MODIFIER : nom du jeu de données
dataset_name = 'CETIROISE_POINT_B'
# Nautical dawn and dusk start when the sun is 12° below the horizon
astral.Depression = 12
# Calcul des heures de lever et coucher du soleil à la position du jeu de données
[hour_sunrise, hour_sunset] = suntime_hour(date_beg, date_end, timeZ, lat,lon,tz)

# Plot figure
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(x_data,hour_sunrise, color='k')
plt.plot(x_data,hour_sunset, color='k')
plt.scatter(Day_det,Hour_det)
locator = mdates.DayLocator(interval=7)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%D'))
ax.grid(color='k', linestyle='-', linewidth=0.2)



#%% Convert the result timebin to a larger timebin (eg hour, day...)


























