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
import easygui

#%% Functions used in the following code
# Return a Dataframe with the annotations of 1 annotator and 1 label

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

#%% User input
#fileName_Results = 'L:/acoustock/Bioacoustique/DATASETS/CETIROISE/ANALYSE/FPOD/PHASE_1_2/POINT_G_Marsouins_minute_positive_PHASE_1_et_2.csv_APLOSE.csv'
# fileName_Results ='C:/Users/torterma/Documents/Projets_OFB/CETIROISE/Analyses/PAMGuard/POINT_D_PHASE_1/PG_rawdata_220509_220827.csv'
#fileName_Results = 'L:/acoustock/Bioacoustique/DATASETS/CETIROISE/ANALYSE/PAMGUARD_threshold_7/PHASE_1_POINT_B/Binary/PG_rawdata_220510_220823.csv'
fileName_Results = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/analysis/C2D1_070722/PG Binary/PG_rawdata_220706_220707.csv'

# Time zone of the plot
tz_info = 'Europe/Paris'
# tmin and tmax for the plot  
tmin = dt.datetime(2022,7,7,0,0,0).astimezone(gettz(tz_info))
tmax = dt.datetime(2022,7,8,0,0,0).astimezone(gettz(tz_info))
# Size of the desired time bin duration for the following Figures (in sec)"
bin_size_det = 60
# Size of the representation time bin duration for the following Figures 
# 1 if days and 24 if hours 24*6 if 10 minutes
bin_size_rep = 24
# Number max of bin_size_det in bin_size_rep
max_y = 60*24
# Set clean_PG_FA to True if you want to remove the FA that occur at the begining of each file
clean_PG_FA = False


# =============================================================================
# =============================================================================
# =============================================================================

# Read result csv
#results_raw = pd.read_csv(fileName_Results)
results = pd.read_csv(fileName_Results)

# Read and print label
list_labels = np.unique(results['annotation'])
list_labels = np.ndarray.tolist(list_labels)
list_annotators = np.unique(results['annotator'])
annotators = np.ndarray.tolist(list_annotators)
print("List of label (detectors) : ", list_labels)
# 1- Ask user which label they want to plot and from which 'detector' or 'annotator'
# easygui multchoicebox
# Select annotator from list
if len(list_annotators)>1:
    text_an = 'Select annotator(s) from list'
    title_an = 'Annotator selection'
    output_an = multchoicebox(text_an, title_an, list_annotators)
    results = results.loc[results['annotator'].isin(output_an)] 
    
# Select label from list
if len(list_labels)>1:
    text_lb = 'Select on or more labels(s) from list'
    title_lb = 'Label selection'
    output_lb = multchoicebox(text_lb, title_lb, list_labels)
    results = results.loc[results['annotation'].isin(output_lb)] 

#%%

# Clean the false alarms at the begining of each files
if clean_PG_FA == True:
    results = clean_PG_FA_fct(results, tz_info)

start = results['start_datetime']
dt_start = [dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f%z") for x in start]
    
# 1- round every datetime to the desired timebin (bin_size_det)
delta = timedelta(seconds=bin_size_det)
dt_start_round = [round_dt(i,delta, tz_info) for i in dt_start]

# 2 - only keep one occurence of each dataframe
dt_start_round_sorted= list(set(dt_start_round))
dt_start_round_sorted.sort()

#%% Plot histogram of detections


fig, ax = plt.subplots(figsize=(20,10))

#bars = ('0', '10', '20','30', '40', '50','60', '70', '80', '90', '100')
#y_pos = np.linspace(0, max_y, num=11)

ax.set_xlim([tmin, tmax])
ax.hist(dt_start_round_sorted, bins = (tmax-tmin).days*bin_size_rep, range = (tmin, tmax))
ax.grid(color='k', linestyle='-', linewidth=0.2)
#locator = mdates.MonthLocator(interval=1, tz = pytz.timezone(tz_info))
locator = mdates.DayLocator(bymonthday=(1,15), tz = pytz.timezone(tz_info))
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %B', tz = pytz.timezone(tz_info)))
#ax.set_title('Détections PamGuard' , fontsize = 28)
#ax.set_title("Label " + str(output_lb) + "\n Annotateur " + str(output_an), fontsize = 28)
ax.tick_params(labelsize=28)
ax.set_xlabel('Date', fontsize = 28) 
ax.set_ylabel('Nombre de minutes positives par jour', fontsize = 28) 
plt.xticks(rotation=45, ha="right")
ax.set_ylim([0, 60*24])

#ax.set_yticks(y_pos)
#ax.set_yticklabels(bars)


#%%

#%% Sun
date_beg = '2022-07-07'
date_end = '2022-07-08'

x_data = np.arange(date_beg,date_end,dtype="M8[D]")


t_detections_dt = [(dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f%z")).astimezone(gettz(tz_info)) for x in results['start_datetime']]
t_detections_struc_time = [(dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(gettz(tz_info)).timetuple()) for x in results['start_datetime']]

Day_det = t_detections_dt
Hour_det = [x.tm_hour + x.tm_min/60 for x in t_detections_struc_time] 

#x_data = [dt.datetime.strptime(np.array2string(x_data[i]), "'%Y-%m-%d'") for i in range(0,len(x_data))]



# A MODIFIER : coordonnées géographiques
lat = "48°4′N"
lon = "4°46'W"
# A MODIFIER : nom du jeu de données

# Nautical dawn and dusk start when the sun is 12° below the horizon
astral.Depression = 12
# Calcul des heures de lever et coucher du soleil à la position du jeu de données
[hour_sunrise, hour_sunset] = suntime_hour(date_beg, date_end, tz_info, lat,lon)

# Plot figure
fig, ax = plt.subplots(figsize=(20,10))
ax.set_xlim([x_data[0], x_data[-1]])
plt.plot(x_data,hour_sunrise, color='k')
plt.plot(x_data,hour_sunset, color='k')
plt.scatter(Day_det,Hour_det)
ax.set_ylim([0, 24.99])
plt.xticks(rotation=45, ha="right")
locator = mdates.DayLocator(interval=15)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
ax.grid(color='k', linestyle='-', linewidth=0.2)
ax.tick_params(labelsize=28)
ax.set_ylabel('Heure locale (Brest)', fontsize = 28) 
ax.set_xlabel('Date', fontsize = 28) 
ax.set_title('Point G', fontsize = 40)




#%% 

# counter_label.plot.pie(subplots=True,figsize=(30, 10), autopct='%.1f')

#fig, ax = plt.subplots(figsize=(30,10))
#ax = plt.pie(x)

#%% 

# fig, ax = plt.subplots(nrows = 2, figsize=(20,20))
# plt.grid(color='k', linestyle='-', linewidth=0.2)
# y_pos = np.linspace(0,60, num=11)

# for i, L in enumerate(annotators):
#     annot_whistlesM = df_annot_label(results, L, 'Odontocete Buzzs')  
#     df_timestamp_beg = annot_whistlesM['start_datetime']
#     t=df_timestamp_beg
#     t_dt=pd.to_datetime(t, format="%Y-%m-%dT%H:%M:%S.%f+00:00")
#     ax[i].grid(color='k', linestyle='-', linewidth=0.2)
#     ax[i].hist(t_dt, bins=24*6) 
#     locator = mdates.HourLocator(interval=2)
#     ax[i].xaxis.set_major_locator(locator)
#     ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#     ax[i].set_title(L, fontsize = 20)
#     ax[i].tick_params(labelsize=20)
#     ax[i].set_yticks(y_pos)
#     ax[i].set_yticklabels(bars)
    
#     #%%

# annot_ref = 'jbeesa'
# label = 'Odontocete whistles'

# True_det = []
# Detect_rate = []
# False_alarm = []
# Missed_det = []   

# # Read the start and end timestamp of each detection of 'annot_ref' 
# beg_det_ref, end_det_ref = CreatVec_datetime_det(results, annot_ref, label)

# # Browse every other annotator (other than 'annot_ref')
# for num_annot, x in enumerate(annotators):
   
#     # Read the start and end timestamp of each detection of annotator 'x'
#     beg_det, end_det = CreatVec_datetime_det(results, x, label)
#     # Initialize the metrics 
#     nDetect = len(beg_det) # Number of detections of annotator 'x'
#     L = len(beg_det_ref) # Number of detections of annot_ref
#     Annot_detect = np.zeros((1,nDetect)) # Array of 0 and 1 with length = number of detections of annotator 'x' -> 1 if the annotation matches with one annot_ref annotation
#     Ref_detect = np.zeros((1, L)) # Array of 0 and 1 with length = number of detections of annotator annot_ref -> 1 if the annotation matches with one annotatator 'x' annotation
#     idx_common_det= np.zeros((1,nDetect))
#     # Initialize loop for each annotator    
#     b = 0 
        
#     for j in range(0,L):
    
#         for i in range(b,nDetect):
            
#             Rinterv_ref = range(beg_det_ref[j], end_det_ref[j])
#             RintervA = range(beg_det[i], end_det[i])
            
#             interv_ref = set(Rinterv_ref)
#             intervA = set(RintervA)
#             len_intervA = len(intervA)            
#             overlapInter = interv_ref.intersection(RintervA) 
            
#             if len(overlapInter) > 0.5* len_intervA:
#                 Annot_detect[0,i] = 1 
#                 idx_common_det[0,i] = 1
#                 Ref_detect[0,j] = 1                
#                 b=i+1
#                 break   
    
#     True_det.append(np.sum(Annot_detect))
#     False_alarm.append(nDetect-np.sum(Annot_detect))    
#     Missed_det.append(L - np.sum(Ref_detect))
    
#     print("Comparaison entre l'annotateur", annot_ref, "et l'annotateur", x)
#     print('Nombre de détections en commun :', np.sum(Annot_detect))
#     print("Nombre de détections manquées par l'autre annotateur :", Missed_det[num_annot])
#     print("Nombre de détections que l'autre annotateur a, mais pas vous :", False_alarm[num_annot])
    






#%%
# column_to_monitor = 'annotation'
# counter_annotator = results.groupby('annotator')[column_to_monitor].apply(Counter).unstack(fill_value=0)

#  # Plot annotations of each annotator
# fig, ax = plt.subplots(figsize=(30,10))
# ax = counter_annotator.plot(kind='bar', ax=ax)
# plt.ylabel('Number of annotated calls')
# plt.xlabel('Annotators')
# plt.xticks(rotation=0)
# plt.title('Number of detections of each annotator')
# ax.set_axisbelow(True)
# ax.yaxis.grid(color='gray', linestyle='dashed')
# ax.tick_params(labelsize=20)
# ax.set_title('Number of detections of each annotator', fontsize = 40)



#%% Compute a dataframe containing the number of annotations per annotator and per label
# counter_label = results.groupby('annotation')['annotator'].apply(Counter).unstack(fill_value=0)

#  # Plot annotations of each annotator
# fig, ax = plt.subplots(figsize=(30,10))
# ax = counter_label.plot(kind='bar', ax=ax)
# plt.ylabel('Number of annotated calls', fontsize=30 )
# plt.xlabel('Labels', fontsize=30 )
# plt.xticks(rotation=0)
# plt.title('Number of detections of each annotator, classified by label type', fontsize=40 )
# ax.set_axisbelow(True)
# ax.yaxis.grid(color='gray', linestyle='dashed')
# ax.tick_params(labelsize=20)
# plt.legend(loc=2, prop={'size': 20});









