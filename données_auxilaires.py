import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from collections import Counter
import matplotlib.dates as mdates
import calendar
from scipy.signal import savgol_filter
import pytz

import pickle
import statistics

from astral.sun import sun
import astral
#%%

# Fonctions utilisées dans 
# Return list of labels and the names of the files that contain this label
def list_class(FileDet):
    # FileDet : file name of the csv file that contains the annotations
    # df : DataFrame of the csv file
    # categories : list of labels used for annotation
    # wav : list of wavfile names
    # annotation : list of annotations
    df = pd.read_csv(FileDet)
    annotation = df['annotation'] # list of call types of each detection

    wav  = df['filename'] # list of filename in which are each detection
    categories=np.unique(annotation)
    print(categories)
    return(df, categories, wav, annotation)



# Convert detection dates in unix format
def time_det(call_type, annotation,wavFile):
    
    # call_type : list of annotation labels
    # annotation : list of all annotations as read in the result csv downloaded from Aplose 
    # UT : list of unix timestamp of each annotation of the 'call_type' label
    time_det = []
    UT = []
    for j, detection in enumerate(annotation):

        if detection == call_type:
            w = wavFile[j]
            date_det = (w.split('_')[2])
            hour_det = (w.split('_')[3])
            time_det.append(date_det + hour_det)
            t = date_det + hour_det
            unixT =calendar.timegm(dt.datetime.strptime(t, '%d%m%y%H%M%S').timetuple())
            UT.append(unixT)
    return UT

# Return a Dataframe df with the annotations of 1 annotator and 1 label
def df_1annot_1label(df0, annotator, label):
    df1= df0.loc[df0['annotator'] == annotator]
    df2 = df1.loc[df1['annotation'] == label]
    df3 = df2.reset_index(drop=True) #reset the indexes of row after sorting the df
    df3['start_datetime'] = pd.to_datetime(df3.start_datetime, format='%d/%m/%Y %H:%M:%S%tz')
    df3['end_datetime'] = pd.to_datetime(df3.end_datetime, format='%d/%m/%Y %H:%M:%S%tz')
    return df3

#secondary y axis for tide charts
def annot_secondyaxis(x):
    return (x/n_annot_max)*h_max
def annot_secondyaxis2(x):
    return (x/h_max)*n_annot_max

# Returns 2 lists containing the start and end datetime of each annotations of one label & one annotator
def CreatVec_datetime_det(df_results, annotator, label_test):
    # Create a DataFrame containing the lines corresponding to the 'label' annotations of 'annot_ref'
    det_annot_ref = df_results.loc[df_results['annotator'].isin([annotator])]
    det_annot_ref_label = det_annot_ref.loc[det_annot_ref['annotation'].isin([label_test])]
    # print("Number of annotations of reference annotator: ", len(det_annot_ref_label))
    # Read the start and end timestamp of each detection of 'annot_ref'
    beg_det_ref = [calendar.timegm(L.timetuple()) for L in det_annot_ref_label['start_datetime']]
    end_det_ref = [calendar.timegm(L.timetuple()) for L in det_annot_ref_label['end_datetime']]
    return beg_det_ref, end_det_ref

def sorting_annot_boxes(FileDet, tz):
    df = pd.read_csv(FilePath1)
    max_freq = max(df['end_frequency'])
    max_time = max(df['end_time'])
    df2 = df.loc[(df['start_time'] == 0) & (df['end_time'] == max_time) & (df['end_frequency'] == max_freq)] #deletion of boxes
    df3 = df2.sort_values('start_datetime') #sorting value according to datetime_start
    df4 = df3.reset_index(drop=True) #reset the indexes of row after sorting the df
    df4['start_datetime'] = pd.to_datetime(df4['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%Z')
    df4['end_datetime'] = pd.to_datetime(df4['end_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%Z')
    
    df4['start_datetime'] = [y.tz_convert(tz) for x,y in enumerate(df4['start_datetime'])] #converting to desired tz
    df4['end_datetime'] = [y.tz_convert(tz) for x,y in enumerate(df4['end_datetime'])] #converting to desired tz

    return (df4)

def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+dt.timedelta(hours=t.minute//30))

def plot_tides(data_path, date_begin, date_end, tz):
    df_maree = pd.read_csv(file_maree, skiprows=13, delimiter=';')
    df_maree['# Date'] = pd.to_datetime(df_maree['# Date'], format='%d/%m/%Y %H:%M:%S')
    df_maree = df_maree[df_maree['Source']==1] #Only data from Source==1
    df_maree = df_maree.drop(['Source'], axis=1) #'Source' column useless
    df_maree['# Date'] = df_maree['# Date'].dt.tz_localize('UTC') #Timezone of source data
    df_maree['# Date'] = df_maree['# Date'].dt.tz_convert(tz) #Change of TZ
    df2_maree = df_maree[(df_maree['# Date'] <= date_end) & (df_maree['# Date'] >= date_begin)] #sorting
    df2_maree = df2_maree.reset_index(drop=True) #reset the indexes of row after sorting the df
    
    dt_maree = df2_maree['# Date']
    hauteur = df2_maree['Valeur']
    hauteur2 = pd.Series(savgol_filter(hauteur, 301, 4)) #filtering
    h_max = max(hauteur2)
    return (dt_maree, hauteur2, h_max)

def read_pkl(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        pkl = pickle.load(f)

    welch = pkl[0]
    time_welch = pkl[1]
    # On trie les welch, car ils ne sont pas rangés dans l'odre dans le fichier pkl
    a = np.argsort(time_welch)
    #np.take_along_axis(welch, a, axis=1)
    welch = welch[a]
    time_welch.sort()
    datetime_welch = [(dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")) for x in time_welch]
    return welch, datetime_welch
#%%TO DO LIST

#créer fonction pour automatiser creation timebin, user choisi taille des bins

#Loader timestamps.csv pour avoir la durée de la campagne et de chaque fichiers OU si pas de timestamp.csv demander suer date début et fin campagane

#plot sun/night : plotté les annot/detec sur un seul x / jour + rendre la fonction + user-friendly + vérifier les TZ
#%%
# FilePath1 = 'L:/acoustock/Bioacoustique/DATASETS/CETIROISE/ANALYSE/220926/CETIROISE_HF 17072022.csv'
FilePath1 = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/336363566/analysis/C2D1/APOCADO_IROISE_C2D1 - results.csv'

tz_data = 'Europe/Paris'

df1 = sorting_annot_boxes(FilePath1, tz_data)

date_begin = df1['start_datetime'][0]
date_end = df1['end_datetime'][len(df1)-1]

print('first start_datetime : ', date_begin)
print('last start_datetime : ', date_end)

date_begin_rounded = hour_rounder(date_begin)
date_end_rounded = hour_rounder(date_end)

date_list = [date_begin_rounded + dt.timedelta(minutes=10*x) for x in range(145)] # 24*60 = 144*10 min in 24h

print('first start_datetime rounded : ', date_begin_rounded)
print('last start_datetime rounded : ', date_end_rounded)

#%%
time_bin = max(df1['end_time'])

print("time_bin : ", time_bin, "s")
#%%
duration = date_end_rounded - date_begin_rounded
duration_h = duration.total_seconds()/3600
print('duration : ', duration_h, 'h')

#%%
annotators = df1['annotator'].unique()
labels = df1['annotation'].unique()

print('annotators : ',annotators,'\nlabels :', labels)

#%%
label = 'Odontocete whistles'
annotator = 'mdupon'

df2 = df_1annot_1label(df1, annotator, label)
print("Il y a", df2.shape[0], "annotations de", label, "par l'annotateur", annotator)

#%%
# Compute a dataframe containing the number of annotations per annotator and per label
counter_label = df1.groupby('annotation')['annotator'].apply(Counter).unstack(fill_value=0)
print(counter_label)

#%%
# Plot annotations of each annotator
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

#%%
annot_ref = 'mdupon'
label_ref = 'Odontocete whistles'
True_det = []
#Detect_rate = []
False_alarm = []
Missed_det = []
# Read the start and end timestamp of each detection of 'annot_ref'
beg_det_ref, end_det_ref = CreatVec_datetime_det(df1, annot_ref, label_ref)

# Browse every other annotator (other than 'annot_ref')
for num_annot, x in enumerate(annotators):
    # Read the start and end timestamp of each detection of annotator 'x'
    beg_det, end_det = CreatVec_datetime_det(df1, x, label_ref)
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

#%% Single plot vs tide
file_maree = 'L:/acoustock/Bioacoustique/DATASETS/CETIROISE/maregraphie/152_2022.csv'
tz_tides = tz_data

(dt_maree, h_maree, h_max) = plot_tides(file_maree, date_begin_rounded, date_end_rounded, tz_tides)

# fig, ax = plt.subplots(figsize=(16,4))
# plt.plot(dt_maree, h_maree)
# plt.setp(ax, ylim=(0,1.2*h_max));
# locator = mdates.HourLocator(interval=2)
# ax.xaxis.set_major_locator(locator)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone(tz_tides)))
# ax.grid(color='k', linestyle='-', linewidth=0.2, axis='both')
# fig.suptitle('Hauteur d\'eau du ' + date_begin_rounded.strftime(' %d/%m/%y UTC%z'), fontsize = 12)

annotator = 'mdupon'
label = 'Odontocete clics'
(dt_maree, h_maree, h_max) = plot_tides(file_maree, date_begin_rounded, date_end_rounded, tz_data)

res = 10 #nb minutes
time_slice = 60*res #10 min
n_annot_max = time_slice/time_bin #nb of annoted time_bin max per time_slice

df_1a1l = df_1annot_1label(df1, annotator, label)
tz_inf = df_1a1l['start_datetime'].iloc[0].tzinfo


fig,ax = plt.subplots(figsize=(20,9))
ax2 = ax.twinx()

ax.hist(df_1a1l['start_datetime'], bins=date_list, color='coral'); #histo annotation
ax2.plot(dt_maree,h_maree, color='royalblue') #plot marée

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))
ax.set_yticks(y_pos, bars);
ax.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)
ax.set_ylabel("taux d'annotation / 10min", fontsize = 20, color='coral')
ax.tick_params(colors='coral',axis='y')
ax2.set_ylabel("hauteur d'eau (m)", fontsize = 20, color='royalblue')
ax2.tick_params(colors='royalblue',axis='y')
fig.suptitle(label + date_begin_rounded.strftime(' - %d/%m/%y UTC%z'), fontsize = 24);

ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz_inf))
plt.xlim(date_begin_rounded, date_end_rounded)
ax.grid(color='k', linestyle='-', linewidth=0.2, axis='both')

#%%
#Multilabel plot vs tides
annotator = 'mdupon';
file_maree = 'L:/acoustock/Bioacoustique/DATASETS/CETIROISE/maregraphie/152_2022.csv'
tz_tides = tz_data


(dt_maree, h_maree, h_max) = plot_tides(file_maree, date_begin_rounded, date_end_rounded, tz_tides)

res = 10
time_slice = 60*res #10 min
n_annot_max = time_slice/time_bin #n of annoted time_bin max per time_slice

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))

locator = mdates.HourLocator(interval=2)

fig, ax = plt.subplots(nrows = 3, figsize=(30,20))

plt.setp(ax, xlim=(date_begin,date_end))
fig.suptitle('Annotations de '+annotator +' du' + date_begin_rounded.strftime(' %d/%m/%y UTC%z'), fontsize = 24, y=0.93)

for i, L in enumerate(labels[0:3]):
    ax2 = ax[i].twinx()
    
    annot_whistlesM = df_1annot_1label(df1, annotator, L)  
    df_timestamp_beg = annot_whistlesM['start_datetime']
    t_dt=pd.to_datetime(df_timestamp_beg, format="%Y-%m-%dT%H:%M:%S.%f%z")
    
    ax[i].hist(t_dt, date_list, color='tab:red') 

    ax[i].xaxis.set_major_locator(locator)
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone(tz_data)))

    ax[i].set_title(L, fontsize = 20)
    ax[i].tick_params(labelsize=20)
    ax[i].set_yticks(y_pos)
    ax[i].set_yticklabels(bars)
    ax[i].grid(color='k', linestyle='-', linewidth=0.2)
    ax[i].set_xlim(date_begin_rounded, date_end_rounded)

    ax[i].set_ylabel("taux d'annotation / "+str(res)+" min", fontsize = 20, color='tab:red')
    ax[i].tick_params(colors='tab:red', axis='y')
    ax[i].grid(color='k', linestyle='-', linewidth=0.2, axis='both')
    
    ax2.plot(dt_maree,h_maree, color='royalblue') #plot marée
    ax2.set_ylabel("hauteur d'eau (m)", fontsize = 20, color='royalblue')
    ax2.tick_params(colors='royalblue',axis='y')

#%% Noise
pkl_filename = 'C:/Users/dupontma2/Downloads/complete_welch.pkl'
(welch, time_welch) = read_pkl(pkl_filename)
welch_dB = [10*np.log(SPL/10e-10) for SPL in welch]
average_SPL = [statistics.mean(W) for W in welch_dB]

average_SPL_f = savgol_filter(average_SPL, 131, 2) # window size 131, polynomial order 2
#average_SPL_dB = [20*math.log(average_SPL[i]/10e-5, 10) for i in range(0,len(SPL))]

# fig, ax = plt.subplots(figsize=(20,5))
# plt.plot(time_welch, average_SPL)
# plt.plot(time_welch, dB_SPL_f)
# locator = mdates.HourLocator(interval=2)
# ax.xaxis.set_major_locator(locator)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# ax.grid(color='k', linestyle='-', linewidth=0.2)

annotator = 'mdupon'

res = 10
time_slice = 60*res #10 min
n_annot_max = time_slice/time_bin #n of annoted time_bin max per time_slice

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))

locator = mdates.HourLocator(interval=2)

fig, ax = plt.subplots(nrows = 3, figsize=(30,20))

plt.setp(ax, xlim=(date_begin,date_end))
fig.suptitle('Annotations de '+annotator +' du' + date_begin_rounded.strftime(' %d/%m/%y UTC%z'), fontsize = 24, y=0.93)

for i, L in enumerate(labels[0:3]):
    ax2 = ax[i].twinx()
    
    annot_whistlesM = df_1annot_1label(df1, annotator, L)  
    df_timestamp_beg = annot_whistlesM['start_datetime']
    t_dt=pd.to_datetime(df_timestamp_beg, format="%Y-%m-%dT%H:%M:%S.%f%z")
    
    ax[i].hist(t_dt, date_list, color='tab:red') 

    ax[i].xaxis.set_major_locator(locator)
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone(tz_data)))

    ax[i].set_title(L, fontsize = 20)
    ax[i].tick_params(labelsize=20)
    ax[i].set_yticks(y_pos)
    ax[i].set_yticklabels(bars)
    ax[i].grid(color='k', linestyle='-', linewidth=0.2)
    ax[i].set_xlim(date_begin_rounded, date_end_rounded)

    ax[i].set_ylabel("taux d'annotation / "+str(res)+" min", fontsize = 20, color='tab:red')
    ax[i].tick_params(colors='tab:red', axis='y')
    ax[i].grid(color='k', linestyle='-', linewidth=0.2, axis='both')
    
    ax2.plot(time_welch, average_SPL_f, color='tab:green') #plot noise
    ax2.set_ylabel("Noise (dB)", fontsize = 20, color='tab:green')
    ax2.tick_params(colors='tab:green',axis='y')

#%%
# Fonction qui permet d'obtenir l'heure de lever et de coucher du soleil selon la position GPS
def suntime_hour(date_beg, date_end, timeZ, lat,lon,tz):
    # Infos sur la localisation
    gps = astral.LocationInfo(timezone=timeZ,latitude=lat, longitude=lon)
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

def CreatVec_datetime_det(df_results, annotator, label):

    # Create a DataFrame containing the lines corresponding to the 'label' annotations of 'annot_ref'
    det_annot_ref = df_results.loc[df_results['annotator'].isin([annotators])]  
    det_annot_ref_label = det_annot_ref.loc[det_annot_ref['annotation'].isin([labels])]
    # print("Number of annotations of reference annotator: ", len(det_annot_ref_label))

    # Read the start and end timestamp of each detection of 'annot_ref' 
    beg_det_struc_time = [(dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f+00:00").timetuple()) for x in det_annot_ref_label['start_datetime']]
    beg_det_datetime = [(dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f+00:00")) for x in det_annot_ref_label['start_datetime']]
    #end_det_ref = [(dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f+00:00")) for x in det_annot_ref_label['end_datetime']]

    return beg_det_struc_time, beg_det_datetime

#%%

annot = 'mdupon'
label = 'Odontocete clics'

date_begin_rounded_str = '2022-06-07'
date_end_rounded_str = '2022-08-07'

# Read the start timestamp of each detection of 'annot_ref' 
df2 = df_1annot_1label(df1, annot, label)


det_annot_ref = df2.loc[df2['annotator'].isin(annotators)]
det_annot_ref_label = df2.loc[det_annot_ref['annotation'].isin(labels)]

beg_det_struct_time = [x.timetuple() for x in det_annot_ref_label['start_datetime']]
beg_det_datetime = det_annot_ref_label['start_datetime']

#beg_det_struct_time, beg_det_datetime = CreatVec_datetime_det(df2, annot, label)


Day_det = beg_det_datetime
Hour_det = [x.tm_hour + x.tm_min/60 for x in beg_det_struct_time]

x_data = np.arange(date_begin_rounded_str, date_end_rounded_str, dtype="M8[D]")

lat = "47°59'N"
lon = "4°41'W"
astral.Depression = 12

[hour_sunrise, hour_sunset] = suntime_hour(date_begin_rounded_str, date_end_rounded_str, 'Europe/Paris', lat,lon,2)


# Plot figure
fig, ax = plt.subplots(figsize=(20,10))
plt.plot(x_data,hour_sunrise, color='k')
plt.plot(x_data,hour_sunset, color='k')
plt.scatter(Day_det,Hour_det)
locator = mdates.DayLocator(interval=7)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%D'))
ax.grid(color='k', linestyle='-', linewidth=0.2)