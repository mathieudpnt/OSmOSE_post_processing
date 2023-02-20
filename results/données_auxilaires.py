import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from collections import Counter
import matplotlib.dates as mdates
import calendar
from scipy.signal import savgol_filter
import pytz
import easygui  # to install : conda install -c conda-forge easygui
import os
import contextlib
import wave


import pickle
import statistics
import math 

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

# From a DataFrame, returns a Dataframe df with the annotations of 1 annotator and 1 label
def df_1annot_1label(df0, annotator, label):
    df1= df0.loc[df0['annotator'] == annotator]
    df2 = df1.loc[df1['annotation'] == label]
    df3 = df2.reset_index(drop=True) #reset the indexes of row after sorting the df
    df3['start_datetime'] = pd.to_datetime(df3.start_datetime, format='%d/%m/%Y %H:%M:%S%tz')
    df3['end_datetime'] = pd.to_datetime(df3.end_datetime, format='%d/%m/%Y %H:%M:%S%tz')
    print(label, ' : ', len(df3),'/',len(df1),'kept annotations')
    return df3

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

# From an Aplose results csv, returns a DataFrame without the Aplose box annotations
def sorting_annot_boxes(FileDet, tz, date_begin, date_end):
    df = pd.read_csv(FileDet)
    max_freq = max(df['end_frequency'])
    max_time = max(df['end_time'])
    df2 = df.loc[(df['start_time'] == 0) & (df['end_time'] == max_time) & (df['end_frequency'] == max_freq)] #deletion of boxes
    df3 = df2.sort_values('start_datetime') #sorting value according to datetime_start
    df4 = df3.reset_index(drop=True) #reset the indexes of row after sorting the df
    df4['start_datetime'] = pd.to_datetime(df4['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%Z')
    df4['end_datetime'] = pd.to_datetime(df4['end_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%Z')
    
    df4['start_datetime'] = [y.tz_convert(tz) for x,y in enumerate(df4['start_datetime'])] #converting to desired tz
    df4['end_datetime'] = [y.tz_convert(tz) for x,y in enumerate(df4['end_datetime'])] #converting to desired tz
    
    df5 = df4[(df4['start_datetime']>=date_begin) & (df4['start_datetime']<=date_end)] #select data within [date_begin;date_end]

    print(len(df5),'/',len(df4),'kept annotations')
    return (df5)

#Get Duration of files in timestamp.csv file
def get_duration(timestamp_file, wav_path, TimeZ):
    ts = pd.read_csv(timestamp_file, delimiter=',', header=None, names=['name', 'timestamp'])
    ts['timestamp'] = pd.to_datetime(ts['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ') #from str to naïve datetime
    ts['timestamp'] = [pytz.timezone(TimeZ).localize(y) for x,y in enumerate(ts['timestamp'])] #add timezone
    wavname = [os.path.join(wav_path, y) for x,y in enumerate(ts['name'])]
    list_duration=[]
    for x,y in enumerate(wavname):
        with contextlib.closing(wave.open(y,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            list_duration.append(duration)
    ts['duration'] = list_duration
    return ts
    
# Rounds to nearest hour by adding a timedelta hour if minute >= 30
def hour_rounder(t):
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

#returns the bins for a user-specified bin resolution used for an annotation plot
def res_timebin_plot(date_begin, date_end, duration_min):
    res_min = easygui.enterbox("résolution temporelle bin ? (min) :")
    if res_min.isnumeric() == False :
        print('Not an integer')
    else: 
        res_min = int(res_min)
        if duration_min%res_min == 0:
            date_list = [date_begin + dt.timedelta(minutes=res_min*x) for x in range(duration_min//res_min)] # 24*60 = 144*10 min in 24h
            return (res_min, date_list)
        else: print('\n\n /!\ duration_min/res_min is not an integer')
#%%TO DO LIST

#plot sun/night : plotté les annot/detec sur un seul x / jour + rendre la fonction + user-friendly + vérifier les TZ

###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
#DONE
#Loader timestamps.csv pour avoir la durée de la campagne et de chaque fichiers OU si pas de timestamp.csv demander suer date début et fin campagne
#créer fonction pour automatiser creation timebin, user choisi taille des bins


#%% Source data

# FilePath = 'L:/acoustock/Bioacoustique/DATASETS/CETIROISE/ANALYSE/220926/CETIROISE_HF 17072022.csv'
# TimestampPath = 'L:/acoustock/Bioacoustique/DATASETS/CETIROISE/DATA/B_Sud Fosse Ouessant/Phase_1/Sylence/2022-07-17/timestamp.csv'
# WavPath = 'L:/acoustock/Bioacoustique/DATASETS/CETIROISE/DATA/B_Sud Fosse Ouessant/Phase_1/Sylence/2022-07-17'

# FilePath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/analysis/C2D1/Aplose results APOCADO_IROISE_C2D1.csv'
# TimestampPath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/analysis/C2D1/timestamp.csv'
# WavPath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/wav'

FilePath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 3/IROISE/335556632/analysis/PG Binary C3D5 - Results/23-11_090454/PG2Aplose table.csv'
TimestampPath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 3/IROISE/335556632/wav/C3D5/timestamp_PG.csv'
WavPath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 3/IROISE/335556632/wav/C3D5'

tz_data ='Europe/Paris'

# file_maree = 'L:/acoustock/Bioacoustique/DATASETS/CETIROISE/maregraphie/152_2022.csv'
file_maree = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/maregraphie/6305_2022.csv'

# pkl_filename = 'C:/Users/dupontma2/Downloads/complete_welch.pkl'
# pkl_filename = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/analysis/C2D1/complete_welch.pkl'


#%% User input
date_begin = pytz.timezone(tz_data).localize(pd.to_datetime(easygui.enterbox("datetime begin ? (yyyy MM dd HH mm ss) :"), format='%Y %m %d %H %M %S'))
date_end =   pytz.timezone(tz_data).localize(pd.to_datetime(easygui.enterbox("datetime end ? (yyyy MM dd HH mm ss) :"), format='%Y %m %d %H %M %S'))

ts = get_duration(TimestampPath, WavPath, tz_data)  

test_ts = [(y > date_begin) & (y < date_end) for x,y in enumerate(ts['timestamp'])]
if sum(test_ts) == 0: print('Aucun fichier compris entre', str(date_begin), 'et', str(date_end))
else : print(sum(test_ts), '/', len(ts), 'fichiers compris entre', str(date_begin), 'et', str(date_end))
#%%
duration_h = (date_end-date_begin).total_seconds()/3600
duration_min = duration_h * 60
if duration_h.is_integer() == True:
    duration_h = int(duration_h)
    print('duration : ', duration_h, 'h')
else: print('duration_h is not an integer')

if duration_min.is_integer() == True:
    duration_min = int(duration_min)
else: print('duration_min is not an integer')

df1 = sorting_annot_boxes(FilePath, tz_data, date_begin, date_end)
pd.read_csv(FilePath)

time_bin = max(df1['end_time'])
print("\ntime_bin : ", time_bin, "s")

annotators = df1['annotator'].drop_duplicates()
print('\nannotators :\n',annotators.reset_index(drop=True).to_string())

labels = df1['annotation'].drop_duplicates()
print('\nlabels :\n',labels.reset_index(drop=True).to_string())


#%% Plot annotations of each annotator
counter_label = df1.groupby('annotation')['annotator'].apply(Counter).unstack(fill_value=0)
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


label_ref = easygui.buttonbox('Select a reference label', '', labels)

if len(annotators)>1:
    annot_ref = easygui.buttonbox('Select a  reference annotator', '', annotators)
elif len(annotators==1):
    annot_ref = annotators[0]


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
    print("\nComparaison entre l'annotateur", annot_ref, "et l'annotateur", x)
    print('Nombre de détections en commun :', np.sum(Annot_detect))
    print("Nombre de détections manquées par l'autre annotateur :", Missed_det[num_annot])
    print("Nombre de détections que l'autre annotateur a, mais pas vous :", False_alarm[num_annot])

#%% Single plot vs tide
dt_maree, h_maree, h_max = plot_tides(file_maree, date_begin, date_end, tz_data)

label_ref = easygui.buttonbox('Select a label', 'Plot label', labels)

if len(annotators)>1:
    annot_ref = easygui.buttonbox('Select a label', 'Plot label', annotators)
elif len(annotators==1):
    annot_ref = annotators[0]


res_min, date_list = res_timebin_plot(date_begin, date_end, duration_min)    
time_slice = 60*res_min #10 min
n_annot_max = time_slice/time_bin #nb of annoted time_bin max per time_slice

df_1a1l = df_1annot_1label(df1, annot_ref, label_ref)

fig,ax = plt.subplots(figsize=(20,9))
ax2 = ax.twinx()

data_hist = ax.hist(df_1a1l['start_datetime'], bins=date_list, color='coral'); #histo annotation
ax2.plot(dt_maree,h_maree, color='royalblue') #plot marée

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))
ax.set_yticks(y_pos, bars);
ax.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)
ax.set_ylabel("taux d'annotation / "+str(res_min)+" min", fontsize = 20, color='coral')

ax.tick_params(colors='coral',axis='y')
ax2.set_ylabel("hauteur d'eau (m)", fontsize = 20, color='royalblue')
ax2.tick_params(colors='royalblue',axis='y')
fig.suptitle(annot_ref +' '+ label_ref + date_begin.strftime(' - %d/%m/%y UTC%z'), fontsize = 24, y=0.95);

ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone(tz_data)))
plt.xlim(date_begin, date_end)
ax.grid(color='k', linestyle='-', linewidth=0.2, axis='both')

#%% Multilabel plot vs tides
dt_maree, h_maree, h_max = plot_tides(file_maree, date_begin, date_end, tz_data)

if len(annotators)>1:
    annot_ref = easygui.buttonbox('Select a label', 'Plot label', annotators)
elif len(annotators==1):
    annot_ref = annotators[0]

selected_labels = labels[0:3] #TODO : checkbox to select desired labels to plot ?

res_min, date_list = res_timebin_plot(date_begin, date_end, duration_min)    
time_slice = 60*res_min #10 min
n_annot_max = time_slice/time_bin #nb of annoted time_bin max per time_slice

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))

locator = mdates.HourLocator(interval=2)

fig, ax = plt.subplots(nrows = len(selected_labels), figsize=(30,20))

plt.setp(ax, xlim=(date_begin,date_end))
fig.suptitle('Annotations de '+annot_ref +' du' + date_begin.strftime(' %d/%m/%y UTC%z'), fontsize = 24, y=0.95)

x_hist=[]
y_hist=[]
for i, L in enumerate(selected_labels):
    ax2 = ax[i].twinx()
    
    annot_whistlesM = df_1annot_1label(df1, annot_ref, L)  
    df_timestamp_beg = annot_whistlesM['start_datetime']
    t_dt=pd.to_datetime(df_timestamp_beg, format="%Y-%m-%dT%H:%M:%S.%f%z")
    
    n1, n2, obj = ax[i].hist(t_dt, date_list, color='tab:red') 
    x_hist.append(n1)
    y_hist.append(n2)
    
    

    ax[i].xaxis.set_major_locator(locator)
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone(tz_data)))

    ax[i].set_title(L, fontsize = 20)
    ax[i].tick_params(labelsize=20)
    ax[i].set_yticks(y_pos)
    ax[i].set_yticklabels(bars)
    ax[i].grid(color='k', linestyle='-', linewidth=0.2)
    ax[i].set_xlim(date_begin, date_end)

    ax[i].set_ylabel("taux d'annotation / "+str(res_min)+" min", fontsize = 20, color='tab:red')
    ax[i].tick_params(colors='tab:red', axis='y')
    ax[i].grid(color='k', linestyle='-', linewidth=0.2, axis='both')
    
    ax2.plot(dt_maree,h_maree, color='royalblue') #plot marée
    ax2.set_ylabel("hauteur d'eau (m)", fontsize = 20, color='royalblue')
    ax2.tick_params(colors='royalblue',axis='y')

#%% Noise
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
    datetime_welch = [(dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")) for x in time_welch]
    # datetime_welch = [(dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")) for x in time_welch]
    return welch, datetime_welch

def plot_noise(tz_data, date_begin, date_end, pkl_filename):
    pkl_filename = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/analysis/C2D1/complete_welch.pkl'
    welch, time_welch = read_pkl(pkl_filename)
    time_welch = [pytz.timezone(tz_data).localize(y) for x,y in enumerate(time_welch)] #add timezone
    df_welch = pd.DataFrame()
    df_welch['welch'] = welch
    df_welch['time'] = time_welch
    df_welch[(df_welch['time']>= date_begin) & (df_welch['time']<= date_end)]
    welch_dB = [10*np.log(SPL) for SPL in df_welch['welch']]
    average_SPL = [statistics.mean(W) for W in welch_dB]
    average_SPL_f = savgol_filter(average_SPL, 201, 2)
    df_welch['SPL_av'] = average_SPL_f
    return df_welch
    
    
#%%
df_w = plot_noise(tz_data, date_begin, date_end, pkl_filename)

label_ref = easygui.buttonbox('Select a label', 'Plot label', labels)

if len(annotators)>1:
    annot_ref = easygui.buttonbox('Select a label', 'Plot label', annotators)
elif len(annotators==1):
    annot_ref = annotators[0]


res_min, date_list = res_timebin_plot(date_begin, date_end, duration_min)    
time_slice = 60*res_min #10 min
n_annot_max = time_slice/time_bin #nb of annoted time_bin max per time_slice

df_1a1l = df_1annot_1label(df1, annot_ref, label_ref)

fig,ax = plt.subplots(figsize=(20,9))
ax2 = ax.twinx()

ax.hist(df_1a1l['start_datetime'], bins=date_list, color='coral'); #histo annotation
ax2.plot(df_w['time'], df_w['SPL_av'], color='royalblue') #plot noise

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))
ax.set_yticks(y_pos, bars);
ax.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)
ax.set_ylabel("taux d'annotation / "+str(res_min)+" min", fontsize = 20, color='coral')

ax.tick_params(colors='coral',axis='y')
ax2.set_ylabel("average SPL (dB)", fontsize = 20, color='royalblue')
ax2.tick_params(colors='royalblue',axis='y')
fig.suptitle(annot_ref +' '+ label_ref + date_begin.strftime(' - %d/%m/%y UTC%z'), fontsize = 24, y=0.95);

ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone(tz_data)))
plt.xlim(date_begin, date_end)
ax.grid(color='k', linestyle='-', linewidth=0.2, axis='both')


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
ax.xaxis.set_major_formatter(mdates.DateFormatter('%D', tz=pytz.timezone(tz_data)))
ax.grid(color='k', linestyle='-', linewidth=0.2)