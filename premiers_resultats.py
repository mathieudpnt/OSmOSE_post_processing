import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from collections import Counter
import matplotlib.dates as mdates
import pytz
import os
import wave
import contextlib
import easygui  
import calendar
#%%

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

# From a DataFrame, returns a Dataframe df with the annotations of 1 annotator and 1 label
def df_1annot_1label(df0, annotator, label):
    df1= df0.loc[df0['annotator'] == annotator]
    df2 = df1.loc[df1['annotation'] == label]
    df3 = df2.reset_index(drop=True) #reset the indexes of row after sorting the df
    df3['start_datetime'] = pd.to_datetime(df3.start_datetime, format='%d/%m/%Y %H:%M:%S%tz')
    df3['end_datetime'] = pd.to_datetime(df3.end_datetime, format='%d/%m/%Y %H:%M:%S%tz')
    print(len(df3),'/',len(df1),'kept annotations')
    return df3



# Rounds to nearest hour by adding a timedelta hour if minute >= 30
def hour_rounder(t):
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+dt.timedelta(hours=t.minute//30))

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
#%% Path file
# FilePath = 'L:/acoustock/Bioacoustique/DATASETS/CETIROISE/ANALYSE/220926/CETIROISE_HF 17072022.csv'
FilePath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/analysis/C2D1/Aplose results APOCADO_IROISE_C2D1.csv'
TimestampPath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/analysis/C2D1/timestamp.csv'
WavPath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/wav'
tz_data ='Europe/Paris'

#%% User input
date_begin = pytz.timezone(tz_data).localize(pd.to_datetime(easygui.enterbox("datetime begin ? (dd MM yyyy HH mm ss) :"), format='%Y %m %d %H %M %S'))
date_end =   pytz.timezone(tz_data).localize(pd.to_datetime(easygui.enterbox("datetime end ? (dd MM yyyy HH mm ss) :"), format='%Y %m %d %H %M %S'))

ts = get_duration(TimestampPath, WavPath, tz_data)  
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

#%%
df1 = sorting_annot_boxes(FilePath, tz_data, date_begin, date_end)

res_min = 10
if duration_min%res_min == 0:
    date_list = [date_begin + dt.timedelta(minutes=res_min*x) for x in range(duration_min//res_min)] # 24*60 = 144*10 min in 24h
else: print('\n\n /!\ duration_min/res_min is not an integer')

time_bin = max(df1['end_time'])
print("\ntime_bin : ", time_bin, "s")

annotators = df1['annotator'].drop_duplicates()
print('\nannotators :\n',annotators.reset_index(drop=True).to_string())

labels = df1['annotation'].drop_duplicates()
print('\nlabels :\n',labels.reset_index(drop=True).to_string())



#%%
# Plot annotations of each annotator
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

#%%
label_ref = easygui.buttonbox('Select a reference label', '', labels)

if len(annotators)>1:
    label_ref = easygui.buttonbox('Select a  reference label', '', annotators)
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
    print("Comparaison entre l'annotateur", annot_ref, "et l'annotateur", x)
    print('Nombre de détections en commun :', np.sum(Annot_detect))
    print("Nombre de détections manquées par l'autre annotateur :", Missed_det[num_annot])
    print("Nombre de détections que l'autre annotateur a, mais pas vous :", False_alarm[num_annot])

#%% Single plot 

label_ref = easygui.buttonbox('Select a label', 'Plot label', labels)

if len(annotators)>1:
    label_ref = easygui.buttonbox('Select a label', 'Plot label', annotators)
elif len(annotators==1):
    annot_ref = annotators[0]


time_slice = 60*res_min #10 min
n_annot_max = time_slice/time_bin #nb of annoted time_bin max per time_slice

df_1a1l = df_1annot_1label(df1, annot_ref, label_ref)


fig,ax = plt.subplots(figsize=(20,9))

ax.hist(df_1a1l['start_datetime'], bins=date_list); #histo annotation

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))
ax.set_yticks(y_pos, bars);
ax.tick_params(labelsize=20)
ax.set_ylabel("taux d'annotation / 10min", fontsize = 20)
ax.tick_params(axis='y')
fig.suptitle(label_ref + date_begin.strftime(' - %d/%m/%y UTC%z'), fontsize = 24);

ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone(tz_data)))
plt.xlim(date_begin, date_end)
ax.grid(color='k', linestyle='-', linewidth=0.2, axis='both')

#%% Multilabel plot

if len(annotators)>1:
    annotator_ref = easygui.buttonbox('Select a label', 'Plot label', annotators)
elif len(annotators==1):
    annotator_ref = annotators[0]


res = 10
time_slice = 60*res #10 min
n_annot_max = time_slice/time_bin #n of annoted time_bin max per time_slice

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))

locator = mdates.HourLocator(interval=2)

fig, ax = plt.subplots(nrows = 3, figsize=(30,20))

plt.setp(ax, xlim=(date_begin,date_end))
fig.suptitle('Annotations de '+annotator_ref +' du' + date_begin.strftime(' %d/%m/%y UTC%z'), fontsize = 24, y=0.93)

for i, L in enumerate(labels[0:3]):    
    annot_whistlesM = df_1annot_1label(df1, annotator_ref, L)  
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
    ax[i].set_xlim(date_begin, date_end)

    ax[i].set_ylabel("taux d'annotation / "+str(res)+" min", fontsize = 20)
    ax[i].tick_params(axis='y')
    ax[i].grid(color='k', linestyle='-', linewidth=0.2, axis='both')
