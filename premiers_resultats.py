import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from collections import Counter
import matplotlib.dates as mdates
import pytz


#%%

# From an Aplose results csv, returns a DataFrame without the Aplose box annotations
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

# From a DataFrame, returns a Dataframe df with the annotations of 1 annotator and 1 label
def df_1annot_1label(df0, annotator, label):
    df1= df0.loc[df0['annotator'] == annotator]
    df2 = df1.loc[df1['annotation'] == label]
    df3 = df2.reset_index(drop=True) #reset the indexes of row after sorting the df
    df3['start_datetime'] = pd.to_datetime(df3.start_datetime, format='%d/%m/%Y %H:%M:%S%tz')
    df3['end_datetime'] = pd.to_datetime(df3.end_datetime, format='%d/%m/%Y %H:%M:%S%tz')
    return df3



# Rounds to nearest hour by adding a timedelta hour if minute >= 30
def hour_rounder(t):
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+dt.timedelta(hours=t.minute//30))

#%%
# FilePath = 'L:/acoustock/Bioacoustique/DATASETS/CETIROISE/ANALYSE/220926/CETIROISE_HF 17072022.csv'
FilePath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/336363566/analysis/C2D1/APOCADO_IROISE_C2D1 - results.csv'
TimestampPath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/336363566/wav/timestamp.csv'

ts = pd.read_csv(TimestampPath, delimiter=',', header=None, names=['name', 'timestamp'])

ts['timestamp']
print(ts)

#%%
tz_data = 'Europe/Paris'

df1 = sorting_annot_boxes(FilePath, tz_data)

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
    # idx_common_det= np.zeros((1,nDetect))
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