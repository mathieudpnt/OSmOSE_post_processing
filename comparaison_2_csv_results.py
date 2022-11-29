import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
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
    print(label, ' : ', len(df3),'/',len(df1),'kept annotations')
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
#%% Path file + TZ
#C3D5 - ST sur filières différentes
# FilePath1 = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 3/IROISE/335556632/analysis/C3D5/Results/24-11_084550/PG2Aplose table.csv'
# FilePath2 = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 3/IROISE/336363566/analysis/C3D5/Results/24-11_165233/PG2Aplose table.csv'

# C2D2 - 08/07/22 12:00 au 11/07/22 08:00 ST séparés de 500m
FilePath1 = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/analysis/C2D2/Results/29-11_122724/PG2Aplose table.csv'
FilePath2 = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/336363566/analysis/C2D2/Results/29-11_100634/PG2Aplose table.csv'

# WavPath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/wav'
# TimestampPath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/wav/timestamp_PG - C2D2.csv'

#C2D1 - 07/07/22 00:00 au 08/07 00:00 ST séparés de 500m
# FilePath1 = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/analysis/C2D1/Results/18-10_111429 whistles/PG2Aplose table.csv'
# FilePath2 = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/336363566/analysis/C2D1/Results/18-10_164453/PG2Aplose table.csv'

tz_data = 'Europe/Paris'
#%% User input
date_begin = pytz.timezone(tz_data).localize(pd.to_datetime(easygui.enterbox("datetime begin ? (dd MM yyyy HH mm ss) :"), format='%d %m %Y %H %M %S'))
date_end =   pytz.timezone(tz_data).localize(pd.to_datetime(easygui.enterbox("datetime end ? (dd MM yyyy HH mm ss) :"), format='%d %m %Y %H %M %S'))

duration_h = (date_end-date_begin).total_seconds()/3600
duration_min = duration_h * 60
if duration_h.is_integer() == True:
    duration_h = int(duration_h)
    print('duration : ', duration_h, 'h')
else: print('duration_h is not an integer')

if duration_min.is_integer() == True:
    duration_min = int(duration_min)
else: print('duration_min is not an integer')

# ts = get_duration(TimestampPath, WavPath, tz_data)  
# date_begin = hour_rounder(ts.timestamp[0])
# date_end = hour_rounder(ts['timestamp'].iloc[-1] + dt.timedelta(seconds = ts['duration'].iloc[-1]))
# duration_h = (date_end-date_begin).total_seconds()/3600
# duration_min = (date_end-date_begin).total_seconds()/60
# if duration_min.is_integer() == True:
#     duration_min = int(duration_min)
# else: print('duration_min is not an integer')

# test_ts = [(y >= date_begin) & (y <= date_end) for x,y in enumerate(ts['timestamp'])]
# if sum(test_ts) == 0: print('Aucun fichier compris entre', str(date_begin), 'et', str(date_end))
# else : print(sum(test_ts), '/', len(ts), 'fichiers compris entre', str(date_begin), 'et', str(date_end))

df1 = sorting_annot_boxes(FilePath1, tz_data, date_begin, date_end)
df2 = sorting_annot_boxes(FilePath2, tz_data, date_begin, date_end)

time_bin = list(set([max(df1['end_time']), max(df2['end_time'])]))
if len(time_bin) != 1 :
    print('time bins are different', time_bin)
else: 
    time_bin = time_bin[0]
    print("\ntime_bin : ", str(time_bin), "s")

annotators1 = df1['annotator'].drop_duplicates()
annotators2 = df2['annotator'].drop_duplicates()
annotators = pd.concat([annotators1, annotators2]).drop_duplicates().reset_index(drop=True);

labels1 = df1['annotation'].drop_duplicates()
labels2 = df2['annotation'].drop_duplicates()
labels = pd.concat([labels1, labels2]).drop_duplicates().reset_index(drop=True);


#%% Single plot 

if len(labels)>1:
    label_ref = easygui.buttonbox('Select a label', 'Plot label', labels)
elif len(annotators==1):
    label_ref = labels[0]

if len(annotators)>1:
    annot_ref = easygui.buttonbox('Select a label', 'Plot label', annotators)
elif len(annotators==1):
    annot_ref = annotators[0]

res_min, date_list = res_timebin_plot(date_begin, date_end, duration_min)   

time_slice = 60*res_min #10 min
n_annot_max = time_slice/time_bin #nb of annoted time_bin max per time_slice

df1_1a1l = df_1annot_1label(df1, annot_ref, label_ref)
df2_1a1l = df_1annot_1label(df2, annot_ref, label_ref)

lim1 = hour_rounder(max([df1_1a1l['start_datetime'][0], df2_1a1l['start_datetime'][0]]))
lim2 = hour_rounder(min([df1_1a1l['start_datetime'].iloc[-1], df2_1a1l['start_datetime'].iloc[-1]]))

fig, ax1 = plt.subplots(nrows = 1, figsize=(30,20))

label_legend = pd.concat([df1.dataset, df2.dataset]).drop_duplicates().reset_index(drop=True)


ax1.hist([df1_1a1l['start_datetime'],df2_1a1l['start_datetime']] , bins=date_list, label=[label_legend[0], label_legend[1]]); #histo annotation


bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))

ax1.set_yticks(y_pos, bars);
ax1.tick_params(labelsize=20)
ax1.set_ylabel("taux d'annotation / "+str(res_min)+" min", fontsize = 20)
if duration_h>24:
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M', tz=pytz.timezone(tz_data)))
    ax1.tick_params(axis='x', rotation= 80)
    
    fig.suptitle(annot_ref +' '+ label_ref + date_begin.strftime(' - %d/%m/%Y') + date_end.strftime(' - %d/%m/%y UTC%z')+'\nDuration : '+str(duration_h)+'h', fontsize = 24, y=0.95);
    
    # grey background on odd/even days
    date_odd_even = [j for i,j in enumerate(date_list) if j.day%2==0] #select odd or even days
    for i,j in enumerate(list(set([j.day for i,j in enumerate(date_odd_even)]))):
        vec = [l for k,l in enumerate(date_list) if l.day==j]
        if (vec[-1].hour == 23) : vec[-1] += dt.timedelta(hours=1)
        elif vec[-1].day == date_end.day: vec[-1] = date_end
        ax1.fill_between([vec[0], vec[-1]],n_annot_max, color='grey', alpha=0.075) 
    
else:
    fig.suptitle(annot_ref +' '+ label_ref + date_begin.strftime(' - %d/%m/%y UTC%z')+'\nDuration : '+str(duration_h)+'h', fontsize = 24, y=0.95);
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone(tz_data)))
    ax1.tick_params(axis='x', rotation= 45)

ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))



plt.ylim(0, n_annot_max)

ax1.set_xlim([date_begin, date_end])
# ax1.grid(color='k', linestyle='-', linewidth=0.2, axis='both')

ax1.legend(fontsize = 30)

#%% Single plot bis

if len(labels)>1:
    label_ref = easygui.buttonbox('Select a label', 'Plot label', labels)
elif len(labels==1):
    label_ref = labels[0]

if len(annotators)>1:
    annot_ref = easygui.buttonbox('Select a label', 'Plot label', annotators)
elif len(annotators==1):
    annot_ref = annotators[0]

res_min, date_list = res_timebin_plot(date_begin, date_end, duration_min)    
time_slice = 60*res_min #10 min
n_annot_max = time_slice/time_bin #nb of annoted time_bin max per time_slice

df1_1a1l = df_1annot_1label(df1, annot_ref, label_ref)
df2_1a1l = df_1annot_1label(df2, annot_ref, label_ref)


fig, (ax1, ax2) = plt.subplots(nrows = 2, figsize=(30,20))

ax1.hist(df1_1a1l['start_datetime'], bins=date_list); #histo annotation
ax2.hist(df2_1a1l['start_datetime'], bins=date_list); #histo annotation

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))

ax1.set_yticks(y_pos, bars);
ax2.set_yticks(y_pos, bars);
ax1.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)
ax1.set_ylabel("taux d'annotation / "+str(res_min)+" min", fontsize = 20)
ax2.set_ylabel("taux d'annotation / "+str(res_min)+" min", fontsize = 20)
ax1.tick_params(axis='y')
ax2.tick_params(axis='y')
fig.suptitle(annot_ref +' '+ label_ref + date_begin.strftime(' - %d/%m/%y UTC%z'), fontsize = 24, y=0.95);

ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone(tz_data)))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone(tz_data)))
# plt.xlim(date_begin, date_end)

ax1.set_xlim([date_begin, date_end])
ax2.set_xlim([date_begin, date_end])
ax1.grid(color='k', linestyle='-', linewidth=0.2, axis='both')
ax2.grid(color='k', linestyle='-', linewidth=0.2, axis='both')











