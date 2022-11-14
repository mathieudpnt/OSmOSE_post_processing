import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import calendar
import pytz
import easygui  # to install : conda install -c conda-forge easygui
import os
import contextlib
import wave
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



#%% Path file + TZ

FilePath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/analysis/C2D1/Aplose results APOCADO_IROISE_C2D1.csv'
TimestampPath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/analysis/C2D1/timestamp.csv'
WavPath = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/wav'

tz_data ='Europe/Paris'

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

time_bin = max(df1['end_time'])
print("\ntime_bin : ", time_bin, "s")

annotators = df1['annotator'].drop_duplicates()
print('\nannotators :\n',annotators.reset_index(drop=True).to_string())

labels = df1['annotation'].drop_duplicates()
print('\nlabels :\n',labels.reset_index(drop=True).to_string())


#%% Multilabel plot vs tides

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


    

