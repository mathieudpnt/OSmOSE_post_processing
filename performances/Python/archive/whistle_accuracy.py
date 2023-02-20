from tqdm import tqdm
import os
import datetime as dt
import pytz
import glob
from tkinter import filedialog
from tkinter import Tk
import pandas as pd
import sys
import numpy as np
import time
sys.path.append('U:/Documents/Git/spyder_scripts')
from def_func import read_header, extract_datetime, from_str2dt, from_str2ts, sorting_annot_boxes, t_rounder


#%% LOAD DATA - User inputs

print('\n\nLoading data...', end='')

tz_data='Europe/Paris'
Ap=0

#PAMGuard detections
root = Tk()
root.withdraw()
pamguard_path = filedialog.askopenfilename(title="Select PAMGuard detection file", filetypes=[("CSV files", "*.csv")])
dfpamguard = pd.read_csv(pamguard_path).sort_values('start_datetime')

#APLOSE annotations
if Ap == 1:
    root = Tk()
    root.withdraw()
    aplose_path = filedialog.askopenfilename(title="Select APLOSE annotation file", filetypes=[("CSV files", "*.csv")])

    
#WAV files 
root = Tk()
root.withdraw()
wavpath = filedialog.askdirectory(title = 'Select wav folder')
wav_files = glob.glob(os.path.join(wavpath, "**/*.wav"), recursive=True)
wav_list = [os.path.basename(file) for file in wav_files]
wav_folder = [os.path.dirname(file) for file in wav_files]
durations = [read_header(file)[-1] for file in wav_files]

print('\tDone!', end='\n')

#%% FORMAT DATA
print('\nFormating data...', end='\n')

start = time.time()
time_bin_duration = 10 #TODO : automatiser cette variable

#Time vector
wav_datetimes = [extract_datetime(x) for x in wav_list] #datetime of wav files

first_date = from_str2dt(dfpamguard['start_datetime'][0]) #1st detection
last_date = from_str2dt(dfpamguard['start_datetime'].iloc[-1]) #last detection

#selection of waf files according to first and last dates
idx_wav_beg = 0 if all(wav_datetimes[i] >= first_date for i in range(len(wav_datetimes))) else [i for i, x in enumerate(wav_datetimes) if x < first_date][-1]
idx_wav_end = len(wav_list) if all(wav_datetimes[i] <= last_date for i in range(len(wav_datetimes))) else [i for i, x in enumerate(wav_datetimes) if x > last_date][0]
wav_datetimes, wav_list, wav_folder, wav_files, durations = wav_datetimes[idx_wav_beg:idx_wav_end], wav_list[idx_wav_beg:idx_wav_end], wav_folder[idx_wav_beg:idx_wav_end], wav_files[idx_wav_beg:idx_wav_end], durations[idx_wav_beg:idx_wav_end]
print('\n1st wav : ' + wav_list[0])
print('last wav : ' + wav_list[-1], end='\n\n')

time_vector = [elem for i in range(len(wav_list)) for elem in extract_datetime(wav_list[i]).timestamp() + np.arange(0, durations[i], time_bin_duration).astype(int)]
time_vector_str = [str(wav_list[i]).split('.wav')[0]+ '_+'  + str(elem) for i in range(len(wav_list)) for elem in np.arange(0, durations[i], time_bin_duration).astype(int)]


#Aplose

if Ap == 1:
    tuple_aplose = sorting_annot_boxes(aplose_path, tz_data, first_date, last_date)
    time_bin = tuple_aplose[0]
    fmax = tuple_aplose[1]
    annotators = tuple_aplose[2]
    labels = tuple_aplose[3]
    df_aplose = tuple_aplose[-1]
    
    # label_ref = ''.join(easygui.buttonbox('Select a label', 'Single plot', labels) if len(labels)>1 else labels)
    selected_annotations = df_aplose.loc[(df_aplose['annotation'] == labels[1])]
        
    times_Ap_beg = sorted(list(set(x.timestamp() for x in selected_annotations['start_datetime'])) )
    times_Ap_end = sorted(list(set(y.timestamp() for y in selected_annotations['end_datetime']))) #set -> Remove recurrent elements ie annotator common annotations in the list, returns a set with random order -> list + sorting
    
    Aplose_vec, ranks, k = [], [], 0
    for i in range(len(times_Ap_beg)):
        for j in range(k, len(time_vector)-1):
            if int(times_Ap_beg[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)) or int(times_Ap_end[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)):
                    ranks.append(j)
                    k=j
                    break
            else: 
                continue 
    ranks = sorted(list(set(ranks)))
    for i in range(len(time_vector)): Aplose_vec.append(1) if i in ranks else Aplose_vec.append(0)
    

#Pamguard
times_PG_beg = [from_str2ts(x) for x in dfpamguard['start_datetime']]
times_PG_end = [from_str2ts(x) for x in dfpamguard['end_datetime']]

PG_vec, ranks, k = [], [], 0
for i in tqdm(range(len(times_PG_beg)), 'Importing PAMGuard detections...'):
    for j in range(k, len(time_vector)-1):
        if int(times_PG_beg[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)) or int(times_PG_end[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)):
                ranks.append(j)
                k=j
                break
        else: 
            continue 
test4_1 = [time_vector_str[ranks[i]].split('_+',1)[0] for i in range(len(ranks))] #Used later in Raven section
ranks = sorted(list(set(ranks)))
for i in tqdm(range(len(time_vector))): PG_vec.append(1) if i in ranks else PG_vec.append(0)
print('\tDone!', end='\n')

##  DETECTION PERFORMANCES
if Ap ==1:
    true_pos, false_pos, true_neg, false_neg, error = 0,0,0,0,0
    for i in range(len(time_vector)):
        if Aplose_vec[i] == 0 and PG_vec[i] == 0:
            true_neg+=1
        elif Aplose_vec[i] == 1 and PG_vec[i] == 1:
            true_pos+=1
        elif Aplose_vec[i] == 0 and PG_vec[i] == 1:
            false_pos+=1   
        elif Aplose_vec[i] == 1 and PG_vec[i] == 0:
            false_neg+=1
        else:error+=1
            
    if error == 0:
        print('\nTrue positive : ', true_pos)
        print('True negative : ', true_neg)
        print('False positive : ', false_pos)
        print('False negative : ', false_neg)   
        
        print('\nPRECISION : ', round(true_pos/(true_pos+false_pos),3))
        print('RECALL : ', round(true_pos/(false_neg+true_pos) ,3    ) )
    else: print('Error : ', error)
    
end = time.time()
print('\nElapsed time : ', round(end-start,2), 's')  
#%% TEST - Other detector

#Import detections
# root = tk.Tk()
# root.withdraw()
# detector_path = filedialog.askopenfilename(title="Select detector detection file", filetypes=[("CSV files", "*.csv")])
# dfdetector = pd.read_csv(detector_path, delimiter=';')

# Dolph_Vec = list(dfdetector['dolphinfree'])
# Dolph_Vec2 = Dolph_Vec[:8640]
# Dolph_Vec3 = [int(Dolph_Vec2[i]) for i in range(len(Dolph_Vec2))]


# ##  DETECTION PERFORMANCES
# true_pos2, false_pos2, true_neg2, false_neg2, error2 = 0,0,0,0,0
# for i in range(len(Dolph_Vec3)):
#     if Aplose_vec[i] == 0 and Dolph_Vec3[i] == 0:
#         true_neg2+=1
#     elif Aplose_vec[i] == 1 and Dolph_Vec3[i] == 1:
#         true_pos2+=1
#     elif Aplose_vec[i] == 0 and Dolph_Vec3[i] == 1:
#         false_pos2+=1   
#     elif Aplose_vec[i] == 1 and Dolph_Vec3[i] == 0:
#         false_neg2+=1
#     else:error2+=1
        
# if error2 == 0:
#     print('\n\nTrue positive : ', true_pos2)
#     print('True negative : ', true_neg2)
#     print('False positive : ', false_pos2)
#     print('False negative : ', false_neg2)   
    
#     print('\nPRECISION : ', round(true_pos2/(true_pos2+false_pos2),3))
#     print('RECALL : ', round(true_pos2/(false_neg2+true_pos2) ,3    ) )
# else: print('Error : ', error2)
#%% EXPORT TO APLOSE FORMAT : TODO reshape df comme on veut + fonction + écrire ?

df_PG2Aplose = pd.DataFrame()
dataset_str = list(set(dfpamguard['dataset']))

start_datetime_str, end_datetime_str, filename = [],[],[]
for i in range(len(time_vector)):
    if PG_vec[i] == 1:
        start_datetime = pytz.timezone('UTC').localize(dt.datetime.utcfromtimestamp(time_vector[i])) #
        start_datetime = start_datetime.astimezone(pytz.timezone(tz_data))
        start_datetime_str.append(start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[:-8]+ start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-5:-2] +':' + start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-2:])
        end_datetime = pytz.timezone('UTC').localize(dt.datetime.utcfromtimestamp(time_vector[i]+10))
        end_datetime = end_datetime.astimezone(pytz.timezone(tz_data))
        end_datetime_str.append(end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[:-8]+ end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-5:-2] +':' + end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-2:])
        filename.append(time_vector_str[i])

df_PG2Aplose['dataset'] = dataset_str*len(start_datetime_str)
df_PG2Aplose['filename'] = filename
df_PG2Aplose['start_time'] = [0]*len(start_datetime_str)
df_PG2Aplose['end_time'] = [time_bin]*len(start_datetime_str)
df_PG2Aplose['start_frequency'] = [0]*len(start_datetime_str)
df_PG2Aplose['end_frequency'] = [fmax]*len(start_datetime_str)

df_PG2Aplose['annotation'] = list(set(dfpamguard['annotation']))*len(start_datetime_str)
df_PG2Aplose['annotator'] = list(set(dfpamguard['annotator']))*len(start_datetime_str)
  
df_PG2Aplose['start_datetime'], df_PG2Aplose['end_datetime'] = start_datetime_str, end_datetime_str

PG2Ap_str = "/PG_formatteddata_" + t_rounder(wav_datetimes[0]).strftime('%y%m%d') + '_' + t_rounder(wav_datetimes[-1]).strftime('%y%m%d') +'_'+ str(time_bin_duration) + 's'+ '.csv'

df_PG2Aplose.to_csv(os.path.dirname(pamguard_path) + PG2Ap_str, index=False)  
print('\n\nAplose formatted data file exported to '+ os.path.dirname(pamguard_path))

#%% EXPORT TO RAVEN FORMAT
##Formatted data
df_PG2Raven = pd.DataFrame()

df_PG2Raven['Selection'] = np.arange(1,len(start_datetime_str)+1)
df_PG2Raven['View'], df_PG2Raven['Channel'] = [1]*len(start_datetime_str), [1]*len(start_datetime_str)

datetime_endfiles, datetime_begfiles=[],[]
for i in range(len(wav_list)):
    datetime_endfiles.append((wav_datetimes[i]+dt.timedelta(seconds=durations[i])).strftime('%Y-%m-%d %H:%M:%S.%f'))
    datetime_begfiles.append((wav_datetimes[i]).strftime('%Y-%m-%d %H:%M:%S.%f'))
    # datetime_begfiles.append(dt.datetime.utcfromtimestamp(time_vector[idx_beg[i]]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]) #datetime beginning of next file according to title

# df_Raven = pd.DataFrame({'datetime begin': datetime_begfiles, 'datetime begin + duration': datetime_endfiles})

offsets =[]
for i in range(len(datetime_endfiles)-1):
    offsets.append(((wav_datetimes[i]+dt.timedelta(seconds=durations[i])).timestamp() - (wav_datetimes[i+1]).timestamp()))
offsets_cumsum=(list(np.cumsum([offsets[i] for i in range(len(offsets))])))
offsets_cumsum.insert(0, 0)

test3 = [wav_list[i].split('.wav')[0] for i in range(len(wav_list))] #names of the waves without extension
start_datetime, end_datetime = [],[] 
for i in range(len(time_vector)):
    if PG_vec[i] == 1:
        test4 = [time_vector_str[i].split('_+')[0] == test3[j] for j in range(len(wav_list))] #finding which wav the detection is belonging to
        idx_wav_Raven = [i for i, x in enumerate(test4) if x][0] #index of the wav the detection i is belonging to
        start_datetime.append(int(time_vector[i] - wav_datetimes[0].timestamp())      + offsets_cumsum[idx_wav_Raven] )
        end_datetime.append(int(time_vector[i] - wav_datetimes[0].timestamp())+10     + offsets_cumsum[idx_wav_Raven] )

df_PG2Raven['Begin Time (s)'] = start_datetime     
df_PG2Raven['End Time (s)'] = end_datetime     

df_PG2Raven['Low Freq (Hz)'] = [0]*len(start_datetime_str)
df_PG2Raven['High Freq (Hz)'] = [fmax//2.5]*len(start_datetime_str)

PG2Raven_str = "/PG2Raven_Formatteddata_" + t_rounder(wav_datetimes[0]).strftime('%y%m%d') + '_' + t_rounder(wav_datetimes[-1]).strftime('%y%m%d') + '_'+ str(time_bin_duration) + 's' + '.txt'

df_PG2Raven.to_csv(os.path.dirname(pamguard_path) + PG2Raven_str, index=False, sep='\t')  
print('\n\nRaven formatted data file exported to '+ os.path.dirname(pamguard_path))




