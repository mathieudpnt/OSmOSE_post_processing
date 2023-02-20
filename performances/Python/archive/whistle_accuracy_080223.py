from tkinter import filedialog
import tkinter as tk
from tkinter import Tk
import time

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select wav file", filetypes=[("wav files", "*.wav")])


#%%
start = time.time()
with open(labels_path, "rb") as wav_file:
    # Read the first 44 bytes, which is the size of the WAV header
    header = wav_file.read(44)

    # Extract the audio format (1 for PCM)
    audio_format = int.from_bytes(header[20:22], byteorder='little')

    # Extract the number of channels (1 for mono, 2 for stereo)
    num_channels = int.from_bytes(header[22:24], byteorder='little')

    # Extract the sample rate
    sample_rate = int.from_bytes(header[24:28], byteorder='little')

    # Extract the bit depth
    bit_depth = int.from_bytes(header[34:36], byteorder='little')
    
    # Calculate the number of samples
    num_samples = int.from_bytes(header[40:44], byteorder='little')

    # Calculate the duration
    duration = num_samples / sample_rate

print("Duration:", duration, "seconds")
print("Audio format:", audio_format)
print("Number of channels:", num_channels)
print("Sample rate:", sample_rate)
print("Bit depth:", bit_depth)
stop = time.time()
t0 = stop-start
print(t0)
#%%
import struct
from typing import Tuple

def read_header(file:str) -> Tuple[int, int, int, int, int]:
    with open(file, "rb") as fh:
       _, size, _ = struct.unpack('<4sI4s', fh.read(12))

       chunk_header = fh.read(8)
       subchunkid, _ = struct.unpack('<4sI', chunk_header)

       if (subchunkid == b'fmt '):
           _, channels, samplerate, _, _, sampwidth = struct.unpack('HHIIHH', fh.read(16))

       sampwidth = (sampwidth + 7) // 8
       framesize = channels * sampwidth
       frames = size // framesize 
       return sampwidth, frames, samplerate, channels, frames/samplerate

test = read_header(file_path)
   

#%%
def read_header(file:str) -> Tuple[int, int, int, int]:
    with open(file, "rb") as fh:
        _, size, _ = struct.unpack('<4sI4s', fh.read(12))

        chunk_header = fh.read(8)
        subchunkid, _ = struct.unpack('<4sI', chunk_header)

        if (subchunkid == b'fmt '):
            _, channels, samplerate, _, _, sampwidth = struct.unpack('HHIIHH', fh.read(16))

        sampwidth = (sampwidth + 7) // 8
        framesize = channels * sampwidth
        frames = size // framesize

        return sampwidth, frames, samplerate, channels



#%%

import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import calendar, time
import os
import datetime as dt
import pytz
import glob
import re
from tkinter import filedialog
import wave
import tkinter as tk
from tkinter import Tk

def extract_datetime_gm(var, formats=None, tz=None):
    if tz is None : tz = 'Europe/Paris'
    if formats is None:
        formats = [r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', r'\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}'] #add more format if necessary
    match = None
    for f in formats:
        match = re.search(f, var)
        if match:
            break
    if match:
        dt_string = match.group()
        if f == r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}':
            dt_format = '%Y-%m-%d_%H-%M-%S'
        elif f == r'\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}':
            dt_format = '%y%m%d%H%M%S'
        elif f == r'\d{2}\d{2}\d{2}_\d{2}\d{2}\d{2}':
            dt_format = '%y%m%d_%H%M%S'
        date_obj = dt.datetime.strptime(dt_string, dt_format)
        date_obj = pytz.timezone(tz).localize(date_obj)
        return date_obj.timestamp()
    else:
        return print("No datetime found")
 
def format_time_PG(date):
    return dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()

def format_time_Ap(date):
    return dt.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()
    
def get_wav_info(folder):
    names, durations=[],[]
    wav_files = glob.glob(os.path.join(folder, "**/*.wav"), recursive=True)
    for file in tqdm(wav_files, 'Getting wav infos...'):
        try:
            with wave.open(file, 'r') as wav_files:
                frames = wav_files.getnframes()
                rate = wav_files.getframerate()
                durations.append(frames / float(rate))
                names.append(file)
        except Exception as e:
            print(f'An error occured while reading the file {file} : {e}') 
    print('\tDone !')
    df = pd.DataFrame({'name': names, 'duration': durations})    
    return df

def format_wavname(date):
    elem = str(date).split('.wav')[0]
    return calendar.timegm(time.strptime(elem, 'channelA_%Y-%m-%d_%H-%M-%S'))+int(str(date).split('.wav_')[1])

def round_nearest_hour(var):
    if var.minute >= 30:
        var += dt.timedelta(hours=1)
        var = var.replace(minute=0, second=0, microsecond=0)
    return var

tz_data='Europe/Paris'
Ap=1

#%% LOAD DATA

#PAMGuard detections
root = tk.Tk()
Tk().withdraw()
pamguard_path = filedialog.askopenfilename(title="Select PAMGuard detection file", filetypes=[("CSV files", "*.csv")])
dfpamguard = pd.read_csv(pamguard_path)
dfpamguard = dfpamguard.sort_values('start_datetime')

#APLOSE annotations
if Ap == 1:
    root = tk.Tk()
    root.withdraw()
    labels_path = filedialog.askopenfilename(title="Select APLOSE annotation file", filetypes=[("CSV files", "*.csv")])
    dflabels = pd.read_csv(labels_path)


#WAV files 
root = Tk()
root.withdraw()
response = tk.messagebox.askyesno("Confirm", "Do you have a wav info csv file ?", default="no")
if response == True:
    wavinfo_path = filedialog.askopenfilename(title="Select wav info file", filetypes=[("CSV files", "*.csv")])
    df_wavinfo = pd.read_csv(wavinfo_path)
else:
    root = tk.Tk()
    root.withdraw()
    wavpath = filedialog.askdirectory(title = 'Select wav folder')
    df_wavinfo = get_wav_info(wavpath)
    df_wavinfo.to_csv(wavpath + '/wavinfo.csv', index=False)  
print('\tDone!')

#%% FORMAT DATA
start = time.time()
time_bin_duration = 10

#Time vector: list of all possible boxes
print('\nTime vector creation...', end='')
wav_files = list(df_wavinfo['name'])
durations = list(df_wavinfo['duration'])
wav_list, wav_folder = [], []
for file in wav_files:
    wav_list.append(os.path.basename(file))
    wav_folder.append(os.path.dirname(file))

#tri dates
wav_datetimes = [dt.datetime.fromtimestamp(extract_datetime_gm(x)) for x in wav_list] #datetime des wav
wav_datetimes = [pytz.timezone(tz_data).localize(wav_datetimes[i]) for i in range(len(wav_datetimes))]

first_date = dt.datetime.fromtimestamp(format_time_PG(dfpamguard['start_datetime'][0])) #jour 1ere détection
first_date = pytz.timezone(tz_data).localize(first_date)
last_date = dt.datetime.fromtimestamp(format_time_PG(dfpamguard['start_datetime'].iloc[-1])) # dernière détection
last_date = pytz.timezone(tz_data).localize(last_date)

test1 = [wav_datetimes[i]<first_date for i in range(len(wav_datetimes))]
idx_wav_beg = 0 if not any(test1) else [i for i, x in enumerate(test1) if x][-1]

test2= [wav_datetimes[i]>last_date for i in range(len(wav_datetimes))]
idx_wav_end = len(wav_list) if not any(test2) else [i for i, x in enumerate(test2) if x][0]

wav_datetimes = wav_datetimes[idx_wav_beg:idx_wav_end]
wav_list = wav_list[idx_wav_beg:idx_wav_end]
wav_folder = wav_folder[idx_wav_beg:idx_wav_end]
wav_files = wav_files[idx_wav_beg:idx_wav_end]
durations = durations[idx_wav_beg:idx_wav_end]

print('\n1st wav : ' + wav_list[0])
print('last wav : ' + wav_list[-1])

time_vector = [elem for i in range(len(wav_list)) for elem in extract_datetime_gm(wav_list[i]) + np.arange(0, durations[i], time_bin_duration).astype(int)]
time_vector_str = [str(wav_list[i]).split('.wav')[0]+ '_+'  + str(elem) for i in range(len(wav_list)) for elem in np.arange(0, durations[i], time_bin_duration).astype(int)]
print('\tDone!')


#Aplose
if Ap == 1:
    print('\nImporting Aplose annotations...', end='')
    len_label = max(dflabels['end_time'])
    selected_annotations = dflabels.loc[(dflabels['annotation'] == 'Odontocete whistles') & (dflabels['start_time']==0) & (dflabels['end_time']==len_label) & (dflabels['start_frequency']==0) & (dflabels['end_frequency']==max(dflabels['end_frequency']))]
    # selected_annotations = dflabels.loc[(dflabels['annotation'] == 'Odontocete clics') & (dflabels['start_time']==0) & (dflabels['end_time']==max(dflabels['end_time'])) & (dflabels['start_frequency']==0) & (dflabels['end_frequency']==max(dflabels['end_frequency']))]
    selected_annotations = selected_annotations.sort_values(by='start_datetime')
    
    times_Ap_beg = selected_annotations['start_datetime'].apply(lambda x : format_time_Ap(x)).to_numpy() 
    times_Ap_beg = np.unique(times_Ap_beg) #Remove recurrent elements  ie annotator common annotations
    times_Ap_end = selected_annotations['end_datetime'].apply(lambda x : format_time_Ap(x)).to_numpy() 
    times_Ap_end = np.unique(times_Ap_end) #Remove recurrent elements  ie annotator common annotations
    
    Aplose_vec =[]
    ranks = []
    k=0
    for i in range(len(times_Ap_beg)):
        for j in range(k, len(time_vector)-1):
            if int(times_Ap_beg[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)) or int(times_Ap_end[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)):
                    ranks.append(j)
                    k=j
                    break
            else: 
                continue 
    ranks = np.unique(ranks)
    for i in range(len(time_vector)): Aplose_vec.append(1) if i in ranks else Aplose_vec.append(0)
    print('\tDone!')
    

#Pamguard
print('\nImporting PAMGuard detections...', end='')
times_PG_beg = dfpamguard['start_datetime'].apply(lambda x : format_time_PG(x)).to_numpy()          #format time
times_PG_end = dfpamguard['end_datetime'].apply(lambda x : format_time_PG(x)).to_numpy()          #format time

ranks, PG_vec = [],[]
k=0
for i in tqdm(range(len(times_PG_beg)), 'Importing PAMGuard detections...'):
    for j in range(k, len(time_vector)-1):
        if int(times_PG_beg[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)) or int(times_PG_end[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)):
                ranks.append(j)
                k=j
                break
        else: 
            continue 
test4_1 = [time_vector_str[ranks[i]].split('_+',1)[0] for i in range(len(ranks))] #Used later in Raven section
ranks = np.unique(ranks)
for i in range(len(time_vector)): PG_vec.append(1) if i in ranks else PG_vec.append(0)
print('\tDone!')

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
        print('\n\nTrue positive : ', true_pos)
        print('True negative : ', true_neg)
        print('False positive : ', false_pos)
        print('False negative : ', false_neg)   
        
        print('\nPRECISION : ', round(true_pos/(true_pos+false_pos),3))
        print('RECALL : ', round(true_pos/(false_neg+true_pos) ,3    ) )
    else: print('Error : ', error)
    
end = time.time()
print('Elapsed time : ', round(end-start), 's')  

#%% EXPORT TO APLOSE FORMAT

df_PG2Aplose = pd.DataFrame()
dataset_str = list(pd.unique(dfpamguard['dataset']))

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
df_PG2Aplose['end_time'] = [10]*len(start_datetime_str)
df_PG2Aplose['start_frequency'] = [0]*len(start_datetime_str)
df_PG2Aplose['end_frequency'] = [sf.info(os.path.join(wav_folder[-1], wav_list[-1])).samplerate]*len(start_datetime_str)
df_PG2Aplose['annotation'] = ['Whistle and moan detector']*len(start_datetime_str)
df_PG2Aplose['annotator'] = ['PAMGuard']*len(start_datetime_str)
  
df_PG2Aplose['start_datetime'], df_PG2Aplose['end_datetime'] = start_datetime_str, end_datetime_str

PG2Ap_str = "/PG_formatteddata_" + round_nearest_hour(wav_datetimes[0]).strftime('%y%m%d') + '_' + round_nearest_hour(wav_datetimes[-1]).strftime('%y%m%d') +'_'+ str(time_bin_duration) + 's'+ '.csv'

df_PG2Aplose.to_csv(os.path.dirname(pamguard_path) + PG2Ap_str, index=False)  
print('\n\nAplose formatted data file exported to '+ os.path.dirname(pamguard_path))

#%% Other detector
#Import detections
root = tk.Tk()
root.withdraw()
detector_path = filedialog.askopenfilename(title="Select detector detection file", filetypes=[("CSV files", "*.csv")])
dfdetector = pd.read_csv(detector_path, delimiter=';')

Dolph_Vec = list(dfdetector['dolphinfree'])
Dolph_Vec2 = Dolph_Vec[:8640]
Dolph_Vec3 = [int(Dolph_Vec2[i]) for i in range(len(Dolph_Vec2))]


##  DETECTION PERFORMANCES
true_pos2, false_pos2, true_neg2, false_neg2, error2 = 0,0,0,0,0
for i in range(len(Dolph_Vec3)):
    if Aplose_vec[i] == 0 and Dolph_Vec3[i] == 0:
        true_neg2+=1
    elif Aplose_vec[i] == 1 and Dolph_Vec3[i] == 1:
        true_pos2+=1
    elif Aplose_vec[i] == 0 and Dolph_Vec3[i] == 1:
        false_pos2+=1   
    elif Aplose_vec[i] == 1 and Dolph_Vec3[i] == 0:
        false_neg2+=1
    else:error2+=1
        
if error2 == 0:
    print('\n\nTrue positive : ', true_pos2)
    print('True negative : ', true_neg2)
    print('False positive : ', false_pos2)
    print('False negative : ', false_neg2)   
    
    print('\nPRECISION : ', round(true_pos2/(true_pos2+false_pos2),3))
    print('RECALL : ', round(true_pos2/(false_neg2+true_pos2) ,3    ) )
else: print('Error : ', error2)


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
df_PG2Raven['High Freq (Hz)'] = [sf.info(os.path.join(wav_folder[-1], wav_list[-1])).samplerate//2.5]*len(start_datetime_str)

PG2Raven_str = "/PG2Raven_Formatteddata" + round_nearest_hour(wav_datetimes[0]).strftime('%y%m%d') + '_' + round_nearest_hour(wav_datetimes[-1]).strftime('%y%m%d') + '_'+ str(time_bin_duration) + 's' + '.txt'

df_PG2Raven.to_csv(os.path.dirname(pamguard_path) + PG2Raven_str, index=False, sep='\t')  
print('\n\nRaven formatted data file exported to '+ os.path.dirname(pamguard_path))

# ##Raw data
# df2_PG2Raven = pd.DataFrame()

# df2_PG2Raven['Selection'] = np.arange(1,len(dfpamguard)+1)
# df2_PG2Raven['View'], df2_PG2Raven['Channel'] = [1]*len(dfpamguard), [1]*len(dfpamguard)

# start_datetime2, end_datetime2 = [],[] 
# for i in tqdm(range(len(dfpamguard))):
#     test4_2 = [test4_1[i] == wav_list[j].split('.wav')[0] for j in range(len(wav_list))] #finding which wav the detection is belonging to
#     idx_wav_Raven = [i for i, x in enumerate(test4_2) if x][0] #index of the wav the detection i is belonging to
#     start_datetime2.append(dfpamguard['Begin_time'][i] + offsets_cumsum[idx_wav_Raven])
#     end_datetime2.append(dfpamguard['End_time'][i] + offsets_cumsum[idx_wav_Raven])

# df2_PG2Raven['Begin Time (s)'] = start_datetime2    
# df2_PG2Raven['End Time (s)'] = end_datetime2        

# df2_PG2Raven['Low Freq (Hz)'] = dfpamguard['Low_Freq']
# df2_PG2Raven['High Freq (Hz)'] = dfpamguard['High_Freq']

# PG2Raven_str2 = "/PG2Raven_Rawdata" + round_nearest_hour(wav_datetimes[0]).strftime('%y%m%d') + '_' + round_nearest_hour(wav_datetimes[-1]).strftime('%y%m%d') + '.txt'

# df2_PG2Raven.to_csv(os.path.dirname(pamguard_path) + PG2Raven_str2, index=False, sep='\t')  
# print('\n\nRaven raw data file exported to '+ os.path.dirname(pamguard_path))




