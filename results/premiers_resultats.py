import os
from pathlib import Path
import datetime as dt
import glob
from tkinter import filedialog
from tkinter import Tk
import pandas as pd
import numpy as np
import easygui
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from post_processing_detections.utilities.def_func import read_header, extract_datetime, sorting_annot_boxes, t_rounder

#%% User inputs

root = Tk()
root.withdraw()
folder = filedialog.askdirectory(title="Select wav folder")
root = Tk()
root.withdraw()
detections_file = filedialog.askopenfilename(initialdir = Path(folder).parents[0], title="Select APLOSE format detection file", filetypes=[("CSV files", "*.csv")])


t_detections = sorting_annot_boxes(detections_file)

time_bin = t_detections[0]
fmax = t_detections[1]
annotators = t_detections[2]
labels = t_detections[3]
df_detections = t_detections[-1]


tz_data = df_detections['start_datetime'][0].tz
wav_files = glob.glob(os.path.join(folder, "**/*.wav"), recursive=True)
wav_names = [os.path.basename(file) for file in wav_files]
test_wav = [j in sorted(list(set([i.split('_')[0] for i in df_detections['filename']]))) for j in [i.split('.wav')[0] for i in wav_names]]
wav_names, wav_files = zip(*[(wav_names[i], wav_files[i]) for i in range(len(wav_names)) if test_wav[i]]) #only the wav files corresponding to the detections are kept


durations = [read_header(file)[-1] for file in wav_files]
total_duration = sum(durations)

# test_datetimes = [(extract_datetime(y) >= date_begin) & (extract_datetime(y) <= date_end) for y in wav_names]
# print('\n'+ str(sum(test_datetimes)), 'fichiers compris entre', str(date_begin), 'et', str(date_end), end='')


# wav_files = [wav_files[i] for i in range(len(wav_files)) if test_datetimes[i]==True ]
# wav_names = [wav_names[i] for i in range(len(wav_names)) if test_datetimes[i]==True ]
# durations = [durations[i] for i in range(len(durations)) if test_datetimes[i]==True ]
# print('\n1st wav : ' + wav_names[0], end='\n')
# print('last wav : ' + wav_names[-1], end='\n')

print("\ntime_bin : ", str(time_bin), "s", end='')
print("\nfmax : ", str(fmax), "Hz", end='')
print('\nannotators :',str(annotators), end='')
print('\nlabels :', str(labels), end='\n')



#%% Single plot 
label_ref = ''.join(easygui.buttonbox('Select a label', 'Single plot', t_detections[3]) if len(t_detections[3])>1 else t_detections[3])
annot_ref = ''.join(easygui.buttonbox('Select an annotator', 'Single plot', t_detections[2]) if len(t_detections[2])>1 else t_detections[2])
res_min = int(easygui.enterbox("résolution temporelle bin ? (min) :"))
    
delta, start_vec, end_vec = dt.timedelta(seconds=60*res_min), t_rounder(extract_datetime(wav_names[0], tz_data)), t_rounder(extract_datetime(wav_names[-1], tz_data) + dt.timedelta(seconds=durations[-1]))

time_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]

n_annot_max = (res_min*60)/time_bin #max nb of annoted time_bin max per res_min slice

df_1annot_1label = df_detections.loc[(df_detections['annotator'] == annot_ref) & (df_detections['annotation'] == label_ref)]
df_1annot_1label=df_1annot_1label.sort_values('start_datetime')

fig,ax = plt.subplots(figsize=(20,9))
ax.hist(df_1annot_1label['start_datetime'], bins=time_vector); #histo

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))
ax.set_yticks(y_pos, bars);
ax.tick_params(axis='x', rotation= 60);
ax.tick_params(labelsize=20)
ax.set_ylabel("% de détections positives ("+str(res_min)+"min)", fontsize = 20)
ax.tick_params(axis='y')
fig.suptitle('annotateur : '+annot_ref +'\n'+ 'label : ' + label_ref, fontsize = 24, y=0.98);
 
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz_data))
plt.xlim(time_vector[0], time_vector[-1])
# plt.xlim(t_rounder(df_detections['start_datetime'][0]), t_rounder(df_detections['end_datetime'].iloc[-1]))
ax.grid(color='k', linestyle='-', linewidth=0.2, axis='both')


#%% Multilabel plot

if len(annotators)>1:
    annot_ref = easygui.buttonbox('Select an annotator', 'Plot label', annotators)
elif len(annotators==1):
    annot_ref = annotators[0]

selected_labels = labels[0:3] #TODO : checkbox to select desired labels to plot ?

res_min = int(easygui.enterbox("résolution temporelle bin ? (min) :"))
# delta, start_vec, end_vec = dt.timedelta(seconds=60*res_min), t_rounder(extract_datetime(wav_names[0])), t_rounder(extract_datetime(wav_names[-1]) + dt.timedelta(seconds=durations[-1]))
delta, start_vec, end_vec = dt.timedelta(seconds=60*res_min), t_rounder(extract_datetime(wav_names[0], tz_data)), t_rounder(extract_datetime(wav_names[-1], tz_data))
time_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]


time_slice = 60*res_min #10 min
n_annot_max = time_slice/time_bin #nb of annoted time_bin max per time_slice

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))

fig, ax = plt.subplots(nrows = len(selected_labels), figsize=(40,20))
fig.tight_layout(pad=10)

for i, label in enumerate(selected_labels):
    
    df_1annot_1label = df_detections.loc[(df_detections['annotator'] == annot_ref) & (df_detections['annotation'] == label)]
    df_1annot_1label = df_1annot_1label.sort_values('start_datetime')

    ax[i].hist(df_1annot_1label['start_datetime'], bins=time_vector, color='teal'); #histo

    bars = range(0,110,10) #from 0 to 100 step 10
    y_pos = np.linspace(0,n_annot_max, num=len(bars))
    ax[i].set_yticks(y_pos, bars);
    ax[i].tick_params(axis='x', rotation= 60);
    ax[i].tick_params(labelsize=20)
    ax[i].set_title(label, fontsize = 20)
    ax[i].set_ylabel("positive detection rate ("+str(res_min)+"min)", fontsize = 20)
    ax[i].tick_params(axis='y')
    # fig.suptitle('annotateur : '+annot_ref +'\n'+ 'label : ' + label_ref, fontsize = 24, y=0.98);
     
    ax[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz_data))
    ax[i].set_xlim(time_vector[0], time_vector[-1])
    ax[i].grid(color='k', linestyle='-', linewidth=0.2, axis='both')








