import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import pytz
import os
import easygui
from post_processing_detections.utilities.def_func import extract_datetime, sorting_detections, t_rounder, get_timestamps

from tkinter import filedialog
from tkinter import Tk
import glob

#%% User input
root = Tk()
root.withdraw()
detections_file1 = filedialog.askopenfilename(title="Select APLOSE formatted detection file", filetypes=[("CSV files", "*.csv")])
timestamps_file1 = get_timestamps()

root = Tk()
root.withdraw()
detections_file2 = filedialog.askopenfilename(title="Select APLOSE formatted detection file", filetypes=[("CSV files", "*.csv")])
timestamps_file2 = get_timestamps()


t1_detections, t2_detections = sorting_detections(detections_file1), sorting_detections(detections_file2)

time_bin = sorted(list(set([t1_detections[0], t2_detections[0]])))
fmax = sorted(list(set([t1_detections[1], t2_detections[1]])))
annotators = sorted(list(set(t1_detections[2] + t2_detections[2])))
labels = sorted(list(set(t1_detections[3] + t2_detections[3])))
df1_detections, df2_detections = t1_detections[-1], t2_detections[-1]

if len(time_bin) != 1 : print('time bins are different', time_bin)
else: time_bin = time_bin[0]

# for i in range(len(df2_detections)):
#     df2_detections['annotation'][i] = 'Odontocete whistle'
#%% Plot
label_legend = ['1', '2']
tz_data = list(set([i.tz for i in pd.concat([df1_detections['start_datetime'], df2_detections['start_datetime']])]))[0]

label_ref = ''.join(easygui.buttonbox('Select a label', 'Single plot', labels) if len(labels)>1 else labels[0])
annot_ref = ''.join(easygui.buttonbox('Select an annotator', 'Single plot', annotators) if len(annotators)>1 else annotators[0])
res_min = int(easygui.enterbox("résolution temporelle bin ? (min) :"))
# res_min = 10

wav_names1 = timestamps_file1['filename']
wav_names2 = timestamps_file2['filename']

delta, start_vec, end_vec = dt.timedelta(seconds=60*res_min), t_rounder(extract_datetime(min(wav_names1[0], wav_names2[0]), tz=tz_data),res = 600), t_rounder(max(extract_datetime(wav_names1.iloc[-1], tz=tz_data) + dt.timedelta(seconds=time_bin), extract_datetime(wav_names2.iloc[-1], tz=tz_data) + dt.timedelta(seconds=time_bin)),res = 600)
          
time_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]
duration_h = int((time_vector[-1] - time_vector[0]).total_seconds()//3600)

n_annot_max = (res_min*60)/time_bin #max nb of annoted time_bin max per res_min slice

df1_1annot_1label, df2_1annot_1label = df1_detections.loc[(df1_detections['annotator'] == annot_ref) & (df1_detections['annotation'] == label_ref)], df2_detections.loc[(df2_detections['annotator'] == 'PAMGuard') & (df2_detections['annotation'] == label_ref)]
df1_1annot_1label, df2_1annot_1label = df1_1annot_1label.sort_values('start_datetime'), df2_1annot_1label.sort_values('start_datetime')


fig,ax = plt.subplots(nrows = 1, figsize=(20,9))
bar_plot = ax.hist([df1_1annot_1label['start_datetime'], df2_1annot_1label['start_datetime']], bins=time_vector, label=[label_legend[0], label_legend[1]]); #histo

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))
ax.set_yticks(y_pos, bars);
ax.tick_params(axis='x', rotation= 60);
ax.tick_params(labelsize=20)
ax.set_ylabel("% de détections positives ("+str(res_min)+"min)", fontsize = 20)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M', tz=tz_data))
ax.tick_params(axis='x', rotation= 80)

if time_vector[0].strftime('%d/%m/%Y') != time_vector[-1].strftime('%d/%m/%Y'):
    fig.suptitle('annotator : '+annot_ref +'\n'+ 'label : ' + label_ref+'\n'+time_vector[0].strftime('%d/%m/%Y') + time_vector[-1].strftime(' - %d/%m/%Y UTC%z')+'\nduration : '+str(duration_h)+'h', fontsize = 24, y=1.06);
else:
    fig.suptitle('annotator : '+annot_ref +'\n'+ 'label : ' + label_ref+'\n'+time_vector[-1].strftime('%d/%m/%Y UTC%z')+'\nduration : '+str(duration_h)+'h', fontsize = 24, y=1.06);

    
# grey background on odd/even days
date_odd_even = [j for i,j in enumerate(time_vector) if j.day%2==0] #select odd or even days
for i,j in enumerate(list(set([j.day for i,j in enumerate(date_odd_even)]))):
    vec = [l for k,l in enumerate(time_vector) if l.day==j]
    if (vec[-1].hour == 23) : vec[-1] += dt.timedelta(hours=1)
    elif vec[-1].day == time_vector[-1].day: vec[-1] = time_vector[-1]
    ax.fill_between([vec[0], vec[-1]],n_annot_max, color='grey', alpha=0.075) 
    
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
# ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.xlim(time_vector[0], time_vector[-1])
plt.ylim(0, n_annot_max)
ax.grid(color='k', linestyle='-', linewidth=0.2, axis='both')
ax.legend(fontsize = 30)

#%% scatter
import seaborn as sns
from scipy import stats

df_corr = pd.DataFrame(bar_plot[0]/n_annot_max, index=['col1', 'col2']).transpose()
# sns.scatterplot(x="col1", y="col2", data=df_corr)
sns.set_style('darkgrid')
plot = sns.lmplot(x="col1", y="col2", data=df_corr, scatter_kws={'s': 10, 'color': 'teal'}, fit_reg=True, markers='.', line_kws={'lw': 1,'color': 'teal'})
plt.xlim(0, 1)
plt.ylim(0, 1)

def annotate(data, **kws):
    r, p = stats.pearsonr(data['col1'], data['col2'])
    ax = plt.gca()
    ax.text(.05, .8, 'R²={:.2f}'.format(r*r),
            transform=ax.transAxes)
    
plot.map_dataframe(annotate)
plt.show()
#%% plot bis

if len(labels)>1:
    label_ref = easygui.buttonbox('Select a label', 'Plot label', labels)
elif len(labels)==1:
    label_ref = labels[0]

if len(annotators)>1:
    annot_ref = easygui.buttonbox('Select a label', 'Plot label', annotators)
elif len(annotators)==1:
    annot_ref = annotators[0]

res_min = int(easygui.enterbox("résolution temporelle bin ? (min) :"))
delta, start_vec, end_vec = dt.timedelta(seconds=60*res_min), t_rounder(extract_datetime(min(wav_names1.iloc[0], wav_names2.iloc[0]), tz=tz_data)), t_rounder(max(extract_datetime(wav_names1.iloc[-1], tz=tz_data) + dt.timedelta(seconds=time_bin), extract_datetime(wav_names2.iloc[-1], tz=tz_data) + dt.timedelta(seconds=time_bin)))
          
time_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]

time_slice = 60*res_min #10 min
n_annot_max = time_slice/time_bin #nb of annoted time_bin max per time_slice

df1_1annot_1label, df2_1annot_1label = df1_detections.loc[(df1_detections['annotator'] == annot_ref) & (df1_detections['annotation'] == label_ref)], df2_detections.loc[(df2_detections['annotator'] == annot_ref) & (df2_detections['annotation'] == label_ref)]
df1_1annot_1label, df2_1annot_1label = df1_1annot_1label.sort_values('start_datetime'), df2_1annot_1label.sort_values('start_datetime')

fig, (ax1, ax2) = plt.subplots(nrows = 2, figsize=(30,20))

ax1.hist(df1_1annot_1label['start_datetime'], bins=time_vector); #histo annotation
ax2.hist(df2_1annot_1label['start_datetime'], bins=time_vector); #histo annotation

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
# fig.suptitle(annot_ref +' '+ label_ref + date_begin.strftime(' - %d/%m/%y UTC%z'), fontsize = 24, y=0.95);

ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz_data))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz_data))

ax1.grid(color='k', linestyle='-', linewidth=0.2, axis='both')
ax2.grid(color='k', linestyle='-', linewidth=0.2, axis='both')











