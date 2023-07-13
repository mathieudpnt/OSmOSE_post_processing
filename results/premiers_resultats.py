import datetime as dt
from tkinter import filedialog
from tkinter import Tk
import pandas as pd
import numpy as np
import easygui
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
import seaborn as sns
from scipy import stats
from post_processing_detections.utilities.def_func import extract_datetime, sorting_annot_boxes, t_rounder, get_timestamps

#%% User inputs

root = Tk()
root.withdraw()
detections_file = filedialog.askopenfilename(title="Select APLOSE formatted detection file", filetypes=[("CSV files", "*.csv")])
timestamps_file = get_timestamps()

t_detections = sorting_annot_boxes(detections_file)
time_bin = t_detections[0]
fmax = t_detections[1]
annotators = t_detections[2]
labels = t_detections[3]
df_detections = t_detections[-1]

tz_data = df_detections['start_datetime'][0].tz

wav_names = timestamps_file['filename']

print("\ntime_bin : ", str(time_bin), "s", end='')
print("\nfmax : ", str(fmax), "Hz", end='')
print('\nannotators :',str(annotators), end='')
print('\nlabels :', str(labels), end='\n')

#%% Overview plots

summary_label = df_detections.groupby('annotation')['annotator'].apply(Counter).unstack(fill_value=0)
summary_annotator = df_detections.groupby('annotator')['annotation'].apply(Counter).unstack(fill_value=0)

print('\n\t%%% Overview of the detections : %%%\n\n {0}'.format(summary_label))
print('\n\t-----------------------------------\n\n {0}'.format(summary_annotator))

fig, (ax1, ax2) = plt.subplots(2, figsize=(10,10), gridspec_kw={'height_ratios':[1, 1]}, facecolor='#36454F')
ax1 = summary_label.plot(kind='bar', ax=ax1, color=['tab:blue', 'tab:orange'], edgecolor='black', linewidth=1)
ax2 = summary_annotator.plot(kind='bar', ax=ax2, edgecolor='black', linewidth=1)

#facecolor
ax1.set_facecolor('#36454F')
ax2.set_facecolor('#36454F')

#spacing between plots
plt.subplots_adjust(hspace=0.4)

#legend
ax1.legend(loc='best', fontsize=10, frameon=1, framealpha=0.6)
ax2.legend(loc='best', fontsize=10, frameon=1, framealpha=0.6)

#ticks
ax1.tick_params(axis='both', colors='w',rotation=0, labelsize=12)
ax2.tick_params(axis='both', colors='w',rotation=0, labelsize=12)

#labels
ax1.set_ylabel('Number of annotated calls', fontsize=15, color='w')
ax1.set_xlabel('Labels', fontsize=15, rotation=0, color='w')
ax2.set_ylabel('Number of annotated calls', fontsize=15, color='w')
ax2.set_xlabel('Annotator', fontsize=15, rotation=0, color='w')

#spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_color('w')
ax1.spines['left'].set_color('w')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_color('w')
ax2.spines['left'].set_color('w')

#y-grids
ax1.yaxis.grid(color='gray', linestyle='--')
ax2.yaxis.grid(color='gray', linestyle='--')
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)

#titles
title_font = {'fontsize': 15, 'color': 'w', 'fontweight': 'bold'}
ax1.set_title('Number of annotations per label', color='w', fontdict=title_font, pad=5)
ax2.set_title('Number of annotations per annotator', color='w', fontdict=title_font, pad=5)


#%% Single plot 
#%TODO date spécifié par utilisateur, de miniut à minuit

label_ref = ''.join(easygui.buttonbox('Select a label', 'Single plot', t_detections[3]) if len(t_detections[3])>1 else t_detections[3])
annot_ref = ''.join(easygui.buttonbox('Select an annotator', 'Single plot', t_detections[2]) if len(t_detections[2])>1 else t_detections[2])
res_min = easygui.integerbox('Enter the bin size (min) ', 'Time resolution', default=10, lowerbound=1, upperbound=86400)
    
delta, start_vec, end_vec = dt.timedelta(seconds=60*res_min), t_rounder(extract_datetime(wav_names.iloc[0], tz_data),res = 600), t_rounder(extract_datetime(wav_names.iloc[-1], tz_data) + dt.timedelta(seconds=time_bin),res = 600)

time_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]

n_annot_max = (res_min*60)/time_bin #max nb of annoted time_bin max per res_min slice

df_1annot_1label = sorting_annot_boxes(detections_file, annotator = annot_ref, label = label_ref)[-1]

fig,ax = plt.subplots(figsize=(20,9), facecolor='#36454F')
ax.hist(df_1annot_1label['start_datetime'], bins=time_vector, color='crimson', edgecolor='black', linewidth=1)

#facecolor
ax.set_facecolor('#36454F')

ax.tick_params(axis='y', colors='w', rotation=0,  labelsize=20)
ax.tick_params(axis='x', colors='w', rotation=60, labelsize=15)

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))
ax.set_yticks(y_pos, bars)

ax.set_ylabel('positive detection rate\n({0} min)'.format(res_min), fontsize = 20, color='w')

#spines
ax.spines['right'].set_color('w')
ax.spines['top'].set_color('w')
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')

#titles
fig.suptitle('annotateur : '+annot_ref +'\n'+ 'label : ' + label_ref, fontsize = 24, y=0.98, color='w')
 
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz_data))
plt.xlim(time_vector[0], time_vector[-1])
# plt.xlim(time_vector[0], dt.datetime.strptime('2022-07-07T22-00-00', '%Y-%m-%dT%H-%M-%S'))
ax.grid(color='w', linestyle='--', linewidth=0.2, axis='both')


#%% Multilabel plot

if len(annotators)>1:
    annot_ref = easygui.buttonbox('Select an annotator', 'Plot label', annotators)
elif len(annotators==1):
    annot_ref = annotators[0]

selected_labels = labels[0:3] #TODO : checkbox to select desired labels to plot ?

res_min = easygui.integerbox('Enter the bin size (min) ', 'Time resolution', default=10, lowerbound=1, upperbound=86400)
delta, start_vec, end_vec = dt.timedelta(seconds=60*res_min), t_rounder(extract_datetime(wav_names[0], tz_data),res = 600), t_rounder(extract_datetime(wav_names.iloc[-1], tz_data) + dt.timedelta(seconds=time_bin),res = 600)
time_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]


time_slice = 60*res_min #10 min
n_annot_max = time_slice/time_bin #nb of annoted time_bin max per time_slice

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))

fig, ax = plt.subplots(nrows = len(selected_labels), figsize=(25,15), facecolor='#36454F')
fig.tight_layout(pad=10)

for i, label in enumerate(selected_labels):
    
    df_1annot_1label = sorting_annot_boxes(detections_file, annotator = annot_ref, label = label)[-1]

    ax[i].hist(df_1annot_1label['start_datetime'], bins=time_vector, color='crimson', edgecolor='black', linewidth=1)

    bars = range(0,110,10) #from 0 to 100 step 10
    y_pos = np.linspace(0,n_annot_max, num=len(bars))
    ax[i].set_facecolor('#36454F')
    ax[i].set_yticks(y_pos, bars)
    ax[i].tick_params(axis='both', colors='w',rotation=0, labelsize=15)
    ax[i].tick_params(axis='x', rotation= 60)
    ax[i].set_title(label, fontsize = 15, color='w')
    ax[i].set_ylabel('positive detection rate\n({0} min)'.format(res_min), fontsize = 15, color='w')

     
    ax[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz_data))
    ax[i].set_xlim(time_vector[0], time_vector[-1])
    ax[i].grid(color='w', linestyle='--', linewidth=0.2, axis='both')

fig.suptitle('Annotator : {0}'.format(annot_ref), fontsize = 25, y=0.98, color='w', weight= 'bold')

#%% Multi-user plot

if len(annotators)>2:
    annot_ref1 = easygui.buttonbox('Select annotator 1', 'Plot label', annotators)
    annot_ref2 = easygui.buttonbox('Select an annotator', 'Plot label', [elem for elem in annotators if elem != annot_ref1])
elif len(annotators)<2:
    print('Not enough annotators to make a comparison')
else:
    annot_ref1 = annotators[0]
    annot_ref2 = annotators[1]

label_ref = ''.join(easygui.buttonbox('Select a label', 'Single plot', t_detections[3]) if len(t_detections[3])>1 else t_detections[3])

res_min = easygui.integerbox('Enter the bin size (min) ', 'Time resolution', default=10, lowerbound=1, upperbound=86400)

delta, start_vec, end_vec = dt.timedelta(seconds=60*res_min), t_rounder(extract_datetime(wav_names.iloc[0], tz_data), res=600), t_rounder(extract_datetime(wav_names.iloc[-1], tz_data) + dt.timedelta(seconds=time_bin), res=600)


time_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]

n_annot_max = (res_min*60)/time_bin #max nb of annoted time_bin max per res_min slice

df1_1annot_1label = sorting_annot_boxes(detections_file, annotator = annot_ref1, label = label_ref)[-1]
df2_1annot_1label = sorting_annot_boxes(detections_file, annotator = annot_ref2, label = label_ref)[-1]

fig,ax = plt.subplots(figsize=(16,6), facecolor='#36454F')
ax.set_facecolor('#36454F')
hist_plot = ax.hist([df1_1annot_1label['start_datetime'], df2_1annot_1label['start_datetime']], bins=time_vector, label=[annot_ref1, annot_ref2], color=['coral','limegreen'], lw=10);
plt.legend(loc='upper right', fontsize = 14)

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))
ax.set_yticks(y_pos, bars);
ax.tick_params(axis='x', rotation= 60);
ax.tick_params(labelsize=20)
ax.set_ylabel('positive detection rate\n({0} min)'.format(res_min), fontsize = 20, c='w')
ax.tick_params(axis='y')
fig.suptitle('annotateurs : '+annot_ref1 + ' & ' + annot_ref2 +'\n'+ 'label : ' + label_ref,color='w', fontsize = 24, y=1.02);
 
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz_data))
plt.xlim(time_vector[0], time_vector[-1])
# plt.xlim(time_vector[0], dt.datetime.strptime('2022-07-07T22-00-00', '%Y-%m-%dT%H-%M-%S'))
ax.grid(color='w', linestyle='-', linewidth=0.2, axis='both')
ax.tick_params(axis='both', colors='w')

# spines
ax.spines['right'].set_color('w')
ax.spines['top'].set_color('w')
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')

# accord inter-annot
list1 = list(sorting_annot_boxes(detections_file, annotator = annot_ref1, label = label_ref)[-1]['filename'])
list2 = list(sorting_annot_boxes(detections_file, annotator = annot_ref2, label = label_ref)[-1]['filename'])
unique_annotations = len([elem for elem in list1 if elem not in list2 ]) + len([elem for elem in list2 if elem not in list1 ])
common_annotations = len([elem for elem in list1 if elem in list2 ])
print('Pourcentage d\'accord entre les annotateurs {0} et {1} sur le label {2}: {3:.0f}%'.format(annot_ref1, annot_ref2, label_ref,100*((common_annotations)/(unique_annotations + common_annotations))))

# scatter
df_corr = pd.DataFrame(hist_plot[0]/n_annot_max, index=[annot_ref1, annot_ref2]).transpose()
plot = sns.lmplot(x=annot_ref1, y=annot_ref2, data=df_corr, scatter_kws={'s': 10, 'color': 'teal'}, fit_reg=True, markers='.', line_kws={'lw': 1,'color': 'teal'})
plt.xlim(0, 1)
plt.ylim(0, 1)

def annotate(data, **kws):
    r, p = stats.pearsonr(data[annot_ref1], data[annot_ref2])
    ax = plt.gca()
    ax.text(.05, .8, '{1}\nR²={0:.2f}'.format(r*r, label_ref),
            transform=ax.transAxes)
    
plot.map_dataframe(annotate)
plt.show()