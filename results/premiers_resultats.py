import datetime as dt
from tkinter import filedialog
from tkinter import Tk
import pandas as pd
import numpy as np
import easygui
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from post_processing_detections.utilities.def_func import extract_datetime, sorting_annot_boxes, t_rounder

#%% User inputs

root = Tk()
root.withdraw()
detections_file = filedialog.askopenfilename(title="Select APLOSE formatted detection file", filetypes=[("CSV files", "*.csv")])
taskstatus_file = filedialog.askopenfilename(initialdir = detections_file, title="Select APLOSE formatted task status file", filetypes=[("CSV files", "*.csv")])

t_detections = sorting_annot_boxes(detections_file)
df_taskstatus = pd.read_csv(taskstatus_file)

time_bin = t_detections[0]
fmax = t_detections[1]
annotators = t_detections[2]
labels = t_detections[3]
df_detections = t_detections[-1]

tz_data = df_detections['start_datetime'][0].tz

wav_names = df_taskstatus['filename']

print("\ntime_bin : ", str(time_bin), "s", end='')
print("\nfmax : ", str(fmax), "Hz", end='')
print('\nannotators :',str(annotators), end='')
print('\nlabels :', str(labels), end='\n')


#%% Single plot 
#%TODO date spécifié par utilisateur, de miniut à minuit
label_ref = ''.join(easygui.buttonbox('Select a label', 'Single plot', t_detections[3]) if len(t_detections[3])>1 else t_detections[3])
annot_ref = ''.join(easygui.buttonbox('Select an annotator', 'Single plot', t_detections[2]) if len(t_detections[2])>1 else t_detections[2])
res_min = int(easygui.enterbox("résolution temporelle bin ? (min) :"))
    
delta, start_vec, end_vec = dt.timedelta(seconds=60*res_min), t_rounder(extract_datetime(wav_names.iloc[0], tz_data)), t_rounder(extract_datetime(wav_names.iloc[-1], tz_data) + dt.timedelta(seconds=time_bin))

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
# plt.xlim(time_vector[0], dt.datetime.strptime('2022-07-07T22-00-00', '%Y-%m-%dT%H-%M-%S'))
ax.grid(color='k', linestyle='-', linewidth=0.2, axis='both')


#%% Multilabel plot

if len(annotators)>1:
    annot_ref = easygui.buttonbox('Select an annotator', 'Plot label', annotators)
elif len(annotators==1):
    annot_ref = annotators[0]

selected_labels = labels[0:3] #TODO : checkbox to select desired labels to plot ?

res_min = int(easygui.enterbox("résolution temporelle bin ? (min) :"))
delta, start_vec, end_vec = dt.timedelta(seconds=60*res_min), t_rounder(extract_datetime(wav_names[0], tz_data)), t_rounder(extract_datetime(wav_names.iloc[-1], tz_data) + dt.timedelta(seconds=time_bin))
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


#%% Multiuser plot

if len(annotators)>2:
    annot_ref1 = easygui.buttonbox('Select annotator 1', 'Plot label', annotators)
    annot_ref2 = easygui.buttonbox('Select an annotator', 'Plot label', [elem for elem in annotators if elem != annot_ref1])
elif len(annotators)<2:
    print('Not enough annotators to make a comparison')
else:
    annot_ref1 = annotators[0]
    annot_ref2 = annotators[1]

label_ref = ''.join(easygui.buttonbox('Select a label', 'Single plot', t_detections[3]) if len(t_detections[3])>1 else t_detections[3])

res_min = int(easygui.enterbox("résolution temporelle bin ? (min) :"))

delta, start_vec, end_vec = dt.timedelta(seconds=60*res_min), t_rounder(extract_datetime(wav_names.iloc[0], tz_data)), t_rounder(extract_datetime(wav_names.iloc[-1], tz_data) + dt.timedelta(seconds=time_bin))

time_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]

n_annot_max = (res_min*60)/time_bin #max nb of annoted time_bin max per res_min slice

df1_1annot_1label = df_detections.loc[(df_detections['annotator'] == annot_ref1) & (df_detections['annotation'] == label_ref)].sort_values('start_datetime')
df2_1annot_1label = df_detections.loc[(df_detections['annotator'] == annot_ref2) & (df_detections['annotation'] == label_ref)].sort_values('start_datetime')

fig,ax = plt.subplots(figsize=(16,6), facecolor='#36454F')
ax.set_facecolor('#36454F')
hist_plot = ax.hist([df1_1annot_1label['start_datetime'], df2_1annot_1label['start_datetime']], bins=time_vector, label=[annot_ref1, annot_ref2], color=['coral','limegreen'], lw=10);
plt.legend(loc='upper right', fontsize = 14)

bars = range(0,110,10) #from 0 to 100 step 10
y_pos = np.linspace(0,n_annot_max, num=len(bars))
ax.set_yticks(y_pos, bars);
ax.tick_params(axis='x', rotation= 60);
ax.tick_params(labelsize=20)
ax.set_ylabel("% de détections positives ("+str(res_min)+"min)", fontsize = 20, c='w')
ax.tick_params(axis='y')
fig.suptitle('annotateurs : '+annot_ref1 + ' & ' + annot_ref2 +'\n'+ 'label : ' + label_ref,color='w', fontsize = 24, y=1.02);
 
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz_data))
plt.xlim(time_vector[0], time_vector[-1])
plt.xlim(time_vector[0], dt.datetime.strptime('2022-07-07T22-00-00', '%Y-%m-%dT%H-%M-%S'))
ax.grid(color='w', linestyle='-', linewidth=0.2, axis='both')
ax.tick_params(axis='both', colors='w')

# spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')

# % accord inter-annot
unique_list = df_detections[(df_detections['annotation'] == label_ref) & (df_detections['annotator'] == annot_ref1)]['filename']
unique_list.append(df_detections[(df_detections['annotation'] == label_ref) & (df_detections['annotator'] == annot_ref2)]['filename'])
unique_annotations = len(list(set(unique_list)))

df_common = df_detections[df_detections['annotation'] == label_ref]
common_annotations_test = df_common[df_common['annotator'] == annot_ref1]['filename'].isin(df_common[df_common['annotator'] == annot_ref2]['filename'])
common_annotations = len(df_common[df_common['annotator'] == annot_ref1][common_annotations_test])

print('Pourcentage d\'accord entre les annotateurs {0} et {1} sur le label {2}: {3:.0f}%'.format(annot_ref1, annot_ref2, label_ref,100*(common_annotations/unique_annotations)))

#%% scatter
import seaborn as sns
from scipy import stats

df_corr = pd.DataFrame(hist_plot[0]/n_annot_max, index=['col1', 'col2']).transpose()
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
