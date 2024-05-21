import datetime as dt
import pandas as pd
import numpy as np
import easygui
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from collections import Counter
import seaborn as sns
from scipy import stats
import sys
import os
from cycler import cycler

# os.chdir(r'U:/Documents_U/Git/post_processing_detections')
os.chdir(r'C:\Users\dupontma2\Desktop\data_local\post_processing_detections-main_17052024')
from utilities.def_func import sorting_detections, t_rounder, get_timestamps, input_date, suntime_hour, read_param

mpl.style.use('seaborn-v0_8-paper')
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams["axes.prop_cycle"] = cycler('color', ['#4590d3', 'darkorange', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

# %% User inputs

# Load parameters from the YAML file
yaml_file_path = os.path.join(os.getcwd(), 'results', 'premiers_resultats_parameters.yaml')
parameters = read_param(file=yaml_file_path)

df_detections, info = pd.DataFrame(), pd.DataFrame()
for args in parameters:
    df_detections_file, info_file = sorting_detections(**args)
    df_detections = pd.concat([df_detections, df_detections_file], ignore_index=True)
    info = pd.concat([info, info_file], ignore_index=True)

'''
time_bin = list(set(info['max_time'].explode()))
fmax = list(set(info['max_freq'].explode()))
annotators = list(set(info['annotators'].explode()))
labels = list(set(info['labels'].explode()))
tz_data = list(set(info['tz_data'].explode()))
if len(tz_data) == 1:
    [tz_data] = tz_data
else:
    raise Exception('More than one timezone in the detections')
'''
time_bin = list(info['max_time'].explode())
fmax = list(info['max_freq'].explode())
annotators = list(info['annotators'].explode())
labels = list(info['labels'].explode())
tz_data = list(info['tz_data'].explode())


'''
Chose your mode :
    -fixed: hard coded date interval
    -auto: the script automatically extract the timestamp from the timestamp file
    -input: you will fill a dialog box with the start and end date
'''
dt_mode = 'fixed'

if dt_mode == 'fixed':
    begin_date = pd.Timestamp('2022-07-06 23:59:47 +0200')
    end_date = pd.Timestamp('2022-07-08 01:59:28 +0200')    
    # begin_date = pd.Timestamp('2022-07-07 09:00:00 +0200')
    # end_date = pd.Timestamp('2022-07-08 00:00:00 +0200')

    # begin_date = pd.Timestamp('2023-02-05 11:39:00 +0100')
    # end_date = pd.Timestamp('2023-02-06 08:51:00 +0100')

    # begin_date = pd.Timestamp('2023-02-11 12:10:47 +0100')
    # end_date = pd.Timestamp('2023-02-12 08:50:00 +0100')

    # begin_date = pd.Timestamp('2023-02-11 12:15:47 +0100')
    # end_date = pd.Timestamp('2023-02-12 09:00:00 +0100')

    # begin_date = pd.Timestamp('2023-04-10 01:50:00 +0200')
    # end_date = pd.Timestamp('2023-04-11 01:40:00 +0200')
    
elif dt_mode == 'auto':
    timestamps_file = get_timestamps()
    begin_date = pd.to_datetime(timestamps_file['timestamp'].iloc[0], format='%Y-%m-%dT%H:%M:%S.%f%z')
    end_date = pd.to_datetime(timestamps_file['timestamp'].iloc[-1], format='%Y-%m-%dT%H:%M:%S.%f%z') + dt.timedelta(seconds=time_bin[0])

print(f"\ntime_bin: {time_bin}", end='')
print(f"\nfmax: {fmax}", end='')
print(f"\nannotators: {annotators}", end='')
print(f"\nlabels: {labels}", end='')
print(f'\nBegin date: {begin_date}', end='')
print(f'\nEnd date: {end_date}', end='')

# %% Overview plots

summary_label = df_detections.groupby('annotation')['annotator'].apply(Counter).unstack(fill_value=0)
summary_annotator = df_detections.groupby('annotator')['annotation'].apply(Counter).unstack(fill_value=0)

print(f'\n- Overview of the detections -\n\n {summary_label}')

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]})
axs[0] = summary_label.plot(kind='bar', ax=axs[0], edgecolor='black', linewidth=1)
axs[1] = summary_annotator.plot(kind='bar', ax=axs[1], edgecolor='black', linewidth=1)

for ax in axs:
    # legend
    ax.legend(loc='best', frameon=1, framealpha=0.6)
    # ticks
    ax.tick_params(axis='both', rotation=0)
    ax.set_ylabel('Number of annotated calls')
    # y-grids
    ax.yaxis.grid(color='gray', linestyle='--')
    ax.set_axisbelow(True)

# labels
axs[0].set_xlabel('Labels')
axs[1].set_xlabel('Annotator')

# titles
axs[0].set_title('Number of annotations per label')
axs[1].set_title('Number of annotations per annotator')

plt.tight_layout()

# %% % labels / species

# Créer une nouvelle colonne 'species' en regroupant les labels par espèce
df_detections['species'] = np.where(df_detections['annotation'].str.startswith('Tt'), 'Tt',
                                   np.where(df_detections['annotation'].str.startswith('Sc'), 'Sc',
                                            np.where(df_detections['annotation'].str.startswith('Pm'), 'Pm',
                                                     np.where(df_detections['annotation'].str.startswith('Gm'), 'Gm', 'Other'))))

# Mapping des nouvelles valeurs pour les espèces
species_mapping = {'Tt': 'Tursiops truncatus', 'Sc': 'Stenella coeruleoalba', 'Pm': 'Physeter macrocephalus', 'Gm': 'Globicephala melas'}

# Remplacer les valeurs de la colonne 'species' par les nouvelles valeurs
df_detections['species'] = df_detections['species'].map(species_mapping)

# Résumé des annotations par espèce
summary_species_percentage = df_detections.groupby('species')['annotator'].count() / len(df_detections) * 100

# Utiliser la palette de couleurs 'coolwarm'
colors = sns.color_palette('husl')

# Création du camembert avec la palette 'coolwarm'
plt.figure(figsize=(8, 8), facecolor='#36454F')
plt.pie(summary_species_percentage, labels=summary_species_percentage.index, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'color': 'w'})

# Titre du camembert
plt.title('Pourcentage d\'annotations par espèce', color='w', fontsize=15, fontweight='bold', pad=20)

# Affichage du camembert
plt.show()


####
# Résumé des annotations par type
summary_label_percentage = df_detections.groupby('annotation')['annotator'].count() / len(df_detections) * 100

# Utiliser la palette de couleurs 'coolwarm'
colors = sns.color_palette('husl', 16)

# Création du camembert avec la palette 'coolwarm'
plt.figure(figsize=(8, 8), facecolor='#36454F')
patches, texts, autotexts = plt.pie(summary_label_percentage, labels=summary_label_percentage.index, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'color': 'w'})

# Ajuster la couleur du texte à l'intérieur des tranches
for autotext in autotexts:
    autotext.set_color('w')

# Titre du camembert
plt.title('Pourcentage d\'annotations par type de vocalisation', color='w', fontsize=15, fontweight='bold', pad=20)

# Affichage du camembert
plt.show()


# %% Single seasonality plot

# ----------- User set mdate time xticks-----------------------------
# One tick per month
# mdate1 = mdates.MonthLocator(interval=1)
# mdate2 = mdates.DateFormatter('%B', tz=tz_data)
# One tick every 2 weeks
# mdate1 = mdates.DayLocator(interval=15, tz=tz_data)
# mdate2 = mdates.DateFormatter('%d-%B', tz=tz_data)
# One tick every day
# mdate1 = mdates.DayLocator(interval=1, tz=tz_data)
# mdate2 = mdates.DateFormatter('%d-%m', tz=tz_data)
# One tick every hour
mdate1 = mdates.HourLocator(interval=1, tz=tz_data[0])
mdate2 = mdates.DateFormatter('%H:%M', tz=tz_data[0])
# -------------------------------------------------------------------

# selection of the user
annot_ref = easygui.buttonbox('Select an annotator', 'Single plot', annotators) if len(annotators) > 1 else annotators[0]
# list of the labels corresponding to the sleected user
list_labels = info[info['annotators'].apply(lambda x: annot_ref in x)]['labels'].reset_index(drop=True)[0]
# selection of the label
label_ref = easygui.buttonbox('Select a label', 'Single plot', list_labels) if len(list_labels) > 1 else list_labels[0]

time_bin_ref = int(info[info['annotators'].apply(lambda x: annot_ref in x)]['max_time'].reset_index(drop=True).iloc[0])
file_ref = info[info['annotators'].apply(lambda x: annot_ref in x)]['file'].reset_index(drop=True).iloc[0]

# Ask user if their resolution_bin is in minutes or in months or in seasons
resolution_bin = easygui.buttonbox(msg='Do you want to chose your resolution bin in minutes or in months', choices=('Minutes', 'Days', 'Weeks', 'Months'))
if resolution_bin == 'Minutes':
    res_min = easygui.integerbox('Enter the bin size (min)', 'Time resolution', default=10, lowerbound=1, upperbound=86400)
    n_annot_max = (res_min * 60) / time_bin_ref  # max nb of annoted time_bin max per res_min slice
    # Est-ce que c'est utile de garder start_vec et end_vec sachant qu'ils sont égaux à begin_date et end_date non ?
    delta, start_vec, end_vec = dt.timedelta(seconds=60 * res_min), t_rounder(begin_date, res=600), t_rounder(end_date + dt.timedelta(seconds=time_bin_ref), res=600)
    time_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]
    y_label_txt = 'Number of detections\n({0} min)'.format(res_min)

elif resolution_bin == 'Days':
    time_vector_ts = pd.date_range(begin_date, end_date, freq='D', tz=tz_data)
    time_vector = [timestamp.date() for timestamp in time_vector_ts]
    n_annot_max = (24 * 60 * 60) / time_bin_ref
    y_label_txt = 'Number of detections per day'

elif resolution_bin == 'Weeks':
    time_vector_ts = pd.date_range(begin_date, end_date, freq='W-MON', tz=tz_data)
    time_vector = [timestamp.date() for timestamp in time_vector_ts]
    n_annot_max = (24 * 60 * 60 * 7) / time_bin_ref
    y_label_txt = 'Number of detections per week (starting every Monday)'

else:
    # Compute the time_vector for a monthly resolution
    time_vector_ts = pd.date_range(begin_date, end_date, freq='MS', tz=tz_data)
    time_vector = [timestamp.date() for timestamp in time_vector_ts]
    n_annot_max = (31 * 24 * 60 * 60) / time_bin_ref
    y_label_txt = 'Number of detections per month'


# df_1annot_1label = sorting_detections(file=file_ref, annotator=annot_ref, label=label_ref, timebin_new=time_bin_ref, fmin_filter=10000)
df_1annot_1label = df_detections[(df_detections['annotator'] == annot_ref) & (df_detections['annotation'] == label_ref)]

fig, ax = plt.subplots(figsize=(20, 9), facecolor='#36454F')
[hist_y, hist_x, _] = ax.hist(df_1annot_1label['start_datetime'], bins=time_vector, color='crimson', edgecolor='black', linewidth=1)

# Compute df_hist for user to check the values contained in the histogram
hist_xt = [pd.to_datetime(x * 24 * 60 * 60, unit='s') for x in hist_x[:-1]]
df_hist = pd.DataFrame({'Date': hist_xt, 'Number of detection': hist_y.tolist()})

# facecolor
ax.set_facecolor('#36454F')

# ticks
ax.tick_params(axis='y', colors='w', rotation=0, labelsize=20)
ax.tick_params(axis='x', colors='w', rotation=60, labelsize=15)

# Du coup c'est pas totalement exact par ce que j'ai calculé qu'un seul n_annot_max alors qu'en vrai il est différent chaque mois vu que tous les mois n'ont pas la même durée...

ax.set_ylabel(y_label_txt, fontsize=20, color='w')

# spines
ax.spines['right'].set_color('w')
ax.spines['top'].set_color('w')
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')

# titles
fig.suptitle('annotateur : ' + annot_ref + '\n' + 'label : ' + label_ref, fontsize=24, y=0.98, color='w')

ax.xaxis.set_major_locator(mdate1)
ax.xaxis.set_major_formatter(mdate2)
plt.xlim(time_vector[0], time_vector[-1])
ax.grid(color='w', linestyle='--', linewidth=0.2, axis='both')

# Ask the user if they want to visualize the Figure in % or in raw values
choice_percentage = easygui.buttonbox(msg='Do you want your results plot in % or in raw values ?', choices=('Percentage', 'Raw values'))
# To change the y scale
# #change value 2 in bars = range(0, 110, 2) to change the space between two ticks
# #change value 0.08 in ax.set_ylim([0,n_annot_max * 0.08]) to change y max
if choice_percentage == 'Percentage':
    bars = np.arange(0, 110, 10)  # from 0 to 100 step 10
    y_pos = [n_annot_max * p / 100 for p in bars]
    ax.set_yticks(y_pos, bars)
    ax.set_ylim([0, n_annot_max])
    if resolution_bin == 'Minutes':
        ax.set_ylabel('Detection rate % \n({0} min)'.format(res_min), fontsize=20, color='w')
    else:
        ax.set_ylabel('Detection rate % per month', fontsize=20, color='w')

# %% Single diel pattern plot (scatter raw detections)

# ----------- User set mdate time xticks-----------------------------
# One tick per month
mdate1 = mdates.MonthLocator(interval=1)
mdate2 = mdates.DateFormatter('%B', tz=tz_data)
# One tick every 2 weeks
# mdate1 = mdates.DayLocator(interval=15, tz=tz_data)
# mdate2 = mdates.DateFormatter('%d-%B', tz=tz_data)
# One tick every day
# mdate1 = mdates.DayLocator(interval=1, tz=tz_data)
# mdate2 = mdates.DateFormatter('%d-%m', tz=tz_data)
# One tick every hour
# mdate1 = mdates.HourLocator(interval=1,tz=tz_data)
# mdate2 = mdates.DateFormatter('%H:%M', tz=tz_data)
# ----------------------------------------------------------------------------

# User input : gps coordinates in Decimal Degrees
title = "Coordinates en degree° minute' "
msg = "Latitudes (N/S) and longitudes (E/W)"
fieldNames = ["Lat Decimal Degree", "Lon Decimal Degree "]
fieldValues = []  # we start with blanks for the values
fieldValues = easygui.multenterbox(msg, title, fieldNames)

# make sure that none of the fields was left blank
while 1:
    if fieldValues is None: break
    errmsg = ""
    for i in range(len(fieldNames)):
        if fieldValues[i].strip() == "":
            errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
    if errmsg == "": break  # no problems found
    fieldValues = easygui.multpasswordbox(errmsg, title, fieldNames, fieldValues)
print("Reply was:", fieldValues)

lat = fieldValues[0]
lon = fieldValues[1]
# Compute sunrise and sunet decimal hour at the dataset location
[hour_sunrise, hour_sunset, _, _, _, _] = suntime_hour(begin_date, end_date, tz_data, lat, lon)

date_beg = begin_date.strftime('%Y-%m-%d')
date_end = end_date.strftime('%Y-%m-%d')

x_data = np.arange(date_beg, date_end, dtype="M8[D]")

dt_detections = [x.to_pydatetime() for x in df_detections['start_datetime']]

Day_det = dt_detections
Hour_det = [x.hour + x.minute / 60 for x in dt_detections]

# Plot figure
fig, ax = plt.subplots(figsize=(20, 10))
plt.plot(x_data, hour_sunrise, color='k')
plt.plot(x_data, hour_sunset, color='k')
plt.scatter(Day_det, Hour_det)

plt.xlim(begin_date, end_date)

ax.xaxis.set_major_locator(mdate1)
ax.xaxis.set_major_formatter(mdate2)
# plt.xlim(time_vector[0], time_vector[-1])
ax.grid(color='k', linestyle='-', linewidth=0.2)

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
ax.tick_params(axis='y', rotation=0, labelsize=20)
ax.tick_params(axis='x', rotation=60, labelsize=15)

ax.set_ylabel('Hour', fontsize=30)
ax.set_xlabel('Date', fontsize=30)

ax.set_title('Time of detections within each day for dataset {}'.format(df_detections['dataset'][0]), fontsize=40)


# %% Single diel pattern plot (Hourly detection rate)

# ----------- User set mdate time xticks-----------------------------
# One tick per month
# mdate1 = mdates.MonthLocator(interval=1)
# mdate2 = mdates.DateFormatter('%B', tz=tz_data)
# One tick every 2 weeks
# mdate1 = mdates.DayLocator(interval=15,tz=tz_data)
# mdate2 = mdates.DateFormatter('%d-%B', tz=tz_data)
# One tick every day
# mdate1 = mdates.DayLocator(interval=1, tz=tz_data)
# mdate2 = mdates.DateFormatter('%d-%m', tz=tz_data)
# One tick every hour
mdate1 = mdates.HourLocator(interval=1,tz=tz_data)
mdate2 = mdates.DateFormatter('%H:%M', tz=tz_data)
# ----------------------------------------------------------------------------

# User input : gps coordinates in Decimal Degrees
title = "Coordinates en degree° minute' "
msg = "Latitudes (N/S) and longitudes (E/W)"
fieldNames = ["Lat Decimal Degree", "Lon Decimal Degree "]
fieldValues = []  # we start with blanks for the values
fieldValues = easygui.multenterbox(msg, title, fieldNames)

# make sure that none of the fields was left blank
while 1:
    if fieldValues is None: break
    errmsg = ""
    for i in range(len(fieldNames)):
        if fieldValues[i].strip() == "":
            errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
    if errmsg == "": break  # no problems found
    fieldValues = easygui.multpasswordbox(errmsg, title, fieldNames, fieldValues)
print("Reply was:", fieldValues)

lat = fieldValues[0]
lon = fieldValues[1]
# Compute sunrise and sunet decimal hour at the dataset location
[hour_sunrise, hour_sunset, _, _, _, _] = suntime_hour(begin_date, end_date, tz_data, lat, lon)

date_beg = begin_date.strftime('%Y-%m-%d')
date_end = end_date.strftime('%Y-%m-%d')

x_data = np.arange(date_beg, date_end, dtype="M8[D]")

a = [dt.datetime.strftime(x, '%y-%m-%d') for x in df_detections['start_datetime']]
b = [dt.datetime.strftime(x, '%H') for x in df_detections['start_datetime']]
df_detections['date'] = a
df_detections['hour'] = b

det_groupby = df_detections.groupby(['date', 'hour']).size()
idx_day_groupby = det_groupby.index.get_level_values(0)
idx_hour_groupby = det_groupby.index.get_level_values(1)

time_vector_ts = pd.date_range(begin_date, end_date, freq='D', tz=tz_data)
time_vector_str = [dt.datetime.strftime(x, '%y-%m-%d') for x in time_vector_ts]


# arr = [[0]*cols]*rows
M = np.zeros((24, len(time_vector_str)))

for idx_j, j in enumerate(time_vector_str):

    # Search for detection in day = j
    f = [idx for idx, det in enumerate(idx_day_groupby) if det == j]
    if f:
        for ff in f:
            hour = idx_hour_groupby[ff]
            M[int(hour), idx_j] = det_groupby[ff]

x_lims = mdates.date2num((begin_date, end_date))

y_lims = [0, 24]
cbarmax = 20

fig, ax = plt.subplots(figsize=(50, 15))
im = ax.imshow(M, extent=[x_lims[0], x_lims[1], y_lims[0], y_lims[1]], vmin=0, vmax=cbarmax, aspect='auto', origin='lower')

# Colorbar
cbar = fig.colorbar(im)
cbar.ax.tick_params(labelsize=30)
cbar.ax.set_ylabel('Nombre de minutes positives', rotation=270, fontsize=30, labelpad=40)

y_lims = [0, 24]

fig, ax = plt.subplots(figsize=(40, 15))
ax.imshow(M, extent=[x_lims[0], x_lims[1], y_lims[0], y_lims[1]], aspect='auto', origin='lower')

plt.plot(x_data, hour_sunrise, color='w', linewidth=4)
plt.plot(x_data, hour_sunset, color='w', linewidth=4)
ax.xaxis_date()
ax.xaxis.set_major_locator(mdate1)
ax.xaxis.set_major_formatter(mdate2)

y_pos = [0, 4, 8, 12, 16, 20, 24]
ax.set_yticks(y_pos)

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
ax.tick_params(axis='y', rotation=0, labelsize=30)
ax.tick_params(axis='x', rotation=60, labelsize=30)

ax.set_ylabel('Heure (UTC +2)', fontsize=40)
ax.set_xlabel('Date', fontsize=40)

# ax.set_xticks(time_vector_str)


# %% Multilabel plot

if len(annotators) > 1:
    annot_ref = easygui.buttonbox('Select an annotator', 'multilabel plot', annotators) if len(annotators) > 1 else annotators[0]

elif len(annotators) == 1:
    annot_ref = annotators[0]

# list of the labels corresponding to the selected user
list_labels = sorted(info[info['annotators'].apply(lambda x: annot_ref in x)]['labels'].reset_index(drop=True)[0])
# selection of the timebin
time_bin_ref = int(info[info['annotators'].apply(lambda x: annot_ref in x)]['max_time'].reset_index(drop=True).iloc[0])
# selection of the detection file
file_ref = info[info['annotators'].apply(lambda x: annot_ref in x)]['file'].reset_index(drop=True).iloc[0]

if isinstance(list_labels, str) == 0:
    selected_labels = list_labels[0:3]

    res_min = easygui.integerbox('Enter the bin size (min) ', 'Time resolution', default=10, lowerbound=1, upperbound=86400)
    delta, start_vec, end_vec = dt.timedelta(seconds=60 * res_min), t_rounder(begin_date, res=600), t_rounder(end_date + dt.timedelta(seconds=time_bin_ref), res=600)

    time_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]

    time_slice = 60 * res_min  # 10 min
    n_annot_max = time_slice / time_bin_ref  # nb of annoted time_bin max per time_slice

    bars = range(0, 110, 10)  # from 0 to 100 step 10
    y_pos = np.linspace(0, n_annot_max, num=len(bars))

    fig, ax = plt.subplots(nrows=len(selected_labels), figsize=(25, 15), facecolor='#36454F')
    fig.tight_layout(pad=10)

    for i, label in enumerate(selected_labels):

        df_1annot_1label = df_detections[(df_detections['annotator'] == annot_ref) & (df_detections['annotation'] == label)]

        ax[i].hist(df_1annot_1label['start_datetime'], bins=time_vector, color='crimson', edgecolor='black', linewidth=1)

        bars = range(0, 110, 10)  # from 0 to 100 step 10
        y_pos = np.linspace(0, n_annot_max, num=len(bars))
        ax[i].set_facecolor('#36454F')
        ax[i].set_yticks(y_pos, bars)
        ax[i].tick_params(axis='both', colors='w', rotation=0, labelsize=15)
        ax[i].tick_params(axis='x', rotation=60)
        ax[i].set_title(label, fontsize=15, color='w')
        ax[i].set_ylabel('positive detection rate\n({0} min)'.format(res_min), fontsize=15, color='w')

        ax[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz_data))
        ax[i].set_xlim(time_vector[0], time_vector[-1])
        ax[i].grid(color='w', linestyle='--', linewidth=0.2, axis='both')

    fig.suptitle('Annotator : {0}'.format(annot_ref), fontsize=25, y=0.98, color='w', weight='bold')

else: sys.exit('Multilabel plot cancelled, annotator {0} only has one label : {1}'.format(annot_ref, list_labels))

# %% Multi-user plot

if len(annotators) > 2:
    annot_ref1 = easygui.buttonbox('Select annotator 1', 'Plot label', annotators)
    annot_ref2 = easygui.buttonbox('Select an annotator', 'Plot label', [elem for elem in annotators if elem != annot_ref1])
elif len(annotators) < 2:
    sys.exit('Multi-user plot cancelled, not enough annotators to make a comparison')

else:
    annot_ref1 = annotators[0]
    annot_ref2 = annotators[1]

# list of the labels corresponding to the selected user
list_labels = info[info['annotators'].apply(lambda x: annot_ref1 in x)]['labels'].reset_index(drop=True)[0]
if isinstance(list_labels, str) == 0:
    label_ref1 = easygui.buttonbox('Select a label for annotator 1 : {0}'.format(annot_ref1), 'Single plot', list_labels)
else:
    label_ref1 = list_labels
    easygui.msgbox('Only one label available for annotator 1, {0} : {1}'.format(annot_ref1, list_labels))

list_labels = info[info['annotators'].apply(lambda x: annot_ref2 in x)]['labels'].reset_index(drop=True)[0]
if isinstance(list_labels, str) == 0:
    label_ref2 = easygui.buttonbox('Select a label for annotator 2 : {0}'.format(annot_ref2), 'Single plot', list_labels)
else:
    label_ref2 = list_labels
    easygui.msgbox('Only one label available for annotator 2, {0} : {1}'.format(annot_ref2, list_labels))

time_bin_ref1 = int(info[info['annotators'].apply(lambda x: annot_ref1 in x)]['max_time'].reset_index(drop=True).iloc[0])
time_bin_ref2 = int(info[info['annotators'].apply(lambda x: annot_ref2 in x)]['max_time'].reset_index(drop=True).iloc[0])
if time_bin_ref1 == time_bin_ref2:
    time_bin_ref = time_bin_ref1
else:
    sys.exit('The timebin of the detections {0}/{1} is {2}s whereas the timebin for {3}/{4} is {5}s!'.format(annot_ref1, label_ref1, time_bin_ref1, annot_ref2, label_ref2, time_bin_ref2))

file_ref1 = info[info['annotators'].apply(lambda x: annot_ref1 in x)]['file'].reset_index(drop=True).iloc[0]
file_ref2 = info[info['annotators'].apply(lambda x: annot_ref2 in x)]['file'].reset_index(drop=True).iloc[0]


res_min = easygui.integerbox('Enter the bin size (min) ', 'Time resolution', default=10, lowerbound=1, upperbound=86400)

delta, start_vec, end_vec = dt.timedelta(seconds=60 * res_min), t_rounder(begin_date, res=600), t_rounder(end_date + dt.timedelta(seconds=time_bin_ref), res=600)

time_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]

n_annot_max = (res_min * 60) / time_bin_ref  # max nb of annoted time_bin max per res_min slice

df1_1annot_1label = df_detections[(df_detections['annotator'] == annot_ref1) & (df_detections['annotation'] == label_ref1)]
df2_1annot_1label = df_detections[(df_detections['annotator'] == annot_ref2) & (df_detections['annotation'] == label_ref2)]

fig, axs = plt.subplots(1, 2, dpi=200, figsize=(10, 4), gridspec_kw={'width_ratios': [8, 2]})

# axs[0].set_facecolor('#36454F')
hist_plot = axs[0].hist([df1_1annot_1label['start_datetime'], df2_1annot_1label['start_datetime']], bins=time_vector, label=[annot_ref1, annot_ref2], lw=10)
axs[0].legend(loc='upper right')

bars = range(0, 110, 10)  # from 0 to 100 step 10
y_pos = np.linspace(0, n_annot_max, num=len(bars))
axs[0].set_yticks(y_pos, bars)
axs[0].tick_params(axis='x', rotation=60)
axs[0].set_ylabel('positive detection rate\n({0} min)'.format(res_min))
axs[0].tick_params(axis='y')
fig.suptitle('[{0}/{1}] VS [{2}/{3}]'.format(annot_ref1, label_ref1, annot_ref2, label_ref2), y=1.02)

axs[0].xaxis.set_major_locator(mdates.HourLocator(interval=4))
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz_data[0]))
axs[0].set_xlim(time_vector[0], time_vector[-1])
axs[0].grid(linestyle='-', linewidth=0.2, axis='both')


# accord inter-annot
list1 = list(df1_1annot_1label['start_datetime'])
list2 = list(df2_1annot_1label['start_datetime'])

unique_annotations = len([elem for elem in list1 if elem not in list2]) + len([elem for elem in list2 if elem not in list1])
common_annotations = len([elem for elem in list1 if elem in list2])
agreement = (common_annotations) / (unique_annotations + common_annotations)
axs[0].text(.05, .9, f'agreement={100 * agreement:.0f}%', transform=axs[0].transAxes)

# scatter
df_corr = pd.DataFrame(hist_plot[0] / n_annot_max, index=[annot_ref1, annot_ref2]).transpose()
sns.scatterplot(x=df_corr[annot_ref1], y=df_corr[annot_ref2], ax=axs[1])

z = np.polyfit(df_corr[annot_ref1], df_corr[annot_ref2], 1)
p = np.poly1d(z)
plt.plot(df_corr[annot_ref1], p(df_corr[annot_ref1]), lw=1)

axs[1].set_xlabel('{0}\n{1}'.format(annot_ref1, label_ref1))
axs[1].set_ylabel('{0}\n{1}'.format(annot_ref2, label_ref2))
axs[1].set_xlim(0, 1)
axs[1].set_ylim(0, 1)
axs[1].grid(linestyle='-', linewidth=0.2, axis='both')

r, p = stats.pearsonr(df_corr[annot_ref1], df_corr[annot_ref2])
axs[1].text(.05, .9, f'R²={r * r:.2f}', transform=axs[1].transAxes)

plt.tight_layout()
plt.show()
# %%
# tb=3600
# df1_test, _ = sorting_detections(file='Y:/Bioacoustique/APOCADO2/Campagne 6/PASSE PARTOUT/bouts rouges/7178/analysis/C6D3/results/APOCADO_C6D3 ST7178_results.csv',
#                                                       timebin_new=tb,
#                                                       annotation='Odontocete whistle')

# df2_test, _ = sorting_detections(file='Y:/Bioacoustique/APOCADO2/Campagne 6/PASSE PARTOUT/bouts rouges/7180/analysis/C6D3/result/APOCADO_C6D3 ST7180_results.csv',
#                                                       timebin_new=tb,
#                                                       annotation='Odontocete whistle') 

# # accord inter-annot
# list12 = list(df1_test['start_datetime'])
# list22 = list(df2_test['start_datetime'])

# unique_annotations2 = len([elem for elem in list12 if elem not in list22]) + len([elem for elem in list22 if elem not in list12])
# common_annotations2 = len([elem for elem in list12 if elem in list22])

# print('Pourcentage d\'accord pour timebin de {1:.0f}s: {0:.0f}%'.format(100 * (common_annotations2) / (unique_annotations2 + common_annotations2), tb))

