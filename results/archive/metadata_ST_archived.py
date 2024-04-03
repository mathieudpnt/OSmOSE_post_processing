import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from tqdm import tqdm
import numpy as np
import re
from collections import Counter
import pytz

os.chdir(r'U:/Documents_U/Git/post_processing_detections')
from utilities.def_func import stat_box_day, stats_diel_pattern, sorting_detections, get_season

# %% Import csv deployments

deploy = pd.read_excel(r'L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO\APOCADO - Suivi déploiements.xlsx', skiprows=[0])
deploy = deploy[deploy['check heure Raven'] == 1]
deploy = deploy.reset_index(drop=True)

deploy['durations_deployments'] = [dt.datetime.combine(deploy['date recovery'][i], deploy['time recovery'][i])\
                                   - dt.datetime.combine(deploy['date deployment'][i], deploy['time deployment'][i]) for i in range(len(deploy))]

deploy['season'] = [get_season(i) for i in deploy['date deployment']]

t_tot = sum(deploy['durations_deployments'], dt.timedelta()).total_seconds() / 3600
print('-total duration: {:.0f} h'.format(t_tot))

num_deploy = len(list(set(deploy['ID platform'])))  # number of unique platform, i.e. number of unique deployment
print(f'-number of deployments: {num_deploy}')




# %% Load all metadata files

path_json = [r'L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO',
            r'Y:\Bioacoustique\APOCADO2',
            r'Z:\Bioacoustique\DATASETS\APOCADO3']

list_json = [file_path for path in path_json for file_path in glob.glob(os.path.join(path, "**/metadata.json"), recursive=True)]

data = []
for file_path in tqdm(list_json, leave=True):
    with open(file_path, 'r') as i:
        data.append(json.load(i))
data = pd.DataFrame.from_dict(data)

data['df_detections'] = [sorting_detections(file=data['pamguard detection file'][i], timebin_new=data['pamguard timebin'].tolist()[i])[0] for i in tqdm(range(len(data)))]
data['season'] = [get_season(i) for i in [pd.to_datetime(d) for d in data['beg_deployment']]]

for i in range(len(data)):
    data['df_detections'][i]['start_deploy'] = pd.to_datetime(data['beg_deployment'][i])
    data['df_detections'][i]['end_deploy'] = pd.to_datetime(data['end_deployment'][i])

deploy['detection_num'] = [len(data.loc[data['deploy_ID'] == ID, 'df_detections'].reset_index(drop=True)[0]) for ID in deploy['ID']]

deploy['detection_rate'] = [((len(data[data['deploy_ID'] == ID].reset_index(drop=True)['df_detections'][0]) * data[data['deploy_ID'] == ID].reset_index(drop=True)['timebin'][0]) / deploy['durations_deployments'][i].total_seconds()) * 100 for i, ID in enumerate(deploy['ID'])]


deploy['dt_begin'] = [dt.datetime.strftime(dt.datetime.combine(deploy['Date début déploiement'][i], deploy['Heure début déploiement'][i]), '%d/%m/%Y %H:%M:%S') for i in range(len(deploy))]
deploy['dt_end'] = [dt.datetime.strftime(dt.datetime.combine(deploy['Date fin déploiement'][i], deploy['Heure fin déploiement'][i]), '%d/%m/%Y %H:%M:%S') for i in range(len(deploy))]
# %% Positive detection rate for whistles and 10s windows

print(f'\n-positive detection rate : {np.mean(deploy["detection_rate"]):.1f}%')

print('\n-observation effort per season :')
for i in list(dict.fromkeys(deploy['season'])):
    print('\t{0} : {1:.0f}%'.format(i, np.mean(deploy[deploy['season'] == i]['detection_rate'])))

# %% Distribution of elapsed time to achieve 75% of the detections

test = data['threshold_perc_elapsed']
np.mean(test)
fig, ax = plt.subplots(figsize=(15, 9))
ax.hist(test, bins=25, density=True)
plt.xlabel('Threshold 75% elapsed')
plt.ylabel('Density')
plt.title('% of time elapsed to achieve 75% of the detections')
plt.grid(True)
plt.show()

# %% Distribution deployment time



test = [t/3600 for t in data['duration_deployment (s)']]
fig, ax = plt.subplots(figsize=(9, 9), facecolor='#36454F')
ax.set_facecolor('#36454F')
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', colors='w')
ax.hist(test, bins=range(0,140,2))
plt.xlabel('Hours', fontsize=16, color='w')
plt.ylabel('Deployments', fontsize=16, color='w')
plt.title('Distribution of deployements durations', color='w')
plt.grid(visible=False)
plt.show()

# %% Recap deployments csv

print('\n#### RESULTS DEPLOYMENTS ####')
# t_tot : duration_deployement divided by 2 if 2 ST are used in the deployment
t_tot = int(sum(deploy[deploy['nb ST/filet'] == 1]['durations_deployments'], dt.timedelta()).total_seconds() / 3600) +\
    0.5 * int(sum(deploy[deploy['nb ST/filet'] != 1]['durations_deployments'], dt.timedelta()).total_seconds() / 3600)
print('-total duration: {:.0f} h'.format(t_tot))

print('\n# Observation effort per season #')
# [print('-{0}:'.format(season), int(sum(deploy[deploy['season']== season]['durations_deployments'], dt.timedelta()).total_seconds()/3600),'h') for season in ['spring', 'summer', 'autumn', 'winter']];
deploy['season_year'] = [int(s[-4:]) for s in deploy['season']]
for y in sorted(list(set(deploy['season_year']))):
    for s in list(dict.fromkeys(deploy[(deploy['season_year'] == y)]['season'])):
        # t : duration_deployement divided by 2 if 2 ST are used in the deployment
        t = int(sum(deploy[(deploy['season_year'] == y) & (deploy['season'] == s) & (deploy['nb ST/filet'] == 1)]['durations_deployments'], dt.timedelta()).total_seconds() / 3600) + int(sum(deploy[(deploy['season_year'] == y) & (deploy['season'] == s) & (deploy['nb ST/filet'] != 1)]['durations_deployments'], dt.timedelta()).total_seconds() / 3600) * 0.5
        print('-{0}: {1:.0f}% - {2:.0f} h'.format(s, 100 * (t / t_tot), t))

print('\n# Duration per net #')
[print('-Filet {0}:'.format(filet), int(sum(deploy[(deploy['Filet'] == filet) & (deploy['nb ST/filet'] == 1)]['durations_deployments'], dt.timedelta()).total_seconds() / 3600) + 0.5 * int(sum(deploy[(deploy['Filet'] == filet) & (deploy['nb ST/filet'] != 1)]['durations_deployments'], dt.timedelta()).total_seconds() / 3600), 'h') for filet in sorted(list(set(deploy['Filet'])))];

print('\n# Mean duration of a deployment per net type #')
[print('-Filet {0}:'.format(filet), round(np.mean(deploy[deploy['Filet'] == filet]['durations_deployments']).total_seconds() / 3600, 1), 'h +/-', round(np.std(deploy[deploy['Filet'] == filet]['durations_deployments']).total_seconds() / 3600, 1), 'h') for filet in sorted(list(set(deploy['Filet'])))];

print('\n# Duration per net length #')
[print('-{:.0f}'.format(L), 'm :', int(sum(deploy[deploy['Longueur (m)'] == L]['durations_deployments'], dt.timedelta()).total_seconds() / 3600), 'h') for L in sorted(list(set(deploy['Longueur (m)'])))];

# %% Write metadata files

# list_csv = glob.glob(os.path.join('L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO', '**/PG_rawdata**.csv'), recursive=True)\
#              + glob.glob(os.path.join('L:/acoustock2/Bioacoustique/APOCADO2', '**/PG_rawdata**.csv'), recursive=True)

# for i in tqdm(list_csv):

#     tb = 10
#     df_detections, t_detections = sorting_detections(i, timebin_new=tb)

#     fmax = int(list(set(t_detections['max_freq']))[0])
#     time_bin = int(list(set(t_detections['max_time']))[0])
#     labels = list(set(t_detections['labels'].explode()))
#     annotators = list(set(t_detections['annotators'].explode()))
#     [tz] = list(set([dt.tz for dt in df_detections['start_datetime']]))

#     [ID0] = list(set(df_detections['dataset']))
#     ID1 = re.search(r'C\d{1,2}D\d{1,2}', ID0).group()  # campaign and deployment identifier
#     ID2 = re.search(r'ST\d+', ID0).group()  # instrument identifier
#     ID_detections = ID1 + ' ' + ID2

#     rank = [i for i, ID in enumerate(deploy['ID']) if ID in ID_detections][0]
#     duration_deployment = int(deploy['durations_deployments'][rank].total_seconds())
#     dt_deployment_beg = pd.Timestamp(dt.datetime.combine(deploy['Date début déploiement'][rank], deploy['Heure début déploiement'][rank]), tz=tz)
#     dt_deployment_end = pd.Timestamp(dt.datetime.combine(deploy['Date fin déploiement'][rank], deploy['Heure fin déploiement'][rank]), tz=tz)

#     net_len = int(deploy['Longueur (m)'][rank])
#     n_instru = deploy['nb ST/filet'][rank] if type(deploy['nb ST/filet'][rank]) is int else int(deploy['nb ST/filet'][rank][0])

#     wav_files = glob.glob(os.path.join(Path(i).parents[2], "**/*.wav"), recursive=True)
#     wav_names = [os.path.basename(file) for file in wav_files]
#     [wav_folder] = list(set([os.path.dirname(file) for file in wav_files]))

#     arg1 = [extract_datetime(dt_from_filename, tz=tz) for dt_from_filename in df_detections['filename']]
#     arg2 = [extract_datetime(wav_name, tz=tz) for wav_name in wav_names]
#     test_wav = [j in arg1 for j in arg2]

#     wav_names, wav_files = zip(*[(wav_names[i], wav_files[i]) for i in range(len(wav_names)) if test_wav[i]])  # only the wav files corresponding to the detections are kept
#     durations = [read_header(file)[-1] for file in wav_files]

#     lat = deploy['Latitude'][rank]
#     lon = deploy['Longitude'][rank]

#     threshold = 0.75
#     total_detections = len(df_detections)
#     threshold_detect = round(total_detections * threshold)
#     bins = pd.date_range(start=dt_deployment_beg, end=dt_deployment_end, freq='1min')
#     hist, _ = np.histogram([dt.timestamp() for dt in df_detections['start_datetime']], bins=[b.timestamp() for b in bins])
#     cumul_count = np.cumsum(hist)
#     bin_index = int(np.argmax(cumul_count >= threshold_detect))
#     dt_thr = bins[bin_index]  # datetime of the threshold
#     min_elapsed = bin_index  # elapsed time in minutes to achieve threshold_detect, freq bins is 1min
#     perc_elapsed = round((dt_thr - bins[0]) / (bins[-1] - bins[0]), 2)  # % of elapsed time to achieve threshold_detect

#     metadata = {'deploy_ID': ID_detections,
#                 'wav_folder': wav_folder,
#                 'detection_file': i,
#                 'beg_deployment': dt_deployment_beg.strftime('%Y-%m-%dT%H:%M:%S%z'),
#                 'end_deployment': dt_deployment_end.strftime('%Y-%m-%dT%H:%M:%S%z'),
#                 'Latitude': lat,
#                 'Longitude': lon,
#                 'duration_deployment (s)': duration_deployment,
#                 'fmax': fmax,
#                 'annotators': annotators,
#                 'labels': labels,
#                 'detections number': total_detections,

#                 'threshold': threshold,
#                 'threshold_number_detections': threshold_detect,
#                 'threshold_elapsed_time (min)': min_elapsed,
#                 'threshold_perc_elapsed': perc_elapsed,

#                 'timebin': time_bin,
#                 'net_length (m)': net_len,
#                 'n_instru': n_instru,
#                 'wav_path': list(wav_files),
#                 'durations': durations}

#     out_file = open(os.path.join(Path(i).parents[0], 'metadata.json'), 'w+')
#     json.dump(metadata, out_file, indent=4)
#     out_file.close()

# %% Distribution of observation effort per net length
# if 2 ST are present on the net, the duration is divided by 2

x = [str(elem) for elem in sorted(list(set(deploy['Longueur (m)'])))]
y = [int(sum(deploy[(deploy['Longueur (m)'] == L) & (deploy['nb ST/filet'] != 1)]['durations_deployments'], dt.timedelta()).total_seconds() * 0.5 / 3600) + int(sum(deploy[(deploy['Longueur (m)'] == L) & (deploy['nb ST/filet'] == 1)]['durations_deployments'], dt.timedelta()).total_seconds() / 3600) for L in sorted(list(set(deploy['Longueur (m)'])))]

fig, ax = plt.subplots(figsize=(9, 9), facecolor='#36454F')
ax.bar(x, y)
ax.set_facecolor('#36454F')
ax.tick_params(axis='both', colors='w')
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(visible=False)
plt.suptitle('', color='w')
ax.set_ylabel('Heures d\'enregistrement', fontsize=16, color='w')
ax.set_xlabel('Longueur filière [m]', fontsize=16, color='w')
plt.title('Distribution of net lengths', color='w')


# %% Cumulated histogram of detections for a single detection file

df_detections = data['df_detections'][100]

# from utilities.def_func import get_csv_file
# import pytz
# # data = get_csv_file(1)
# df_detections, _ = sorting_detections(files=data, timebin_new=10, tz=pytz.FixedOffset(60), label='Odontocete buzz')

threshold = 0.75

[ID0] = list(set(df_detections['dataset']))
ID1 = re.search(r'C\d{1,2}D\d{1,2}', ID0).group()  # campaign and deployment identifier
ID2 = re.search(r'ST\d+', ID0).group()  # instrument identifier
ID_test = ID1 + ' ' + ID2

data_histo = df_detections['start_datetime']  # detections datetimes
tb = df_detections['end_time'][0]  # timebin
deploy_dt = [df_detections['start_deploy'][0], df_detections['end_deploy'][0]]  # beginning and end of deployment
# deploy_dt = [pd.Timestamp('2023-02-11 12:00:00 +0100'), pd.Timestamp('2023-02-12 09:00:00 +0100')]  # beginning and end of deployment

res_min = 1
delta, start_vec, end_vec = dt.timedelta(seconds=60 * res_min), deploy_dt[0], deploy_dt[1]
bins = pd.date_range(start=deploy_dt[0], end=deploy_dt[1], freq='1min')
n_annot_max = (res_min * 60) / tb  # max nb of annoted time_bin max per res_min slice

total_detections = len(data_histo)
threshold_detect = round(total_detections * threshold)

hist, _ = np.histogram([dt.timestamp() for dt in data_histo], bins=[b.timestamp() for b in bins])
cumul_count = np.cumsum(hist)
bin_index = int(np.argmax(cumul_count >= threshold_detect))
dt_thr = bins[bin_index]  # datetime of the threshold
min_elapsed = bin_index  # elapsed time in minutes to achieve threshold_detect, freq bins is 1min
perc_elapsed = round((dt_thr - bins[0]) / (bins[-1] - bins[0]), 2)  # % of elapsed time to achieve threshold_detect

bin_height = cumul_count[bin_index] / total_detections

min_elapsed = bin_index * res_min  # elapsed time in minutes to achieve threshold_detect
perc_elapsed = (dt_thr - bins[0]) / (bins[-1] - bins[0])  # % of elapsed time to achieve threshold_detect

fig, ax = plt.subplots(figsize=(20, 9))
ax.hist(data_histo, bins, cumulative=True)  # histo
# vline = ax.axvline(x=dt_thr, ymax = bin_height*0.92 ,color='r', linestyle='-', label='75% of Detections')
vline = ax.axvline(x=dt_thr, ymax=1, color='r', linestyle='-', label='75% of Detections')
# hline = ax.axhline(y=threshold_detect, xmax = (dt_thr-bins[0])/(bins[-1]-bins[0]), color='r', linestyle='-', label='75% of Detections')
hline = ax.axhline(y=threshold_detect, xmax=1, color='r', linestyle='-', label='75% of Detections')
ax.set_ylabel("Detections", fontsize=20)
ax.grid(color='k', linestyle='-', linewidth=0.2, axis='both')
ax.text(bins[bin_index + 10], 50, '{0} min - {1:.0f}%'.format(min_elapsed, 100 * perc_elapsed), color='r', ha='left', fontsize=16, fontweight='bold')
ax.text(bins[0], threshold * total_detections * 1.02 + 2, '{0} detections'.format(threshold_detect), color='r', ha='left', fontsize=16, fontweight='bold')
tz_data = deploy_dt[0].tz
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz_data))
fig.suptitle('Cumulated detections - {0}\n{1} - {2}'.format(ID_test, deploy_dt[0].strftime('%d/%m/%y'), deploy_dt[1].strftime('%d/%m/%y')), fontsize=24, y=0.98)

# %% Histogram of the detections for a single detection file
# the histogram is divided into n periods of equal durations
# the periods number with the most/least detections are printed

data_test = data.iloc[100]
n_periods = 10

df_detections = data_test['df_detections']
# df_detections, _ = sorting_detections(files=data, timebin_new=10, tz=pytz.FixedOffset(60), label='Odontocete buzz')

data_histo = df_detections['start_datetime']
periods = pd.date_range(start=data_test['beg_deployment'], end=data_test['end_deployment'], freq=str(int(data_test['duration_deployment (s)'] / n_periods)) + 's')
# periods = pd.date_range(start=pd.Timestamp('2023-02-11 12:00:00 +0100'), end=pd.Timestamp('2023-02-12 09:00:00 +0100'), freq=str(int((end-start).total_seconds() / n_periods)) + 's')

fig, ax = plt.subplots(figsize=(20, 9), facecolor='#36454F')
ax.set_facecolor('#36454F')
detect_periods, _, bars = ax.hist(data_histo, periods, edgecolor='None')

# spines
ax.spines['right'].set_color('w')
ax.spines['top'].set_color('w')
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')

ax.tick_params(axis='both', colors='w')

# grids
ax.yaxis.grid(color='gray', linestyle='--')

# x-labels
ax.set_xlabel('Periods', color='w', fontsize=20)

[ax.axvline(x=period, color='lime') for period in periods]
ax.grid(color='k', linestyle='-', linewidth=0.1, axis='both')

max_value = np.max(detect_periods)
rank_max = np.argmax(detect_periods) + 1
rank_min = np.argmin(detect_periods) + 1
bars[rank_max - 1].set_color('orange')
bars[rank_min - 1].set_color('#CD3333')

ax.set_ylabel("Detections", fontsize=20, color='w')

perc_max = detect_periods[rank_max - 1] / data_test['detections number']
perc_min = detect_periods[rank_min - 1] / data_test['detections number']
fig.suptitle('Odontocete whistle', fontsize=24, y=0.98, color='w')

print(f'\nThe period {rank_max}/{n_periods} contains the most detections with {detect_periods[rank_max-1]:.0f}/{data_test["detections number"]} detections ({100*perc_max:.0f}%)')
print(f'\nThe period {rank_min}/{n_periods} contains the least detections with {detect_periods[rank_min-1]:.0f}/{data_test["detections number"]} detections ({100*perc_min:.0f}%)')

# %% Distribution of all the detection dataframes most populated time period,
# first, each detection dataframe is divided into n periods of equal durations
# Then the distribution of the period with the most detections for each df is plotted

# data2 = data[data['season'] == 'spring 2023']
data2 = data[data['n_instru'] == 2].reset_index(drop=True)
data3 = data2[data2.index % 2 == 1]
# TODO : ajouter filtering

n_periods = 10

rank_max, rank_min = [], []
for i in range(len(data3)):
    data_test = data3.iloc[i]
    df_detections = data_test['df_detections']
    data_histo = df_detections['start_datetime']
    periods = pd.date_range(start=data_test['beg_deployment'], end=data_test['end_deployment'], freq=str(int(data_test['duration_deployment (s)'] / n_periods)) + 's')
    detect_periods, _ = np.histogram(data_histo, periods)
    rank_max.append(np.argmax(detect_periods) + 1)
    rank_min.append(np.argmin(detect_periods) + 1)

total = len(data3)
value_max_counts = Counter(rank_max)
value_min_counts = Counter(rank_min)

# Fill in the counts for null values
for value in range(1, n_periods + 1, 1):
    if value not in value_max_counts:
        value_max_counts[value] = 0
    if value not in value_min_counts:
        value_min_counts[value] = 0

value_max_counts = dict(sorted(value_max_counts.items()))
value_min_counts = dict(sorted(value_min_counts.items()))

count_max = [max_c for max_c in list(value_max_counts.values())]
count_min = [min_c for min_c in list(value_min_counts.values())]
values = list(value_max_counts.keys())

print("\nPeriod | Max | Min")
print("-------------------")
[print(f"{value:6} | {i:3} | {j:2}") for value, i, j in zip(values, count_max, count_min)]
print(f" Total |{total:7}")

fig, (ax, ax2) = plt.subplots(figsize=(15, 10), facecolor='#36454F', nrows=2)
ax.set_facecolor('#36454F')
ax2.set_facecolor('#36454F')

# bar plots
ax.bar(values, count_max, align='center', alpha=0.8, edgecolor='w', color='orange')
ax2.bar(values, count_min, align='center', alpha=0.8, edgecolor='w', color='teal')

# spines
ax.spines['right'].set_color('w')
ax.spines['top'].set_color('w')
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')
ax2.spines['right'].set_color('w')
ax2.spines['top'].set_color('w')
ax2.spines['bottom'].set_color('w')
ax2.spines['left'].set_color('w')

# grids
ax.yaxis.grid(color='gray', linestyle='--')
ax2.yaxis.grid(color='gray', linestyle='--')

# x-labels
ax.set_xlabel('Periods', color='w')
ax2.set_xlabel('Periods', color='w')

# titles
ax.set_title('Distribution of the most populated time periods', color='w')
ax2.set_title('Distribution of the least populated time periods', color='w')

# ticks
ax.set_xticks(values)
ax2.set_xticks(values)
ax.set_yticks(range(0, max(count_max) + 1, 3))
ax2.set_yticks(range(0, max(count_min) + 1, 3))
ax.tick_params(axis='both', colors='w')
ax2.tick_params(axis='both', colors='w')

plt.show()

# %% export csv for QGIS
# deploy_out = deploy.drop(columns=['lat D', 'lat DM', 'lat DD', 'long D', 'long DM', 'long DD', 'Date début campagne', 'Date fin campagne', 'Heure début campagne', 'Heure fin campagne', 'check heures', 'Conditions météo', 'Conditions météo.1', 'Présence cétacés', 'Présence cétacés.1', 'Commentaire'])

# for i in range(len(deploy['nb ST/filet'])):
#     if '1' in str(deploy['nb ST/filet'][i]): deploy_out['nb ST/filet'][i] = 1
#     if '2' in str(deploy['nb ST/filet'][i]): deploy_out['nb ST/filet'][i] = 2

# deploy_out.to_csv('L:/acoustock/Bioacoustique/DATASETS/APOCADO/Code/Data QGIS/APOCADO - Suivi déploiements 09102023.csv', index=False, encoding='latin1')

# %% Mean number of detection per hour of the day

# filtering = data[(data['deploy_ID'].str.contains('C6')) & (data['deploy_ID'].str.contains('7179'))]
# filtering = data[data['deploy_ID'].str.contains('C3')]
# filtering = data[data['season'] == 'spring 2023']
filtering = data[data['season'] == 'winter 2022']

idx = list(filtering.index)
detection_files = data['detection_file'][idx]
detection_files2 = list(detection_files)
[tb] = list(set(data['timebin']))

df_final = pd.DataFrame()
# for i in range(len(detection_files2)):
for i in tqdm(range(len(detection_files2))):
    df_detections, _ = sorting_detections(detection_files2[i], timebin_new=tb)
    df_final = df_final.append(df_detections)

df_detections, _ = sorting_detections(detection_files, timebin_new=tb)
# df_detections, _ = sorting_detections(files=data, timebin_new=10, tz=pytz.FixedOffset(60), label='Odontocete buzz')


df_result = stat_box_day(filtering, df_detections)

hour_list = ['{:02d}:00'.format(i) for i in range(24)]
hour_list.append('00:00')

df_detections['date'] = [dt.datetime.strftime(i.date(), '%d/%m/%Y') for i in df_detections['start_datetime']]
df_detections['season'] = [get_season(i) for i in df_detections['start_datetime']]
df_detections['dataset'] = [i.replace('_', ' ') for i in df_detections['dataset']]

# result = {}
# list_dates = sorted(list(set(df_detections['date'])))  # list of dates
# for date in list_dates:
#     detection_bydate = df_detections[df_detections['date'] == date]  # sub-dataframe : per date
#     list_datasets = sorted(list(set(detection_bydate['dataset'])))  # dataset list for date=date

#     for dataset in list_datasets:
#         df = detection_bydate[detection_bydate['dataset'] == dataset].set_index('start_datetime')  # sub-dataframe : per date & per dataset

#         # number of detections per hour of the day at date and at dataset
#         detection_per_dataset = [len(df.between_time(hour_list[j], hour_list[j + 1], inclusive='left')) for j in (range(len(hour_list) - 1))]

#         deploy_beg_ts, deploy_end_ts = int(pd.Timestamp('2023-02-11 12:00:00 +0100').timestamp()), int(pd.Timestamp('2023-02-12 09:00:00 +0100').timestamp())  # beginning and end of deployment

#         list_present_h = [dt.datetime.fromtimestamp(i) for i in list(range(deploy_beg_ts, deploy_end_ts, 3600))]
#         list_present_h2 = [dt.datetime.strftime(list_present_h[i], '%d/%m/%Y %H') for i in range(len(list_present_h))]

#         list_deploy_d = sorted(list(set([dt.datetime.strftime(dt.datetime.fromtimestamp(i), '%d/%m/%Y') for i in list(range(deploy_beg_ts, deploy_end_ts, 3600))])))
#         list_deploy_d2 = [d for i, d in enumerate(list_deploy_d) if d in date][0]

#         list_present_h3 = []
#         for item in list_present_h2:
#             if item.startswith(list_deploy_d2):
#                 list_present_h3.append(item)

#         list_deploy = [df['date'][0] + ' ' + n for n in [f'{i:02}' for i in range(0, 24)]]

#         for i, h in enumerate(list_deploy):
#             if h not in list_present_h3:
#                 detection_per_dataset[i] = np.nan

#         result[dataset, date] = detection_per_dataset

# df_result = pd.DataFrame(result).T




fig, ax = plt.subplots(figsize=(10, 6), facecolor='#36454F')
ax.set_facecolor('#36454F')
positions = np.arange(0.5, 25 - 1 + 0.5)
hour_list = ['{:02d}:00'.format(i) for i in range(24)]
hour_list.append('00:00')

ax.boxplot([df_result[i].dropna().tolist() for i in df_result],
           positions=positions,
           widths=1,
           labels=hour_list[:-1],
           patch_artist=True,
           notch=False,
           showfliers=False,
           boxprops=dict(facecolor='#769dc8', color='#437ab4', linewidth=2),
           capprops=dict(color='#437ab4', linewidth=2),
           medianprops=dict(color='#437ab4', linewidth=2),
           flierprops=dict(markeredgecolor='#437ab4', linewidth=2),
           whiskerprops=dict(color='#437ab4', linewidth=2))

ax.tick_params(axis='both', colors='w')
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')
ax.spines['right'].set_color('w')
ax.spines['top'].set_color('w')
plt.grid(visible=False)

plt.xticks(range(len(hour_list)), hour_list, rotation=90)
plt.title('Mean detection number per hour of the day - Odontocete buzz', color='w', fontsize=14)
plt.xlabel('Hours of the day', fontsize=14, color='w')
plt.ylabel('Detections', fontsize=14, color='w')

plt.show()

# %% stat diel plot

mode = 'solo'
if mode == 'solo':
    i = [11]
    # data_test = data.iloc[i].iloc[0]
    # df_detections = data_test['df_detections']
    df_detections, _ = sorting_detections(file=data['detection_file'][i], timebin_new=10, tz=pytz.FixedOffset(60), annotation='Odontocete whistle')
    

    # begin_deploy = pd.to_datetime(data_test['beg_deployment'], format='%Y-%m-%dT%H:%M:%S%z')
    # end_deploy = pd.to_datetime(data_test['end_deployment'], format='%Y-%m-%dT%H:%M:%S%z')
    # duration = pd.to_timedelta(data_test['duration_deployment (s)'], unit='s')
    begin_deploy = pd.Timestamp('2023-02-11 12:00:00 +0100')
    end_deploy = pd.Timestamp('2023-02-12 09:00:00 +0100')
    duration = pd.to_timedelta((end_deploy - begin_deploy).total_seconds(), unit='s')

    # lat = data_test['Latitude']
    # lon = data_test['Longitude']
    lat = 47.862
    lon = -4.502

elif mode == 'multiple':
    # filtering = data[data['season'] == 'spring 2023']
    filtering = data[(data['deploy_ID'].str.contains('C6')) & (data['deploy_ID'].str.contains('7179'))]
    # filtering = data[data['deploy_ID'].str.contains('C7')]['deploy_ID']
    # filtering = data[(data['duration_deployment (s)'] > 86400)]['deploy_ID']
    # filtering = data['deploy_ID']

    idx = filtering.index

    detection_files = data['detection_file'][idx]
    [tb] = list(set(data['timebin']))
    df_detections, _ = sorting_detections(detection_files, timebin_new=tb)
    begin_deploy = pd.to_datetime(min(data['beg_deployment'][idx]), format='%Y-%m-%dT%H:%M:%S%z')
    end_deploy = pd.to_datetime(max(data['beg_deployment'][idx]), format='%Y-%m-%dT%H:%M:%S%z')
    duration = pd.to_timedelta(sum(data['duration_deployment (s)'][idx]), unit='s')
    lat = np.mean(data['Latitude'][idx])
    u_lat = np.std(data['Latitude'][idx])
    lon = np.mean(data['Longitude'][idx])
    u_lon = np.std(data['Longitude'][idx])

lr, BoxNames = stats_diel_pattern(df_detections=df_detections, begin_date=begin_deploy, end_date=end_deploy, lat=lat, lon=lon)
LR = lr[(lr[BoxNames[0]] != 0) & (lr[BoxNames[1]] != 0) & (lr[BoxNames[2]] != 0) & (lr[BoxNames[3]] != 0)]

fig, ax = plt.subplots(facecolor='#36454F')
ax.set_facecolor('#36454F')
ax.tick_params(axis='both', colors='w')
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')
ax.spines['right'].set_color('w')
ax.spines['top'].set_color('w')
ax.grid(visible=False)

ax.boxplot(x=LR,
           patch_artist=True,
           notch=False,
           showfliers=False,
           boxprops=dict(facecolor='#899ded', color='#6e58e5', linewidth=3),
           capprops=dict(color='#6e58e5', linewidth=3),
           medianprops=dict(color='#6e58e5', linewidth=3),
           flierprops=dict(markeredgecolor='#6e58e5', linewidth=3),
           whiskerprops=dict(color='#6e58e5', linewidth=3))
plt.xticks([1, 2, 3, 4], BoxNames)

print(f'\nbegin date : {begin_deploy}')
print(f'end date : {end_deploy}')
print(f'duration : {duration}')
# print(f'latitude : {lat:.2f} +/- {u_lat:.2f}')
# print(f'longitude : {lon:.2f} +/- {u_lon:.2f}')

plt.title('Odontocete buzz', color='w', fontsize=14)


# %% Mean number of detection per hour of the day + stat diel plot

print('\nComputing data ...', end=' ')
# filtering = data[data['season'] == 'winter 2022']
# filtering = data[(data['deploy_ID'].str.contains('C6')) & (data['deploy_ID'].str.contains('7179'))]
filtering = data[(data['deploy_ID'].str.contains('C6'))]

idx = list(filtering.index)
detection_files = data['detection_file'][idx]
[tb] = list(set(data['timebin']))
df_detections, _ = sorting_detections(detection_files, timebin_new=tb)

begin_deploy = pd.to_datetime(min(data['beg_deployment'][idx]), format='%Y-%m-%dT%H:%M:%S%z')
end_deploy = pd.to_datetime(max(data['beg_deployment'][idx]), format='%Y-%m-%dT%H:%M:%S%z')
duration = pd.to_timedelta(sum(data['duration_deployment (s)'][idx]), unit='s')
lat = np.mean(filtering['Latitude'][idx])
lon = np.mean(filtering['Longitude'][idx])
# LR
lr, BoxNames = stats_diel_pattern(df_detections=df_detections, begin_date=begin_deploy, end_date=end_deploy, lat=lat, lon=lon)
LR = lr[(lr[BoxNames[0]] != 0) & (lr[BoxNames[1]] != 0) & (lr[BoxNames[2]] != 0) & (lr[BoxNames[3]] != 0)]

# mean detections per hour of the day
df_result = stat_box_day(filtering, df_detections)
print('Done!', end='\n')
print('\nPlotting results ...', end=' ')

fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10), facecolor='#36454F')
ax1.set_facecolor('#36454F')
ax2.set_facecolor('#36454F')

for ax in [ax1, ax2]:
    ax.tick_params(axis='both', colors='w')
    ax.spines['bottom'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.grid(visible=False)

# subfigure 1
ax1.boxplot(x=LR,
            patch_artist=True,
            notch=False,
            showfliers=False,
            boxprops=dict(facecolor='#899ded', color='#6e58e5', linewidth=4),
            capprops=dict(color='#6e58e5', linewidth=3),
            medianprops=dict(color='#6e58e5', linewidth=3),
            flierprops=dict(markeredgecolor='#6e58e5', linewidth=3),
            whiskerprops=dict(color='#6e58e5', linewidth=3))

ax1.set_xticks([1, 2, 3, 4], BoxNames, color='w')
# ax1.set_xlabel('Light Regime', fontsize=12, color='w')
ax1.set_ylabel('Detection proportion', fontsize=12, color='w')

# subfigure 2
positions_boxes = np.arange(0.5, 25 - 1 + 0.5)
hour_list = ['{:02d}:00'.format(i) for i in range(24)]
hour_list.append('00:00')

ax2.boxplot([df_result[i].dropna().tolist() for i in df_result],
            positions=positions_boxes,
            widths=1,
            labels=hour_list[:-1],
            patch_artist=True,
            notch=False,
            showfliers=False,
            boxprops=dict(facecolor='#769dc8', color='#437ab4', linewidth=2),
            capprops=dict(color='#437ab4', linewidth=2),
            medianprops=dict(color='#437ab4', linewidth=2),
            flierprops=dict(markeredgecolor='#437ab4', linewidth=2),
            whiskerprops=dict(color='#437ab4', linewidth=2))

ax2.set_xticks(range(len(hour_list)), hour_list, rotation=90)
# ax2.set_xlabel('Hours of the day', fontsize=12, color='w')
ax2.set_ylabel('Detections', fontsize=12, color='w')

plt.show()
print('Done!', end='\n')
print(f'\nbegin date : {begin_deploy}')
print(f'end date : {end_deploy}')
print(f'duration : {duration}')
