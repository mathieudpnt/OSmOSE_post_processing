import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from tqdm import tqdm
import numpy as np
from collections import Counter
import glob
import json
from scipy.stats import norm

os.chdir(r'U:/Documents_U/Git/post_processing_detections')
from utilities.def_func import stat_box_day, stats_diel_pattern, sorting_detections, get_season

# %%
detector = ['pamguard', 'thalassa']
arg = ['season', 'net', 'all']
timebin1 = 60  # en seconde
timebin2 = 1  # en minute

# %% Import csv deployments

deploy = pd.read_excel('L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/APOCADO - Suivi déploiements.xlsx', skiprows=[0])
deploy = deploy[deploy['check heure Raven'] == 1]
deploy = deploy.reset_index(drop=True)

deploy['datetime deployment'] = [pd.Timestamp.combine(deploy['date deployment'][i], deploy['time deployment'][i]) for i in range(len(deploy))]
deploy['datetime recovery'] = [pd.Timestamp.combine(deploy['date recovery'][i], deploy['time recovery'][i]) for i in range(len(deploy))]
deploy['duration'] = [deploy['datetime recovery'][i] - deploy['datetime deployment'][i] for i in range(len(deploy))]
deploy['season_y'] = [get_season(i) for i in deploy['datetime deployment']]
deploy['season'] = [i.split(' ')[0] for i in deploy['season_y']]
deploy['year'] = [int(s[-4:]) for s in deploy['season_y']]

# result deployments
print('\n# Observation effort #')
t_tot = (deploy['duration'] / deploy['recorder number']).sum().total_seconds() / 3600
print(f'-data collected: {t_tot:.0f} h')
t_tot2 = round(deploy['duration'].sum().total_seconds() / 3600)
print(f'-data collected (with ST appaired): {t_tot2:.0f} h')

print('\n# Observation effort per season #')
list_season = ['spring', 'summer', 'autumn', 'winter']
for s in list_season:
    t = (deploy[deploy['season'] == s]['duration'] / deploy[deploy['season'] == s]['recorder number']).sum().total_seconds() / 3600
    print(f'-{s}: {100 * (t / t_tot):2.0f}% - {t:4.0f} h')

print('\n# Observation effort per season & per year #')
list_year = sorted(list(set(deploy['year'])))
for y in list_year:
    list_season_y = list(dict.fromkeys(deploy[(deploy['year'] == y)]['season_y']))
    for sy in list_season_y:
        t = (deploy[deploy['season_y'] == sy]['duration'] / deploy[deploy['season_y'] == sy]['recorder number']).sum().total_seconds() / 3600
        print(f'-{sy}: {100 * (t / t_tot):2.0f}% - {t:4.0f} h')

print('\n# Duration per net #')
net_type = sorted(list(set(deploy['net'])))
for net in net_type:
    duration_net = (deploy[deploy['net'] == net]['duration'] / deploy[deploy['net'] == net]['recorder number']).sum().total_seconds() / 3600
    print(f'-{net}: {duration_net:2.0f}h')

print('\n# Mean duration of a deployment per net type #')
for net in net_type:
    duration_net_mean = (deploy[deploy['net'] == net]['duration'] / deploy[deploy['net'] == net]['recorder number']).mean().total_seconds() / 3600
    duration_net_std = (deploy[deploy['net'] == net]['duration'] / deploy[deploy['net'] == net]['recorder number']).std().total_seconds() / 3600
    print(f"{net}: {duration_net_mean:.0f}h +/- {duration_net_std:.0f}h")

print('\n# Duration per net length #')
[print('-{:.0f}'.format(L), 'm :', int(sum(deploy[deploy['net length'] == L]['duration'], dt.timedelta()).total_seconds() / 3600), 'h') for L in sorted(list(set(deploy['net length'])))]

# distribution of deployment durations
test = [t.total_seconds() / 3600 for t in deploy['duration']]
fig, ax = plt.subplots(figsize=(18, 9), facecolor='#36454F')
ax.set_facecolor('#36454F')
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', colors='w', labelsize=14)
ax.hist(test, bins=range(0, int(1.05 * max(test)), 2))
plt.xlabel('Hours', fontsize=16, color='w')
plt.ylabel('Deployment number', fontsize=16, color='w')
plt.title('Distribution of deployment durations', color='w', fontsize=18)
ax.grid(axis='y', linestyle='--', color='w', alpha=0.5)

# distribution of net lengths
# if 2 ST are present on the net, the duration is divided by 2
net_length = sorted(list(set(deploy['net length'])))
x = [str(elem) for elem in net_length]
y = [(deploy[deploy['net length'] == L]['duration'] / deploy[deploy['net length'] == L]['recorder number']).sum().total_seconds() / 3600 for L in net_length]
fig, ax = plt.subplots(figsize=(18, 9), facecolor='#36454F')
plt.title('Distribution of net lengths', color='w', fontsize=18)
ax.bar(x, y)
ax.set_facecolor('#36454F')
ax.tick_params(axis='both', colors='w', labelsize=14)
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Data collected [hours]', fontsize=16, color='w')
ax.set_xlabel('Net length [m]', fontsize=16, color='w')
ax.grid(axis='y', linestyle='--', color='w', alpha=0.5)

# %% Load all metadata files

path_json = [r'L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO',
             r'Y:\Bioacoustique\APOCADO2',
             r'Z:\Bioacoustique\DATASETS\APOCADO3'
             ]

list_json = [file_path for path in path_json for file_path in glob.glob(os.path.join(path, "**/metadata.json"), recursive=True)]

data = []
for i in tqdm(range(len(list_json)), desc="Scanning metadata files"):
    r_file = open(list_json[i], 'r')
    data.append(json.load(r_file))
    r_file.close()
data = pd.DataFrame.from_dict(data)

df_detections_pamguard, df_detections_thalassa = [], []
for i in tqdm(range(len(data))):
    f1 = data['pamguard detection file'][i]
    f2 = data['thalassa detection file'][i]
    tz = data['datetime deployment'][i][-5:-2] + ':' + data['datetime deployment'][i][-2:]
    df_detections_file1, info_file = sorting_detections(file=f1, tz=tz, timebin_new=timebin1)
    df_detections_file2, info_file = sorting_detections(file=f2, tz=tz, timebin_new=timebin1)
    df_detections_pamguard.append(df_detections_file1)
    df_detections_thalassa.append(df_detections_file2)
data['df pamguard'] = df_detections_pamguard
data['df thalassa'] = df_detections_thalassa

data['platform'] = ['C' + str(c) + 'D' + str(d) for c, d in zip(data['campaign'], data['deployment'])]
data['datetime deployment'] = [pd.to_datetime(d) for d in data['datetime deployment']]
data['datetime recovery'] = [pd.to_datetime(d) for d in data['datetime recovery']]
data['season_y'] = [get_season(i) for i in data['datetime deployment']]
data['season'] = [i.split(' ')[0] for i in data['season_y']]
data['year'] = [int(s[-4:]) for s in data['season_y']]
data['duration'] = [pd.Timedelta(d) for d in data['duration']]

for d in detector:
    data[f'detection rate {d}'] = [(len(data[f'df {d}'][i]) / (pd.to_timedelta(data['duration'][i]).total_seconds() / 86400)) / (86400 / timebin1) for i in range(len(data))]

# %% filtering data

detector = ['pamguard', 'thalassa']

filtered_df_1 = data[(data['recorder number'] == 1)]  # data with 1 ST
# filtered_df_1 = data[(data['recorder number'] == 1) & (data['duration'] >= pd.Timedelta(hours=17)) & (data['duration'] <= pd.Timedelta(hours=30))].reset_index(drop=True)
filtered_data = {}

for d in detector:

    if d == 'thalassa':
        filtered_df_2 = data[(data['recorder number'] == 2) & (data['net length'] <= 200)]
        sub_df_2 = filtered_df_2.groupby('platform').first().reset_index()  # only the first ST
        filtered_df_3 = data[(data['recorder number'] == 2) & (data['net length'] > 200)]
        filtered_data[f'{d}'] = pd.concat([filtered_df_1, sub_df_2, filtered_df_3]).reset_index(drop=True)

    elif d == 'pamguard':
        filtered_df_2 = data[(data['recorder number'] == 2) & (data['net length'] <= 500)]
        sub_df_2 = filtered_df_2.groupby('platform').first().reset_index()  # only the first ST
        filtered_df_3 = data[(data['recorder number'] == 2) & (data['net length'] > 500)]
        filtered_data[f'{d}'] = pd.concat([filtered_df_1, sub_df_2, filtered_df_3]).reset_index(drop=True)

    mean_filter_df = filtered_data[f'{d}'][f'detection rate {d}'].mean()
    std_filter_df = filtered_data[f'{d}'][f'detection rate {d}'].std()
    max_filter_df = filtered_data[f'{d}'][f'detection rate {d}'].max()
    min_filter_df = filtered_data[f'{d}'][f'detection rate {d}'].min()
    print(f"\n{d} / all season: {mean_filter_df * 100:.0f}% +/- {100 * std_filter_df:.0f}%, max={100 * max_filter_df:.0f}%, min={100 * min_filter_df:.0f}%, N={len(filtered_data[f'{d}'])}")

    fig, ax = plt.subplots(figsize=(18, 9), facecolor='#36454F')
    hist, bins, _ = plt.hist(100 * filtered_data[f'{d}'][f'detection rate {d}'], bins=np.arange(0, 101, 5), density=False, edgecolor='black', alpha=0.5)
    # Fit a normal distribution to the data
    mu, std = norm.fit(100 * filtered_data[f'{d}'][f'detection rate {d}'])
    # Create range for the smoothed line
    x_range = np.linspace(0, 100, 1000)
    # Calculate the Gaussian curve
    pdf = norm.pdf(x_range, mu, std)
    # Plot the Gaussian curve
    plt.plot(x_range, (pdf / max(pdf)) * max(hist), color='limegreen', linewidth=3)
    ax.set_facecolor('#36454F')
    ax.spines['bottom'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', colors='w', labelsize=14)
    plt.xticks(np.arange(0, 100 + 1, 10))
    plt.xlabel('Positive minute detection rate / day', fontsize=16, color='w')
    plt.ylabel('Deployment number', fontsize=16, color='w')
    plt.title(f'Distribution of deployment detection rate\ndetector: {d}', color='w', fontsize=18)
    plt.show()

    for i in list(dict.fromkeys(filtered_data[f'{d}']['season_y'])):
        N = len(filtered_data[f'{d}'][filtered_data[f'{d}']['season_y'] == i])
        mean = filtered_data[f'{d}'][filtered_data[f'{d}']['season_y'] == i][f'detection rate {d}'].mean()
        std = filtered_data[f'{d}'][filtered_data[f'{d}']['season_y'] == i][f'detection rate {d}'].std()
        max_stat = filtered_data[f'{d}'][filtered_data[f'{d}']['season_y'] == i][f'detection rate {d}'].max()
        min_stat = filtered_data[f'{d}'][filtered_data[f'{d}']['season_y'] == i][f'detection rate {d}'].min()

        print(f"\t{d} / {i}: {mean * 100:.0f}% +/- {100 * std:.0f}%, max={100 * max_stat:.0f}%, min={100 * min_stat:.0f}%, N={N}")

        # fig, ax = plt.subplots(figsize=(18, 9), facecolor='#36454F')
        # # plt.hist(100 * filtered_data[f'{d}'][f'detection rate {d}'], bins=range(0, 101, 4), density=False, alpha=0.5, edgecolor='black')
        # hist, bins, _ = plt.hist(100 * filtered_data[f'{d}'][filtered_data[f'{d}']['season_y'] == i][f'detection rate {d}'], bins=np.arange(0, 101, 5), density=False, edgecolor='black', alpha=0.5)
        # # Fit a normal distribution to the data
        # mu, std = norm.fit(100 * filtered_data[f'{d}'][filtered_data[f'{d}']['season_y'] == i][f'detection rate {d}'])
        # # Create range for the smoothed line
        # x_range = np.linspace(0, 100, 1000)
        # # Calculate the Gaussian curve
        # pdf = norm.pdf(x_range, mu, std)
        # # Plot the Gaussian curve
        # plt.plot(x_range, (pdf / max(pdf)) * max(hist), color='limegreen', linewidth=3)
        # ax.set_facecolor('#36454F')
        # ax.spines['bottom'].set_color('w')
        # ax.spines['left'].set_color('w')
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # ax.tick_params(axis='both', colors='w', labelsize=14)
        # plt.xticks(np.arange(0, 100 + 1, 10))
        # plt.xlabel('Positive minute detection rate / day', fontsize=16, color='w')
        # plt.ylabel('Deployment number', fontsize=16, color='w')
        # plt.title(f'Distribution of deployment detection rate\nseason: {i} - detector: {d} - N={N}', color='w', fontsize=18)
        # plt.show()

# %% analyze metadata - distribution of the period with most detections

'''
Distribution of all the detection dataframes most populated time period,
first, each detection dataframe is divided into n periods of equal durations
Then the distribution of the period with the most detections for each df is plotted
'''
n_periods = 12
detector = ['pamguard', 'thalassa']
arg = ['season', 'net', 'all']

for d in detector:

    if d == 'pamguard':
        colorplot = 'orange'
    elif d == 'thalassa':
        colorplot = 'teal'

    data2 = filtered_data[f'{d}'][(filtered_data[f'{d}']['duration'] >= pd.Timedelta(hours=17)) & (filtered_data[f'{d}']['duration'] <= pd.Timedelta(hours=28))].reset_index(drop=True)
    print(f"{d}: {100 * len(data2) / len(filtered_data[f'{d}']):.0f}% of deployments selected ({len(data2)}/{len(filtered_data[f'{d}'])} deployments)")

    # # distribution of deployment durations
    # test = [t.total_seconds() / 3600 for t in data2['duration']]
    # fig, ax = plt.subplots(figsize=(18, 9), facecolor='#36454F')
    # ax.set_facecolor('#36454F')
    # ax.spines['bottom'].set_color('w')
    # ax.spines['left'].set_color('w')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.tick_params(axis='both', colors='w', labelsize=14)
    # # ax.hist(test, bins=range(0, int(1.05 * max(test)), 2))
    # ax.hist(test, bins=range(0, 151, 2))
    # plt.xlabel('Hours', fontsize=16, color='w')
    # plt.ylabel('Deployment number', fontsize=16, color='w')
    # plt.title('Distribution of deployement durations', color='w', fontsize=18)
    # ax.grid(axis='y', linestyle='--', color='w', alpha=0.5)
    # plt.show()

    for a in arg:
        if a != 'all':
            filter_list = list(dict.fromkeys(data2[f'{a}']))

            for f in filter_list:
                data3 = data2[data2[f'{a}'] == f]

                N = len(data3)
                rank_max, rank_min = [], []
                for i in range(len(data3)):
                    data_test = data3.iloc[i]
                    df_detections = data_test[f'df {d}']
                    data_histo = df_detections['start_datetime']
                    periods = pd.date_range(start=data_test['datetime deployment'], end=data_test['datetime recovery'], freq=str(int(data_test['duration'].total_seconds() / n_periods)) + 's')
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

                fig, ax = plt.subplots(figsize=(15, 6), facecolor='#36454F')
                ax.set_facecolor('#36454F')

                # bar plots
                ax.bar(values, count_max, align='center', alpha=0.8, edgecolor='w', color=colorplot)

                # spines
                ax.spines['right'].set_color('w')
                ax.spines['top'].set_color('w')
                ax.spines['bottom'].set_color('w')
                ax.spines['left'].set_color('w')

                # grid
                ax.yaxis.grid(color='gray', linestyle='--')

                # x-label
                ax.set_xlabel('Periods', color='w')

                # title
                ax.set_title(f'Distribution of the period with most detections\n{a}: {f} - detector: {d} - N={N}', color='w')

                # ticks
                ax.set_xticks(values)
                # ax.set_yticks(range(0, max(count_max) + 1, 3))
                ax.set_yticks(range(0, 20, 2))
                ax.tick_params(axis='both', colors='w')

                plt.ylim(0, max(count_max) + 2)
                # plt.ylim(0, 20)
                plt.show()

        else:
            data3 = data2

            N = len(data3)
            rank_max, rank_min = [], []
            for i in range(len(data3)):
                data_test = data3.iloc[i]
                df_detections = data_test[f'df {d}']
                data_histo = df_detections['start_datetime']
                periods = pd.date_range(start=data_test['datetime deployment'], end=data_test['datetime recovery'], freq=str(int(data_test['duration'].total_seconds() / n_periods)) + 's')
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

            fig, ax = plt.subplots(figsize=(15, 6), facecolor='#36454F')
            ax.set_facecolor('#36454F')

            # bar plots
            ax.bar(values, count_max, align='center', alpha=0.8, edgecolor='w', color=colorplot)

            # spines
            ax.spines['right'].set_color('w')
            ax.spines['top'].set_color('w')
            ax.spines['bottom'].set_color('w')
            ax.spines['left'].set_color('w')

            # grid
            ax.yaxis.grid(color='gray', linestyle='--')

            # x-label
            ax.set_xlabel('Periods', color='w')

            # title
            ax.set_title(f'Distribution of the period with most detections\ndetector: {d} - N={N}', color='w')

            # ticks
            ax.set_xticks(values)
            # ax.set_yticks(range(0, max(count_max) + 1, 3))
            ax.set_yticks(range(0, 20, 2))
            ax.tick_params(axis='both', colors='w')

            plt.ylim(0, max(count_max) + 1)
            # plt.ylim(0, 19)
            plt.show()


# %% analyze metadata - stat diel plot
'''
Distribution of all the detection over the different periods of the day
'''

detector = ['pamguard', 'thalassa']
arg = ['season', 'net', 'all']

for d in detector:
    data2 = filtered_data[f'{d}'][(filtered_data[f'{d}']['duration'] >= pd.Timedelta(hours=17)) & (filtered_data[f'{d}']['duration'] <= pd.Timedelta(hours=28))].reset_index(drop=True)

    if d == 'pamguard':
        inside_color = '#D9C5A0'
        contour_color = '#D97F11'
    elif d == 'thalassa':
        inside_color = '#65A6A6'
        contour_color = '#0A7373'

    for a in arg:
        if a != 'all':
            filter_list = list(dict.fromkeys(data2[f'{a}']))

            for f in filter_list:
                data3 = data2[data2[f'{a}'] == f]
                N = len(data3)

                df_detections = pd.DataFrame()
                for df in data3[f'df {d}']:
                    df_detections = pd.concat([df_detections, df], ignore_index=True)

                begin_deploy = min(data3['datetime deployment'])
                end_deploy = max(data3['datetime deployment'])
                lat = data3['latitude'].mean()
                u_lat = data3['latitude'].std()
                lon = data3['longitude'].mean()
                u_lon = data3['longitude'].std()

                lr, BoxNames = stats_diel_pattern(df_detections=df_detections, begin_date=begin_deploy, end_date=end_deploy, lat=lat, lon=lon)
                LR = lr[(lr[BoxNames[0]] != 0) & (lr[BoxNames[1]] != 0) & (lr[BoxNames[2]] != 0) & (lr[BoxNames[3]] != 0)]

                fig, ax = plt.subplots(figsize=(10, 6), facecolor='#36454F')
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
                           boxprops=dict(facecolor=inside_color, color=contour_color, linewidth=2),
                           capprops=dict(color=contour_color, linewidth=2),
                           medianprops=dict(color=contour_color, linewidth=2),
                           flierprops=dict(markeredgecolor=contour_color, linewidth=2),
                           whiskerprops=dict(color=contour_color, linewidth=2))
                plt.xticks([1, 2, 3, 4], BoxNames)
                plt.title(f'Detection distribution\n{a}: {f} - detector: {d} - N={N}', color='w', fontsize=14)
        else:
            data3 = data2
            N = len(data3)

            df_detections = pd.DataFrame()
            for df in data3[f'df {d}']:
                df_detections = pd.concat([df_detections, df], ignore_index=True)

            begin_deploy = min(data3['datetime deployment'])
            end_deploy = max(data3['datetime deployment'])
            lat = data3['latitude'].mean()
            u_lat = data3['latitude'].std()
            lon = data3['longitude'].mean()
            u_lon = data3['longitude'].std()

            lr, BoxNames = stats_diel_pattern(df_detections=df_detections, begin_date=begin_deploy, end_date=end_deploy, lat=lat, lon=lon)
            LR = lr[(lr[BoxNames[0]] != 0) & (lr[BoxNames[1]] != 0) & (lr[BoxNames[2]] != 0) & (lr[BoxNames[3]] != 0)]

            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#36454F')
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
                       boxprops=dict(facecolor=inside_color, color=contour_color, linewidth=2),
                       capprops=dict(color=contour_color, linewidth=2),
                       medianprops=dict(color=contour_color, linewidth=2),
                       flierprops=dict(markeredgecolor=contour_color, linewidth=2),
                       whiskerprops=dict(color=contour_color, linewidth=2))
            plt.xticks([1, 2, 3, 4], BoxNames)
            plt.title(f'Detection distribution per period of the day\ndetector: {d} - N={N}', color='w', fontsize=14)


# %% analyze metadata - cumulated histogram of detections for a single detection file

i = 50
detector = ['pamguard', 'thalassa']

for d in detector:
    data2 = filtered_data[f'{d}']
    df_detections = data2[f'df {d}'][i]
    threshold = 0.75
    ID_test = data2['platform'][i] + ' ST' + data2['recorder'][i]

    data_histo = df_detections['start_datetime']
    tb = timebin1
    deploy_dt = [data2['datetime deployment'][i], data2['datetime recovery'][i]]  # beginning and end of deployment

    res_min = timebin2
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
    ax.text(bins[bin_index + 10], 50, '{0} min - {1:.0f}%'.format(min_elapsed, 100 * perc_elapsed), color='r', ha='left', fontsize=16, fontweight='bold')
    ax.text(bins[0], threshold * total_detections * 1.02 + 2, '{0} detections'.format(threshold_detect), color='r', ha='left', fontsize=16, fontweight='bold')
    tz_data = deploy_dt[0].tz
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz_data))
    fig.suptitle(f"Cumulated detections - {ID_test}\n{deploy_dt[0].strftime('%d/%m/%y')} - {deploy_dt[1].strftime('%d/%m/%y')} - detector: {d}", fontsize=24, y=0.98)
    ax.grid(color='k', linestyle='-', linewidth=0.2, axis='both')

# %% analyze metadata - histogram of the detections for a single detection file
'''
The histogram is divided into n periods of equal durations
the periods number with the most detections are printed
'''

i = 50
n_periods = 12
detector = ['pamguard', 'thalassa']

for d in detector:
    data2 = filtered_data[f'{d}']
    data_test = data2.iloc[i]
    name = data_test['platform'] + ' ST' + data_test['recorder']

    df_detections = data_test[f'df {d}']

    data_histo = df_detections['start_datetime']
    periods = pd.date_range(start=data_test['datetime deployment'], end=data_test['datetime recovery'], freq=str(int(data_test['duration'].total_seconds() / n_periods)) + 's')

    fig, ax = plt.subplots(figsize=(20, 9), facecolor='#36454F')
    ax.set_facecolor('#36454F')
    detect_periods, _, bars = ax.hist(data_histo, periods, edgecolor='None')

    # spines
    ax.spines['right'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['bottom'].set_color('w')
    ax.spines['left'].set_color('w')

    ax.tick_params(axis='both', colors='w', size=20)

    # grids
    ax.yaxis.grid(color='gray', linestyle='--')

    # x-labels
    ax.set_xlabel('Periods', color='w', fontsize=20)

    [ax.axvline(x=period, color='lime') for period in periods]
    ax.grid(color='k', linestyle='-', linewidth=0.1, axis='both')

    max_value = np.max(detect_periods)
    rank_max = np.argmax(detect_periods) + 1
    # rank_min = np.argmin(detect_periods) + 1
    bars[rank_max - 1].set_color('orange')
    # bars[rank_min - 1].set_color('#CD3333')

    ax.set_ylabel("Detections", fontsize=20, color='w')

    perc_max = detect_periods[rank_max - 1] / len(data_test[f'df {d}'])
    # perc_min = detect_periods[rank_min - 1] / len(data_test[f'df {d}'])
    fig.suptitle(f'{name}\ndetector: {d}', fontsize=24, y=0.98, color='w')

    print(f"\n{d}: most detections at period {rank_max}/{n_periods}, {detect_periods[rank_max-1]:.0f}/{len(data_test[f'df {d}'])} detections ({100*perc_max:.0f}%)", end='')

# %% export csv for QGIS

deploy2 = deploy[(deploy['campaign'] <= 7)].sort_values(by=['campaign', 'deployment', 'ID recorder']).reset_index(drop=True)

detector = 'thalassa'

filtered_df_1 = deploy2[(deploy2['recorder number'] == 1)]  # data with 1 ST

if detector == 'thalassa':
    filtered_df_2 = deploy2[(deploy2['recorder number'] == 2) & (deploy2['net length'] <= 200)]
    filtered_df_3 = deploy2[(deploy2['recorder number'] == 2) & (deploy2['net length'] > 200)]
elif detector == 'pamguard':
    filtered_df_2 = deploy2[(deploy2['recorder number'] == 2) & (deploy2['net length'] <= 500)]
    filtered_df_3 = deploy2[(deploy2['recorder number'] == 2) & (deploy2['net length'] > 500)]

sub_df_2 = filtered_df_2.groupby('ID platform').first().reset_index()  # only the first ST
deploy3 = pd.concat([filtered_df_1, sub_df_2, filtered_df_3]).sort_values(by=['campaign', 'deployment', 'ID recorder']).reset_index(drop=True)

df_deploy = []
for c, d, r in zip(deploy3['campaign'], deploy3['deployment'], deploy3['ID recorder']):
    df_deploy.append(data[(data['campaign'] == c) & (data['deployment'] == d) & (data['recorder'] == r)][f'df {detector}'].iloc[-1])

deploy3[f'detection rate {detector}'] = [(len(df_deploy[i]) / (pd.to_timedelta(deploy3['duration'][i]).total_seconds() / 86400) / 1440) * 100 for i in range(len(deploy3))]

deploy_out = deploy3.drop(columns=['lat D', 'lat DM', 'lat DD', 'long D', 'long DM', 'long DD', 'weather', 'weather.1', 'comment', 'species'])
date_today = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
# deploy_out.to_csv(os.path.join(r'C:\Users\dupontma2\Desktop\code_local\Delmoges V2', f'APOCADO - Suivi déploiements {detector}_{date_today}.csv'), index=False, encoding='latin1')