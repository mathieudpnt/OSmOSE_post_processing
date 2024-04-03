import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import os
import glob
import seaborn as sns
from scipy import stats
import json
from tqdm import tqdm
from scipy.stats import linregress

os.chdir(r'U:/Documents_U/Git/post_processing_detections')
from utilities.def_func import extract_datetime, sorting_detections, t_rounder

# %%
detector = ['pamguard', 'thalassa']
arg = ['season', 'net']
timebin1 = 10 #  en seconde
timebin2 = 1 #  en minute

# %% Load data

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

for d in detector:
    data[f'detection rate {d}'] = [len(data[f'df {d}'][i]) * data[f'{d} timebin'][i] / pd.to_timedelta(data['duration'][i]).total_seconds() for i in range(len(data))]

# %% Plot regressions

data1 = data[(data['recorder number'] == 2)].reset_index(drop=True)

for d in detector:

    if d == 'thalassa':
        net_category = []
        for i in range(len(data1)):
            if data1['net length'].iloc[i] <= 200:
                net_category.append('1: [0 - 200] m')
            elif (data1['net length'].iloc[i] > 200) & (data1['net length'].iloc[i] <= 300):
                net_category.append('2: ]200 - 300] m')
            elif (data1['net length'].iloc[i] > 300) & (data1['net length'].iloc[i] <= 400):
                net_category.append('3: ]300 - 400] m')
            elif (data1['net length'].iloc[i] > 400):
                net_category.append('4: +400m')
            else: raise ValueError(f'{i}')
        data1['net category'] = net_category

    elif d == 'pamguard':
        net_category = []
        for i in range(len(data1)):
            if data1['net length'].iloc[i] <= 300:
                net_category.append('1: [0 - 300] m')
            elif (data1['net length'].iloc[i] > 300) & (data1['net length'].iloc[i] <= 500):
                net_category.append('2: ]300 - 500] m')
            elif data1['net length'].iloc[i] > 500:
                net_category.append('3: +500m')
            else: raise ValueError(f'{i}')
        data1['net category'] = net_category

    data_corr = pd.DataFrame()

    list_length = sorted(list(set(data1['net category'])))
    n_samples = pd.Series.sort_index(0.5 * data1['net category'].value_counts()).astype(int)

    for length in list_length:
        sub_data = data1[data1['net category'] == length].sort_values(by=['deployment ID'])
        list_sub_ID = sorted(list(set([i.split(' ')[0] for i in sub_data['deployment ID']])))
        for ID in list_sub_ID:
            sub_data2 = sub_data[sub_data['deployment ID'].str.contains(ID)].reset_index(drop=False)
            if len(sub_data2) == 2:
                df1_detections = sub_data2[f'df {d}'][0]
                df2_detections = sub_data2[f'df {d}'][1]

                tz_data = df1_detections['start_datetime'][0].tz

                time_bin = list(set(sub_data2[f'{d} timebin']))[0]

                label_legend = [sub_data2['deployment ID'][0], sub_data2['deployment ID'][1]]

                res_min = timebin2

                delta = dt.timedelta(seconds=60 * res_min)
                start_vec = t_rounder(pd.Timestamp(min(sub_data2['datetime deployment'])), res=60)
                end_vec = t_rounder(pd.Timestamp(max(sub_data2['datetime recovery'])), res=60)

                time_vector = [start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)]
                duration_h = int((time_vector[-1] - time_vector[0]).total_seconds() // 3600)

                n_annot_max = (res_min * 60) / time_bin  # max nb of annoted time_bin max per res_min slice

                hist1, _ = np.histogram([dt.timestamp() for dt in df1_detections['start_datetime']], bins=[dt.timestamp() for dt in time_vector])
                hist2, _ = np.histogram([dt.timestamp() for dt in df2_detections['start_datetime']], bins=[dt.timestamp() for dt in time_vector])
                counts1 = 100 * (hist1 / n_annot_max)
                counts2 = 100 * (hist2 / n_annot_max)

                df_corr = pd.DataFrame({'deployment': label_legend[0].split(' ')[0] + ' ' + label_legend[0].split(' ')[-1] + '/' + label_legend[1].split(' ')[-1],
                                        'ST1': list(counts1),
                                        'ST2': list(counts2),
                                        'net_len': length})

                if len(data_corr) == 0: data_corr = df_corr
                else: data_corr = pd.concat([data_corr, df_corr])

    print('Done')

    # ST correlation
    sns.set(font_scale=1.8)

    sns.set_style({
        'figure.facecolor': '#36454F',
        'axes.facecolor': '#36454F',
        'axes.edgecolor': 'w',
        'grid.color': 'grey',
        'xtick.color': 'w',
        'ytick.color': 'w',
        'axes.labelcolor': 'w',
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.right': True,
        'axes.spines.top': True,
    })

    g = sns.lmplot(
        data=data_corr,
        x='ST1',
        y='ST2',
        hue='net_len',
        scatter_kws={'s': 1},
        line_kws={'lw': 1},
        ci=95,
        legend=False,
        height=12,
        markers='',
        palette='tab10',
    )

    color_leg = [g.ax.lines[i].get_color() for i in range(len(g.ax.lines))]
    for i, n in enumerate(sorted(list(set(data_corr['net_len'])))):
        r, p = stats.pearsonr(data_corr[data_corr['net_len'] == n]['ST1'], data_corr[data_corr['net_len'] == n]['ST2'])

        slope, intercept, r_value, p_value, std_err = linregress(data_corr[data_corr['net_len'] == n]['ST1'], data_corr[data_corr['net_len'] == n]['ST2'])

        ax = plt.gca()
        ax.text(
            0.6,
            0.25 - (i * 0.035),
            f'{n} - R²={r*r:.2f} - N={n_samples[i]}',
            transform=ax.transAxes,
            color=color_leg[i],
            size=16,
            bbox=dict(facecolor='white', alpha=0.8)
        )

    g.ax.set(
        xlabel='ST1 - 1min positive detection rate',
        ylabel='ST2 - 1min positive detection rate'
    )

    plt.xlim(0, 101)
    plt.ylim(0, 101)
    plt.title(f"Appaired SoundTraps correlation\n detector: {d} - {timebin1}s positive detection per {res_min}min", y=1, size=20, color='w')

    plt.show()
