import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
import gpxpy
import time
from pathlib import Path

import def_func
import utils
from utils.def_func import sort_detections
from utils.premiers_resultats_utils import load_parameters_from_yaml
from utils.glider_utils import load_glider_nav
from premiers_resultats_utils import plot_detection_timeline

from utils.trajectoryFda import TrajectoryFda

def get_track_data(gpx_path):
    gpx_file = open(gpx_path, "r")
    gpx = gpxpy.parse(gpx_file)

    # Compute lists of lat lon depth and time
    latitude = []
    longitude = []
    time_dt = []
    depth = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                latitude.append(point.latitude)
                longitude.append(point.longitude)
                time_dt.append(point.time)
                depth.append(point.elevation)

    time_unix = [time.mktime(t.timetuple()) for t in time_dt]
    track_data = np.column_stack(
        (np.array(time_unix), np.array(longitude), np.array(latitude), depth)
    )

    return track_data


def compute_loc_from_time(track_data, time_detections):
    dict_mmsi = {}
    key_mmsi = dict_mmsi.keys()

    # ix : index de la position
    # row : ligne de la position (time, lat, lon, depth)
    for ix, row in track_data.iterrows():

        # on commence par chercher si le navire existe déjà dans le flux
        if 0 not in key_mmsi:
            dict_mmsi[0] = TrajectoryFda(0, 0.001, 3)

        dict_mmsi[0].setNewData(row.iloc[0], row.iloc[2], row.iloc[1])

    ts_min = min(time_detections)
    ts_max = max(time_detections)

    res = []
    for ts in time_detections:
        if ts_min <= ts <= ts_max:
            lat, lon = dict_mmsi[0].getPosition(ts)

            if len(lon) > 0:
                res.append([ts, lon[0][0], lat[0][0]])

    return res

# %% Load detections and navigation data

df_detections, time_bin, annotators, labels, fmax, datetime_begin, datetime_end, _ = (
    load_parameters_from_yaml()
)
print(
    f"\ntime_bin: {time_bin}\nfmax: {fmax}\nannotators: {annotators}\nlabels: {labels}"
)

directory = Path(r'L:\acoustock\Bioacoustique\DATASETS\GLIDER\GLIDER SEA034\MISSION_46_DELGOST\APRES_MISSION\NAV')
glider_data = load_glider_nav(directory)

# %% 'Timeline' figure'
plot_detection_timeline(df_detections)

# %% Compute acoustic diversity

# track_data: glider positions at every timestamp
glider_data['Timestamp_unix'] = [ts.timestamp() for ts in glider_data['Timestamp']]
track_data = glider_data[['Timestamp_unix', 'Lat', 'Lon', 'Depth']]

# time_detections: datetime of every detections
time_detections = [ts.timestamp() for ts in df_detections['start_datetime'].drop_duplicates().to_list()]

# Compute localisation of each detection
track_det = compute_loc_from_time(track_data, time_detections)

# Compute localisation of each detection
task_status_directory = Path(r'L:\acoustock\Bioacoustique\DATASETS\GLIDER\GLIDER SEA034\MISSION_46_DELGOST\ANALYSES\ANNOTATION\HF')
task_status_files = list(task_status_directory.glob('*/*task_status.csv'))
list_det = pd.DataFrame(columns=['Timestamp', 'Latitude', 'Longitude', 'Acoustic Diversity'])

for deployment in tqdm(task_status_files):

    task_status_filenames = pd.read_csv(deployment)['filename']
    time_vector = [def_func.extract_datetime(fn).tz_localize('UTC').timestamp() for fn in task_status_filenames]
    track_time_vector = compute_loc_from_time(track_data, time_vector)

    # delete duplicate detection in case of several users
    det = df_detections.drop_duplicates(subset=["annotation", "start_datetime"])

    # Delete unkown detections
    det.drop(det[det["annotation"] == "UnidentifiedCalls"].index, inplace=True)

    # unix time of detections
    time_det_unix = [ts.timestamp() for ts in det["start_datetime"]]

    acoustic_diversity = np.zeros(len(time_vector))
    for i, ts in enumerate(time_vector[:-1]):
        for ts_det in time_det_unix:
            if ts <= ts_det <= ts + 1:
                acoustic_diversity[i] += 1

            new_row = pd.DataFrame({
                'Timestamp': [str(pd.Timestamp(ts, unit='s').tz_localize('UTC'))],
                'Latitude': [track_det[i][1]],
                'Longitude': [track_det[i][2]],
                'Acoustic Diversity': [acoustic_diversity[i]]
            })

           list_det = new_row if list_det.empty else pd.concat([list_det, new_row], ignore_index=True)

# %%----------- Plot detection according to the depth of the glider ------------
# User inputs (APLOSE csv)

files_list = get_csv_file(1)

arguments_list = [
    {
        "file": files_list[0],
        #'timebin_new': 60,
        "tz": pytz.FixedOffset(120),
        #'fmin_filter': 10000
    },
]

df_detections, info = pd.DataFrame(), pd.DataFrame()
for args in arguments_list:
    df_detections_file, info_file = sort_detections(**args)
    df_detections = pd.concat([df_detections, df_detections_file], ignore_index=True)
    info = pd.concat([info, info_file], ignore_index=True)


time_bin = list(set(info["max_time"].explode()))
fmax = list(set(info["max_freq"].explode()))
annotators = list(set(info["annotators"].explode()))
labels = list(set(info["labels"].explode()))
tz_data = list(set(info["tz_data"].explode()))
if len(tz_data) == 1:
    [tz_data] = tz_data
else:
    raise Exception("More than one timezone in the detections")

# %% selection of the user
det = df_detections.drop_duplicates(subset=["annotation", "start_datetime"])
list_labels = list(det["annotation"].unique())
# selection of the label
# label_ref = easygui.buttonbox('Select a label', 'Single plot', list_labels) if len(list_labels) > 1 else list_labels[0]


# %% Import gpx
# Download track data
gpx_paths = get_gpx(1)

for i, p in enumerate(gpx_paths):
    td = get_track_data(p)
    if i == 0:
        track_data = td
    else:
        track_data = np.concatenate((track_data, td), axis=0)
# Sort track data
track_data = track_data[np.argsort(track_data[:, 0])]

# Array with unix time of files
time_unix = np.array([val[0] for val in track_data])

# Plot Figure
mpl_timestampG = mdates.epoch2num(time_unix)
# Set colors to species
list_color = {
    "label": [
        "Odontocete whistles",
        "Sperm whale clics",
        "Odontocete clics",
        "UnidentifiedCalls",
        "Odontocete buzz",
        "Blackfish whistles",
        "Fin whale 40 Hz",
        "Fin whale 20 Hz",
    ],
    "color": [
        "#7fd779",
        "#e8718d",
        "#e77148",
        "#1c4a64",
        "#b7484b",
        "#72450a",
        "black",
        "gray",
    ],
}
df_color = pd.DataFrame(data=list_color)

for i, label in enumerate(list_labels):
    det = df_detections[(df_detections["annotation"] == label)]
    time_det = det["start_datetime"]
    time_det_unix = [time.mktime(t.timetuple()) for t in time_det]
    depthD_np = np.array(det["depth"])
    # Convert detections timestamps to date format
    mpl_timestampD = mdates.epoch2num(time_det_unix)

    fig, ax = plt.subplots(figsize=(20, 8))
    plt.plot(mpl_timestampG, depth, zorder=1, color="darkgray", linewidth=0.5)

    locator = mdates.HourLocator(interval=4)
    formatter = mdates.DateFormatter("%H:%M")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.grid(color="k", linestyle="-", linewidth=0.2)

    c = df_color.loc[df_color["label"] == label, "color"].values[0]

    plt.scatter(mpl_timestampD, depthD_np, s=10, zorder=2, color=c)

    plt.xlim(mpl_timestampG[0], mpl_timestampG[-1])

    ax.set_title(label=label, fontsize=30)
    ax.set_ylabel("Profondeur (m)", fontsize=30)
    ax.set_xlabel("Date", fontsize=30)
    ax.tick_params(labelsize=20)
    # savename = 'C:/Users/torterma/Documents/Projets_GLIDER/TAAF/Delgost/depth_det_D2_' +label + '.png'
    # plt.savefig(savename)


# %% Stack all depth with detections to try and see a pattern ?

fig, ax = plt.subplots(figsize=(20, 8))

plt.scatter([0] * len(depthD_np), depthD_np, s=10, zorder=2, color=c)

plt.grid(color="k", linestyle="-", linewidth=0.2)
plt.ylim(-750, 0)

# %% Figure boite à moustache des profondeurs de détection


files_list = get_csv_file(6)

arguments_list = [
    {
        "file": files_list[0],
        #'timebin_new': 60,
        "tz": pytz.FixedOffset(120),
        #'fmin_filter': 10000
    },
    {
        "file": files_list[1],
        #'timebin_new': 60,
        "tz": pytz.FixedOffset(120),
        #'fmin_filter': 10000
    },
    {
        "file": files_list[2],
        #'timebin_new': 60,
        "tz": pytz.FixedOffset(120),
        #'fmin_filter': 10000
    },
    {
        "file": files_list[3],
        #'timebin_new': 60,
        "tz": pytz.FixedOffset(120),
        #'fmin_filter': 10000
    },
    {
        "file": files_list[4],
        #'timebin_new': 60,
        "tz": pytz.FixedOffset(120),
        #'fmin_filter': 10000
    },
    {
        "file": files_list[5],
        #'timebin_new': 60,
        "tz": pytz.FixedOffset(120),
        #'fmin_filter': 10000
    },
]

df_detections, info = pd.DataFrame(), pd.DataFrame()
for args in arguments_list:
    df_detections_file, info_file = sort_detections(**args)
    df_detections = pd.concat([df_detections, df_detections_file], ignore_index=True)
    info = pd.concat([info, info_file], ignore_index=True)


time_bin = list(set(info["max_time"].explode()))
fmax = list(set(info["max_freq"].explode()))
annotators = list(set(info["annotators"].explode()))
labels = list(set(info["labels"].explode()))
tz_data = list(set(info["tz_data"].explode()))
if len(tz_data) == 1:
    [tz_data] = tz_data
else:
    raise Exception("More than one timezone in the detections")


# drop duplicate annotations
det = df_detections.drop_duplicates(subset=["annotation", "start_datetime"])
det.drop(det[det["annotation"] == "UnidentifiedCalls"].index, inplace=True)
d = det.groupby("annotation")

list_labels = list(det["annotation"].unique())
counts = [len(v) for k, v in d]
total = float(sum(counts))
cases = len(counts)
widths = [c / total for c in counts]
# labels = [labels[0], labels[2],labels[1],labels[3],labels[4],labels[5],labels[6], labels[7]]
dic = {key: [] for key in list_labels}
# %%
for i, label in enumerate(list_labels):
    dic[label] = df_detections.loc[df_detections["annotation"] == label, "depth"].values


l, data = [*zip(*dic.items())]  # 'transpose' items to parallel key, value lists

fig, ax = plt.subplots(figsize=(20, 8))
positions = np.arange(0.5, len(list_labels) + 0.5)
plt.boxplot(
    data,
    # positions=positions,
    widths=0.5,
    # labels=hour_list[:-1],
    patch_artist=True,
    notch=False,
    # showfliers=False,
    boxprops=dict(facecolor="#769dc8", color="#437ab4", linewidth=2),
    capprops=dict(color="#437ab4", linewidth=2),
    medianprops=dict(color="#437ab4", linewidth=2),
    flierprops=dict(markeredgecolor="#437ab4", linewidth=2),
    whiskerprops=dict(color="#437ab4", linewidth=2),
)
plt.xticks(range(1, len(l) + 1), l)
ax.set_ylabel("Profondeur (m)", fontsize=30)
ax.set_xlabel("Label", fontsize=30)
# ax.set_xticklabels(['%s%d'%(k), for k in dic], fontsize=12)
ax.set_xticklabels(["%d" % (len(k)) for k in data], fontsize=12)
ax.tick_params(labelsize=12)
plt.grid(color="k", linestyle="-", linewidth=0.2)
plt.show()


# %%

fig, ax = plt.subplots(figsize=(10, 6), facecolor="#36454F")
ax.set_facecolor("#36454F")
positions = np.arange(0.5, len(list_labels) + 0.5)
plt.boxplot(
    data,
    # positions=positions,
    widths=0.5,
    # labels=list_labels,
    patch_artist=True,
    notch=False,
    # showfliers=False,
    boxprops=dict(facecolor="#769dc8", color="#437ab4", linewidth=2),
    capprops=dict(color="#437ab4", linewidth=2),
    medianprops=dict(color="#437ab4", linewidth=2),
    flierprops=dict(markeredgecolor="#437ab4", linewidth=2),
    whiskerprops=dict(color="#437ab4", linewidth=2),
)


ax.tick_params(axis="both", colors="w")
ax.spines["bottom"].set_color("w")
ax.spines["left"].set_color("w")
ax.spines["right"].set_color("w")
ax.spines["top"].set_color("w")

ax.set_ylabel("Profondeur (m)", fontsize=25, color="white")
ax.set_xlabel("Label", fontsize=25, color="white")
plt.xticks(range(1, len(l) + 1), l)
# ax.set_xticklabels(['%s\n$n$=%d'%(k, len(v)) for k, v in d], fontsize=12)
plt.grid(color="w", linestyle="-", linewidth=0.2)


# %%


ax = det.boxplot(column="depth", by="annotation", showfliers=True)
ax.get_figure().suptitle("")
ax.set_title("")


ax.set_xticklabels(["%s\n$n$=%d" % (k, len(v)) for k, v in d])
ax.set_ylabel("Profondeur (m)", fontsize=30)
ax.set_xlabel("Label", fontsize=30)
