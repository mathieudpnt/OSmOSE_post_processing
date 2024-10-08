import datetime as dt
import pandas as pd
import numpy as np
import easygui
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from collections import Counter
import seaborn as sns
from scipy import stats
import sys
import pytz
from utilities.trajectoryFda import TrajectoryFda
import gpxpy
import time

from utils.def_func import (
    get_csv_file,
    sort_detections,
    t_rounder,
    get_timestamps,
    input_date,
    suntime_hour,
)

# %% User inputs

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
annot_ref = (
    easygui.buttonbox("Select an annotator", "Single plot", annotators)
    if len(annotators) > 1
    else annotators[0]
)
# list of the labels corresponding to the sleected user
list_labels = info[info["annotators"].apply(lambda x: annot_ref in x)][
    "labels"
].reset_index(drop=True)[0]
# selection of the label
# label_ref = easygui.buttonbox('Select a label', 'Single plot', list_labels) if len(list_labels) > 1 else list_labels[0]


# %% Import gpx

gpx_filename = (
    "C:/Users/torterma/Documents/Projets_GLIDER/TAAF/Delgost/deployment_2.gpx"
)
# gpx_filename = 'L:/acoustock/Bioacoustique/DATASETS/GLIDER/GLIDER SEA034/MISSION_46_DELGOST/ANALYSES/carto/output_glider3.gpx'
gpx_file = open(gpx_filename, "r")

gpx = gpxpy.parse(gpx_file)

time_dt = []
depth = []
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            time_dt.append(point.time)
            depth.append(point.elevation)


time_unix = [time.mktime(t.timetuple()) for t in time_dt]
track_data = np.column_stack((np.array(time_unix), depth))


# %% Number of detections for the mission
# nb_det_mission = len(timestampD2)
# Convert depth to numpy array

mpl_timestampG = mdates.epoch2num(time_unix)

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
    det = df_detections[
        (df_detections["annotator"] == annot_ref)
        & (df_detections["annotation"] == label)
    ]
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
    savename = (
        "C:/Users/torterma/Documents/Projets_GLIDER/TAAF/Delgost/depth_det_D2_"
        + label
        + ".png"
    )
    plt.savefig(savename)


# %% Figure sur laquel on met le même abscisse à toutes les profondeurs juste pour voir leur répartition


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
d = det.groupby("annotation")

counts = [len(v) for k, v in d]
total = float(sum(counts))
cases = len(counts)
widths = [c / total for c in counts]
labels = [
    labels[0],
    labels[2],
    labels[1],
    labels[3],
    labels[4],
    labels[5],
    labels[6],
    labels[7],
]
dic = {key: [] for key in labels}
# %%
for i, label in enumerate(labels):
    dic[label] = df_detections.loc[df_detections["annotation"] == label, "depth"].values


l, data = [*zip(*dic.items())]  # 'transpose' items to parallel key, value lists

fig, ax = plt.subplots(figsize=(20, 8))
positions = np.arange(0.5, len(labels) + 0.5)
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
# ax.set_xticklabels(['%s\n$n$=%d'%()])
ax.tick_params(labelsize=12)
plt.grid(color="k", linestyle="-", linewidth=0.2)
plt.show()


# %%
# data2=[data[0],data[2], data[1], data[3], data[4], data[5], data[6], data[7]]
# labels2 = [labels[0], labels[2],labels[1],labels[3],labels[4],labels[5],labels[6], labels[7]]
# counts2 = [counts[0], counts[2], counts[1], counts[3], counts[4], counts[5], counts[6], counts[7],]
fig, ax = plt.subplots(figsize=(10, 6), facecolor="#36454F")
ax.set_facecolor("#36454F")
positions = np.arange(0.5, len(labels) + 0.5)
plt.boxplot(
    data,
    positions=positions,
    widths=0.5,
    labels=labels,
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
ax.set_xticklabels(["%s\n$n$=%d" % (k, len(v)) for k, v in d], fontsize=12)

plt.grid(color="w", linestyle="-", linewidth=0.2)


# %%


ax = det.boxplot(column="depth", by="annotation", showfliers=True)
ax.get_figure().suptitle("")
ax.set_title("")


ax.set_xticklabels(["%s\n$n$=%d" % (k, len(v)) for k, v in d])
ax.set_ylabel("Profondeur (m)", fontsize=30)
ax.set_xlabel("Label", fontsize=30)
