import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
from utils.trajectoryFda import TrajectoryFda
import gpxpy
from tkinter import filedialog
import time
from tkinter import Tk
from pathlib import Path

from utils.def_func import get_csv_file, sort_detections


def get_gpx(num_files):
    root = Tk()
    root.withdraw()

    file_paths = []
    for _ in range(num_files):
        file_path = filedialog.askopenfilename(
            title=f"Select gpx ({len(file_paths) + 1}/{num_files})",
            filetypes=[("GPX files", "*.gpx")],
            parent=None,
        )
        if not file_path:
            break  # User cancelled or closed the file dialog
        file_paths.append(file_path)
    return file_paths


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


def compute_loc_from_time(track_data, time_unix):
    dict_mmsi = {}
    key_mmsi = dict_mmsi.keys()
    # ix : index de la position
    # row : ligne de la position (time, lat, lon, depth)
    for ix, row in enumerate(track_data):
        # on commence par chercher si le navire existe déjà dans le flux
        if 0 not in key_mmsi:
            dict_mmsi[0] = TrajectoryFda(0, 0.001, 3)

        dict_mmsi[0].setNewData(row[0], row[2], row[1])

    ts_min = time_unix[0]
    ts_max = time_unix[-1]
    res = []
    for ts in time_unix:
        if ts_min <= ts <= ts_max:
            lat, lon = dict_mmsi[0].getPosition(ts)

            if len(lon) > 0:
                res.append([ts, lon[0][0], lat[0][0]])

    return res


def write_gps_in_csv(files_list, df_detections, gpx_path):

    track_data = get_track_data(gpx_path)
    time_unix = np.array([val[0] for val in track_data])
    depth_gps = np.array([val[3] for val in track_data])

    ts_min = time_unix[0]
    ts_max = time_unix[-1]

    time_det = df_detections["start_datetime"]
    time_det_unix = [time.mktime(t.timetuple()) for t in time_det]

    res = compute_loc_from_time(track_data, time_det_unix)

    # Find depth of glider for each detections
    depthD = []
    for j, detT in enumerate(time_det_unix):
        if ts_min < detT < ts_max:
            a = np.abs(np.array(time_unix) - detT)
            idx = np.where(a == a.min())
            depthD.append(depth_gps[np.array(idx[0]).min()])

        else:
            continue

    df_detections["longitude"] = [res[i][1] for i in list(range(0, len(res)))]
    df_detections["latitude"] = [res[i][2] for i in list(range(0, len(res)))]
    df_detections["depth"] = depthD

    # Save the csv file in the same folder
    fn = (files_list[0].split("/")[-1]).split(".")[0]
    df_detections.to_csv(fn + "_position.csv", index=False)


# %% First step : write a (new?) csv result file containng the position and depth of each detection

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

write_gps_in_csv(files_list, df_detections, gpx_paths[0])


# %%┴Figure 'planning'
# Select all csv files with detections/annotations that will appear in the following Figures
files_list = get_csv_file(2)

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


det = df_detections.drop_duplicates(subset=["annotation", "start_datetime"])

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

list_labels = list(det["annotation"].unique())

df_color = pd.DataFrame(data=list_color)
fig, ax = plt.subplots(figsize=(20, 8))
for i, label in enumerate(list_labels):
    det_label = df_detections[(df_detections["annotation"] == label)]
    time_det = det_label["start_datetime"]
    time_det_unix = [time.mktime(t.timetuple()) for t in time_det]
    mpl_time_det = mdates.epoch2num(time_det_unix)

    l_data = len(mpl_time_det)
    x = np.ones((l_data, 1), int) * i
    c = df_color.loc[df_color["label"] == label, "color"].values[0]

    plt.scatter(mpl_time_det, x, s=38, color=c)
    print(i)
    print(label)

    # locator = mdates.HourLocator(interval=24)
    # formatter = mdates.DateFormatter('%d/%m - %H:%M')
    locator = mdates.DayLocator(interval=1)
    formatter = mdates.DateFormatter("%d-%m")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.grid(color="k", linestyle="-", linewidth=0.2)

    # ticks labels
    # plt.yticks(np.arange(0, 6, 1.0))
    plt.ylim(-0.5, len(list_labels) - 0.5)
    ax.set_yticks(np.arange(0, len(list_labels), 1.0))
    ax.set_yticklabels(list_labels)

    ax.set_ylabel("Label", fontsize=25)
    ax.set_xlabel("Jour", fontsize=25)
    # ax.tick_params(labelsize=20)
    # plt.xlabel('Date (dd.mm)', fontsize=22)


# %% Compute acoustic diversity
# filename_audioF = 'C:/Users/torterma/Documents/Projets_GLIDER/Delgost/DELGOST2_D2 HF_task_status.csv'
filename_audioF = get_csv_file(2, "Select task status csv")
# Put a coordinate on each audio file
for i, f in enumerate(filename_audioF):

    l = pd.read_csv(f, delimiter=",")
    if i == 0:
        list_audioF = l
    else:
        list_audioF = pd.concat([list_audioF, l])


# Download track data
gpx_paths = get_gpx(2)

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

# Compute localisation of each audio file
res = compute_loc_from_time(track_data, time_unix)

# %% Read detections
files_list = get_csv_file(2)

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

det = df_detections.drop_duplicates(subset=["annotation", "start_datetime"])
# DElete unkown detections
det.drop(det[det["annotation"] == "UnidentifiedCalls"].index, inplace=True)

# Create array with unix time of detections
time_det = det["start_datetime"]
time_det_unix = [time.mktime(t.timetuple()) for t in time_det]


AD = np.zeros(len(time_unix))
list_det = []

for i, file in enumerate(time_unix):
    for d in time_det_unix:
        if d == file:
            AD[i] += 1

    list_det.append([file, res[i][1], res[i][2], AD[i]])


np.savetxt(
    "C:/Users/torterma/Documents/Projets_GLIDER/TAAF/Delgost/Acoustic_Diversity_D2.csv",
    [p for p in list_det],
    delimiter=",",
    fmt="%i,%f,%f,%i",
)


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
