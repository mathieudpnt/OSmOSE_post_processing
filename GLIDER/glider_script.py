import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from tqdm import tqdm
import def_func
import utils
import csv
import pytz

from utils.premiers_resultats_utils import load_parameters_from_yaml
from GLIDER.glider_utils import (
    load_glider_nav,
    plot_nav_state,
    compute_acoustic_diversity,
    plot_detection_with_nav_data,
)
from premiers_resultats_utils import plot_detection_timeline


mpl.rcdefaults()
mpl.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["figure.figsize"] = [10, 6]

# %% Load detections and navigation data

df_detections, _, _, labels, _, _, _, _ = load_parameters_from_yaml()

directory = Path(
    r"L:\acoustock\Bioacoustique\DATASETS\GLIDER\GLIDER SEA034\MISSION_46_DELGOST\APRES_MISSION\NAV"
)
glider_data = load_glider_nav(directory)

# %% 'Timeline' figure'
plot_detection_timeline(df_detections)

# %% Compute acoustic diversity

task_status_directory = Path(
    r"L:\acoustock\Bioacoustique\DATASETS\GLIDER\GLIDER SEA034\MISSION_46_DELGOST\ANALYSES\ANNOTATION\HF"
)
task_status_files = list(task_status_directory.glob("*/*task_status.csv"))
time_vector = []
for deployment in task_status_files:
    task_status_filenames = pd.read_csv(deployment)["filename"]

    for fn in task_status_filenames:
        timestamp = def_func.extract_datetime(fn).tz_localize("UTC").timestamp()
        time_vector.append(timestamp)

acoustic_diversity = compute_acoustic_diversity(
    df=df_detections[df_detections["annotation"] != "UnidentifiedCalls"],
    nav=glider_data,
    time_vector=time_vector,
)
acoustic_diversity.to_csv(
    deployment.parents[3] / "carto" / f"Acoustic_Diversity_all_new2.csv", index=False
)

# %% Plot detection according to the depth of the glider

for label in labels:
    plot_detection_with_nav_data(
        df=df_detections, nav=glider_data, criterion="NavState", annotation=label
    )

# %%


def load_from_csv(
    path: Path,
    begin_datetime: pd.Timestamp,
    t_res: int,
    f_res: int,
    date_min: pd.Timestamp = None,
    date_max: pd.Timestamp = None,
    sensitivity: int = None,
    duty_cycle: int = 100,
):

    # get matrix shape
    with open(path, mode="r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        first_line = next(reader)
        shape_f = len(first_line) - 1
        shape_t = sum(1 for _ in reader)

    freq = np.linspace(start=0, num=shape_f * f_res, stop=shape_f, dtype=int).tolist()
    end_datetime = begin_datetime + pd.Timedelta(
        int(t_res // (0.01 * duty_cycle)) * shape_t, "second"
    )
    time = pd.date_range(
        start=begin_datetime,
        end=end_datetime,
        freq=str(int(t_res // (0.01 * duty_cycle))) + "s",
    ).to_list()

    index = [0, len(time)]
    if date_min is not None and date_max is not None:
        time2 = [t for t in time if date_min <= t <= date_max]
        index = [time.index(t) for t in [time2[0], time2[-1]]]
        time = time2

    # matrix_raw = pd.read_csv(path, delimiter=',', skiprows=2, header=None, nrows=index[-1], low_memory=False)

    chunks = []
    chunk_size = 1000

    try:
        for i, chunk in tqdm(
            enumerate(
                pd.read_csv(
                    path,
                    delimiter=",",
                    skiprows=0,
                    header=None,
                    chunksize=chunk_size,
                    low_memory=False,
                    on_bad_lines="skip",
                )
            ),
            desc="Reading csv",
            total=(shape_t // chunk_size) + 1,
            unit="chunk",
        ):
            chunk.replace(-np.inf, np.nan, inplace=True)
            chunk.dropna(inplace=True)
            chunks.append(chunk)
    except pd.errors.ParserError as e:
        print(f"Parser error: {e}")
    print("Done!")

    matrix_raw = pd.concat(chunks, ignore_index=True)

    matrix = matrix_raw.iloc[index[0] : index[-1]]
    matrix = np.array(matrix, dtype=np.float64).transpose()

    if sensitivity:
        matrix += np.abs(sensitivity)

    # self.time = time
    # self.welch = matrix[:-1]
    # self.freq = freq
    # self.f_max = freq[-1]


path = Path(
    r"L:\acoustock\Bioacoustique\DATASETS\GLIDER\GLIDER SEA034\MISSION_58_OHAGEODAMS_2023\ANALYSES\LTAS\ltsa_nfft96000_winsize1min.csv"
)
begin_datetime = pd.Timestamp("2024-02-22 03:54:44+00:00")
# %%
welch = matrix[:-1]
dyn_min = 0
dyn_max = 120


fig, ax = plt.subplots()
plt.imshow(
    welch[:-1, :-1],
    aspect="auto",
    vmin=dyn_min,
    vmax=dyn_max,
    origin="lower",
    extent=[time[0], time[-1], freq[0], freq[-1]],
)
plt.colorbar(label="dB ref 1µPa² @ 1m")

duration = time[-1] - time[0]


date_fmt = "%d-%B"
locator = mdates.DayLocator(interval=1)

tz = pytz.FixedOffset(time[0].utcoffset().total_seconds() // 60)

ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt, tz=tz))
ax.xaxis.set_major_locator(locator)


plt.title(
    f"LTAS from {time[0].strftime('%Y/%m/%d %H:%M')} to {time[-1].strftime('%Y/%m/%d %H:%M')}"
)
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time")
plt.tight_layout()

plt.show()

# %% Navigation data
input_dir = Path(
    r"L:\acoustock\Bioacoustique\DATASETS\GLIDER\GLIDER SEA034\MISSION_58_OHAGEODAMS_2023\APRES_MISSION\NAV"
)
df_nav = load_glider_nav(input_dir)

# LTAS data
path_LTAS = r"C:\Users\dupontma2\Downloads\LTAS_all.npz"
npz_mat = np.load(path_LTAS, allow_pickle=True)
plot_nav_state(df_nav, npz_mat)
