from LTAS import LTAS
from def_func import extract_datetime
import pandas as pd
import os
from matplotlib import pyplot as plt
import pickle
import glob
import pytz
import json
from tqdm import tqdm
from pathlib import Path
from SoundTrap import RECORDER_GAIN
import matplotlib.image as mpimg

# %% Load data
with open(
    r"L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO\data_Deployment.pkl",
    "rb",
) as f:
    data = pickle.load(f)

path_acoustock = [
    r"L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO",
    r"Y:\Bioacoustique\APOCADO2",
    r"Z:\Bioacoustique\DATASETS\APOCADO3",
]

list_signal = [
    file_path
    for path in path_acoustock
    for file_path in glob.glob(
        os.path.join(path, r"**\signal.ravensignal"), recursive=True
    )
]
list_ltas_csv = [
    file_path
    for path in path_acoustock
    for file_path in glob.glob(os.path.join(path, r"**\ltsa.csv"), recursive=True)
]
list_ltas = [
    file_path
    for path in path_acoustock
    for file_path in glob.glob(os.path.join(path, r"**\ltsa.ravenltsa"), recursive=True)
]
list_img = [
    file_path
    for path in path_acoustock
    for file_path in glob.glob(
        os.path.join(path, r"**\soundscape\LTAS**.png"), recursive=True
    )
]
list_img2 = [
    file_path
    for path in path_acoustock
    for file_path in glob.glob(
        os.path.join(path, r"**\soundscape\PSD**.png"), recursive=True
    )
]
list_npz = [
    file_path
    for path in path_acoustock
    for file_path in glob.glob(
        os.path.join(path, r"**\soundscape\LTAS**.npz"), recursive=True
    )
]
# %%
time_resolution_LTAS, frequency_resolution_LTAS = [], []
begin_LTAS, campaign_LTAS, recorder_LTAS = [], [], []

for i in tqdm(range(len(list_signal))):
    with open(list_ltas[i], "r", errors="ignore") as file:
        for line in file:
            clean_line = line.strip()
            if "timeResolution" in clean_line:
                parts = line.split(":")
                if len(parts) > 1:
                    time_resolution_LTAS.append(int(parts[1].strip().strip('",')))
            if "frequencyResolution" in clean_line:
                parts = line.split(":")
                if len(parts) > 1:
                    frequency_resolution_LTAS.append(int(parts[1].strip().strip('",')))
                    break

    with open(list_signal[i], "rb") as f:
        text = f.read()
    path = json.loads(text.decode("latin-1"))["fileList"][0]["filePath"]["path"]
    path_splitted = json.loads(text.decode("latin-1"))["fileList"][0]["filePath"][
        "path"
    ].split("\\")

    for elem in path_splitted:
        if "Campagne" in elem:
            campaign_LTAS.append(int(elem.split(" ")[1]))

    recorder_LTAS.append(path_splitted[-3])

    begin_LTAS.append(extract_datetime(path.split(".")[-2]))

data_LTAS = pd.DataFrame(
    list(
        zip(
            campaign_LTAS,
            recorder_LTAS,
            begin_LTAS,
            time_resolution_LTAS,
            frequency_resolution_LTAS,
            list_ltas_csv,
        )
    ),
    columns=[
        "campaign_LTAS",
        "recorder_LTAS",
        "begin_LTAS",
        "time_resolution_LTAS",
        "frequency_resolution_LTAS",
        "csv",
    ],
).sort_values(["campaign_LTAS", "begin_LTAS"])

# %% Select deployment

(
    name,
    campaign,
    deployment,
    recorder,
    gain,
    begin_date,
    deployment,
    recovery,
    csv_LTAS,
    check_LTAS,
    time_resolution,
    frequency_resolution,
    output_folder,
) = ([], [], [], [], [], [], [], [], [], [], [], [], [])
for i, d in enumerate(data):
    name.append(d.name)
    campaign.append(d.campaign)
    deployment.append(d.deployment)
    recorder.append(d.recorder)
    deployment.append(d.datetime_deployment)
    recovery.append(d.datetime_recovery)

    output_folder.append(os.path.join(Path(d.path_metadata).parents[1], "soundscape"))

    LTAS_file = data_LTAS[
        (data_LTAS["campaign_LTAS"] == d.campaign)
        & (data_LTAS["recorder_LTAS"] == d.recorder)
    ].iloc[-1]

    dt_LTAS = LTAS_file["begin_LTAS"]
    begin_date_naive = extract_datetime(
        os.path.basename(glob.glob(os.path.join(d.wav_folder, "*.wav"))[0])
    )
    tz = pytz.FixedOffset(d.segment_timestamp[0].utcoffset().total_seconds() // 60)
    begin_date.append(begin_date_naive.tz_localize(tz=tz))

    check_LTAS.append(dt_LTAS == begin_date_naive)

    csv_LTAS.append(LTAS_file["csv"])
    time_resolution.append(LTAS_file["time_resolution_LTAS"])
    frequency_resolution.append(LTAS_file["frequency_resolution_LTAS"])

    gain.append(RECORDER_GAIN[d.recorder])

if not all(check_LTAS):
    raise ValueError("Error check Raven")

df = pd.DataFrame(
    list(
        zip(
            name,
            campaign,
            deployment,
            recorder,
            gain,
            begin_date,
            deployment,
            recovery,
            csv_LTAS,
            time_resolution,
            frequency_resolution,
            output_folder,
        )
    ),
    columns=[
        "name",
        "campaign",
        "deployement",
        "recorder",
        "gain",
        "begin_campaign",
        "deployment",
        "recovery",
        "csv_LTAS",
        "time_resolution_LTAS",
        "frequency_resolution_LTAS",
        "output_folder",
    ],
).sort_values(["campaign", "deployment"])


# %% plot LTAS and save figure

for i in tqdm(range(len(df))):
    LTAS_deploy = LTAS(
        path=df["csv_LTAS"][i],
        t_res=df["time_resolution_LTAS"][i],
        f_res=df["frequency_resolution_LTAS"][i],
        begin_datetime=df["begin_campaign"][i],
        datetime_min=df["deployment"][i],
        datetime_max=df["recovery"][i],
        sensitivity=abs(df["gain"][i]),
    )
    # print(LTAS_deploy)
    out_folder = df["output_folder"][i]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    LTAS_deploy.plot_LTAS(
        output_path=out_folder, output_name=df["name"][i], dyn_min=20, dyn_max=100
    )
    LTAS_deploy.plot_PSD(output_path=out_folder, output_name=df["name"][i])


# %%
item = range(0, len(list_npz), 10)
for i in tqdm(item):
    LTAS_load = LTAS(path=list_npz[i])
    LTAS_load.plot_LTAS(dyn_min=20, dyn_max=80)
# %%
# path = r'L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO\Campagne 3\IROISE\335556632\analysis\LTAS\ltsa.csv'

list_path = df["csv_LTAS"].unique()

for path in tqdm(list_path):
    LTAS_load = LTAS(
        path=path,
        t_res=df[df["csv_LTAS"] == path]["time_resolution_LTAS"].unique()[0],
        f_res=df[df["csv_LTAS"] == path]["frequency_resolution_LTAS"].unique()[0],
        begin_datetime=df[df["csv_LTAS"] == path]["begin_campaign"].unique()[0],
        datetime_min=df[df["csv_LTAS"] == path]["deployment"].min(),
        datetime_max=df[df["csv_LTAS"] == path]["recovery"].max(),
        sensitivity=abs(df[df["csv_LTAS"] == path]["gain"].unique()[0]),
    )
    LTAS_load.plot_LTAS(dyn_min=20, dyn_max=80)


# %%
cond1 = "C11D"
cond2 = ""
filter_img = [file for file in list_img if cond1 in file and cond2 in file]

# Read and display the image
for im in filter_img:
    img = mpimg.imread(im)
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.tight_layout()
    plt.show()

# %%
cond1 = "C5D10"
cond2 = "7190"
filter_img = [file for file in list_img if cond1 in file and cond2 in file]

# Read and display the image
for im in filter_img:
    img = mpimg.imread(im)
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.tight_layout()
    plt.show()

# %%
LTAS_plot = []

cond1 = "C6D1"
cond2 = "7178"
filter_img = [file for file in list_npz if cond1 in file and cond2 in file]

for i in filter_img:
    LTAS_load = LTAS(path=i)
    LTAS_plot.append(
        LTAS_load.plot_LTAS(output_path=None, output_name=None, dyn_min=20, dyn_max=100)
    )

    plt.imshow(
        LTAS_load.welch[:-1, :-1],
        aspect="auto",
        vmin=20,
        vmax=100,
        origin="lower",
        extent=[
            LTAS_load.time[0],
            LTAS_load.time[-1],
            LTAS_load.freq[0],
            LTAS_load.freq[-1],
        ],
    )

# fig, ax = plt.subplots()
# plt.plot(LTAS_load.freq[:-1], PSD[0], label="C4D7 ST336363566")
# plt.plot(LTAS_load.freq[:-1], PSD[1], label="C1D2 ST336363566")
#
# plt.grid(True, which="both", color="gainsboro")
# ax.set_ylabel("Amplitude (dB ref 1µPa²/Hz)")
# ax.set_xlabel("Fréquence (Hz)")
# ax.set_xscale("log")
# ax.legend(loc="upper right")
# plt.ylim(20, 140)
# plt.title("PSD")
