import glob
import pandas as pd
import numpy as np
from datetime import timedelta
import os
from tqdm import tqdm
import soundfile as sf
from pydub import AudioSegment
import shutil
from utilities.def_func import read_header, extract_datetime


def get_data(dt1, dt2, file_list):
    """Extracts data frames that are comprised between dt1 and dt2
    Parameters :
        dt1 : datetime begin
        dt2 : datetime end
        file_list : DataFrame containing filename / decimated_date / dt_start / dt_end for each original wav file
    Returns :
        samples : relevant data
    """
    samples, info = [], []

    # selection of the files that have data between dt1 and dt2
    selected_files = file_list[
        np.logical_or(
            np.logical_or(
                np.logical_and(file_list.dtstart >= dt1, file_list.dtstart <= dt2),
                np.logical_and(file_list.dtend >= dt1, file_list.dtend <= dt2),
            ),
            np.logical_and(file_list.dtstart <= dt1, file_list.dtend >= dt2),
        )
    ]

    selected_files = selected_files.sort_values(by="dtstart")

    if len(selected_files) == 0:
        raise ValueError(f"No data selected between dt1={dt1} and dt2={dt2}")

    for i in range(len(selected_files)):
        dtfilestart = selected_files.iloc[i]["dtstart"]
        nbech = selected_files.iloc[i]["nbech"]
        filename = selected_files.iloc[i]["filename"]

        data, fs = sf.read(filename)
        info = sf.info(filename).subtype

        posstart = int(np.max([0, fs * (dt1 - dtfilestart).total_seconds()]))
        posfin = int(np.min([nbech, fs * (dt2 - dtfilestart).total_seconds()]))

        samples.append(data[posstart:posfin])

    samples = np.concatenate(samples)
    return samples, info


# %% Files list

dir_file = os.path.dirname(
    "L:/acoustock3/Bioacoustique/DATASETS/MIRACETI_ACOUSTIQUE/2018/"
)
dur_new = 10  # duration of the reshaped files in seconds


# %%

lst_wav = sorted(glob.glob(os.path.join(dir_file, "**/*.wav"), recursive=True))
lst_mp3 = sorted(glob.glob(os.path.join(dir_file, "**/*.mp3"), recursive=True))

dir_reformatted = os.path.join(dir_file, "reformatted")
if not os.path.exists(dir_reformatted):
    os.mkdir(dir_reformatted)

for file in lst_mp3:
    sound = AudioSegment.from_mp3(file)
    sound = sound.set_channels(1)
    new_fn = os.path.join(
        dir_reformatted, os.path.basename(file).split(".")[0] + "_formatted.wav"
    )
    sound.export(new_fn, format="wav")

for file in lst_wav:
    sound = AudioSegment.from_wav(file)
    sound = sound.set_channels(1)
    new_fn = os.path.join(
        dir_reformatted, os.path.basename(file).split(".")[0] + "_formatted.wav"
    )
    sound.export(new_fn, format="wav")

# Reshape & normalize

lst_file_formatted = sorted(glob.glob(os.path.join(dir_reformatted, "*.wav")))

df_file = pd.DataFrame(
    index=range(len(lst_file_formatted)),
    columns=["filename", "dtstart", "dtend", "nbech", "duration", "sample_rate"],
)

for i, file in enumerate(lst_file_formatted):
    _, nbech, fs, _, nb_sec = read_header(file)
    namefile = os.path.basename(file)
    dt_start_file = extract_datetime(var=namefile)
    dt_end_file = dt_start_file + pd.to_timedelta(arg=nb_sec, unit="s")
    df_file.loc[i] = [file, dt_start_file, dt_end_file, nbech, nbech / fs, fs]

df_file.reset_index(drop=True, inplace=True)

dir_reshaped = os.path.join(dir_file, "reshaped_" + str(dur_new) + "s")
if not os.path.exists(dir_reshaped):
    os.mkdir(dir_reshaped)


for i, file in enumerate(lst_file_formatted):
    _, nbech, fs, _, nb_sec = read_header(file)
    namefile = os.path.basename(file)

    dt_begin = extract_datetime(var=namefile)
    dt_end = dt_begin + pd.to_timedelta(arg=nb_sec, unit="s")

    if not os.path.exists(os.path.join(dir_reshaped, str(fs))):
        os.mkdir(os.path.join(dir_reshaped, str(fs)))

    while dt_begin <= dt_end:

        nameparts = os.path.basename(file).split("_")
        nameparts[1] = dt_begin.strftime("%Y%m%dT%H%M%S")
        new_fn = "_".join(nameparts)
        path_save = os.path.join(dir_reshaped, str(fs), new_fn)

        dt_fileend = dt_begin + timedelta(seconds=dur_new)

        data_out, info = get_data(dt1=dt_begin, dt2=dt_fileend, file_list=df_file)
        data_norm = np.where(
            data_out > 0,
            data_out / data_out.max(),
            np.where(data_out < 0, -data_out / data_out.min(), data_out),
        )
        sf.write(file=path_save, data=data_norm, samplerate=fs, subtype=info)

        print(f"{new_fn}  -  {len(data_out) / fs}s")

        dt_begin += timedelta(seconds=dur_new)

shutil.rmtree(dir_reformatted)
print(f"\nfiles saved in {os.path.dirname(path_save)}")
