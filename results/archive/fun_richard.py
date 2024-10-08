import glob
import pandas as pd
import numpy as np
from datetime import timedelta
import os
from tqdm import tqdm
import soundfile as sf
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

    # selection of relevant frames for each selected file
    for i in range(len(selected_files)):
        dtfilestart = selected_files.iloc[i]["dtstart"]
        nbech = selected_files.iloc[i]["nbech"]
        filename = selected_files.iloc[i]["filename"]

        data, fs = sf.read(filename)
        info = sf.info(filename).subtype

        # start position of the original file (in frames)
        posstart = int(np.max([0, fs * (dt1 - dtfilestart).total_seconds()]))
        # print(f'possart={posstart}')

        # end position of the original file (in frames)
        posfin = int(np.min([nbech, fs * (dt2 - dtfilestart).total_seconds()]))
        # print(f'posfin={posfin}')

        # nbframes = int(posfin - posstart)
        # print(f'nbframes= {nbframes}')

        samples.append(data[posstart:posfin])

    samples = np.concatenate(samples)
    return samples, info


lst_fichiers = sorted(
    glob.glob(
        "L:/acoustock3/Bioacoustique/DATASETS/MIRACETI_ACOUSTIQUE/2017 concat/44100/*.wav"
    )
)
# %% From multichannel to mono if necessary
"""
from pydub import AudioSegment

for file in tqdm(lst_fichiers) :
    sound = AudioSegment.from_wav(file)
    sound = sound.set_channels(1)
    new_fn = os.path.join(os.path.dirname(file), 'reformatted', os.path.basename(file))
    sound.export(new_fn, format="wav")
"""
# TODO : mettre un datetime aux noms de fichiers
# %%
dir_reshaped = os.path.join(os.path.dirname(lst_fichiers[0]), "reshaped")
if not os.path.exists(dir_reshaped):
    os.mkdir(dir_reshaped)

df_file = pd.DataFrame(
    index=range(len(lst_fichiers)),
    columns=[
        "filename",
        "dtstart",
        "dtend",
        "nbech",
        "duration",
        "cumsum",
        "sample_rate",
    ],
)
cumsum = 0

for i, file in tqdm(enumerate(lst_fichiers), total=len(lst_fichiers)):
    _, nbech, fs, _, nb_sec = read_header(file)
    cumsum += nb_sec
    namefile = os.path.basename(file)

    dt_start_file = extract_datetime(var=namefile)
    dt_end_file = dt_start_file + pd.to_timedelta(arg=nb_sec, unit="s")

    df_file.loc[i] = [file, dt_start_file, dt_end_file, nbech, nbech / fs, cumsum, fs]

df_file.reset_index(drop=True, inplace=True)
# %%
dur_new = 10

for i, file in enumerate(lst_fichiers):

    dt_begin = df_file["dtstart"].iloc[i]
    dt_end = df_file["dtend"].iloc[i]

    while dt_begin <= dt_end:

        nameparts = os.path.basename(file).split("_")
        nameparts[1] = dt_begin.strftime("%Y%m%dT%H%M%S")
        new_fn = "_".join(nameparts)
        path_save = os.path.join(dir_reshaped, new_fn)

        # Date de fin du nouveau fichier wav redécoupé
        dt_fileend = dt_begin + timedelta(seconds=dur_new)

        data_out, info = get_data(dt1=dt_begin, dt2=dt_fileend, file_list=df_file)

        sf.write(path_save, data_out, fs, subtype=info)
        print(f"{new_fn}  -  {len(data_out) / fs}s")

        dt_begin += timedelta(seconds=dur_new)

print(f"\nfiles saved in {os.path.dirname(path_save)}")
