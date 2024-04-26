import os
import glob
import json
import pandas as pd
from tqdm import tqdm
import re
from pathlib import Path

os.chdir(r'U:/Documents_U/Git/post_processing_detections')
from utilities.def_func import sorting_detections, extract_datetime, read_header

# %% Write metadata files

path_csv = [r'L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO',
            r'Y:\Bioacoustique\APOCADO2',
            r'Z:\Bioacoustique\DATASETS\APOCADO3']

pamguard_csv = [glob.glob(os.path.join(p, r'**/PG_rawdata_**.csv'), recursive=True) for p in path_csv]
pamguard_csv = [file for sublist in pamguard_csv for file in sublist]

thalassa_csv = [glob.glob(os.path.join(p, r'**/thalassa_**.csv'), recursive=True) for p in path_csv]
thalassa_csv = [file for sublist in thalassa_csv for file in sublist]

deploy = pd.read_excel(r'L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO\APOCADO - Suivi déploiements.xlsm', skiprows=[0])
deploy = deploy.loc[(deploy['check heure Raven'] == 1)].reset_index(drop=True)

deploy['duration deployment'] = [pd.Timestamp.combine(deploy['date recovery'][n], deploy['time recovery'][n])
                                 - pd.Timestamp.combine(deploy['date deployment'][n], deploy['time deployment'][n]) for n in range(len(deploy))]

# %%
for p, t in tqdm(zip(pamguard_csv, thalassa_csv), total=len(pamguard_csv), ncols=50, mininterval=3):

    df_detections = pd.read_csv(p)

    tz = pd.to_datetime(df_detections['start_datetime'])[0].tz

    [ID0] = list(set(df_detections['dataset']))
    platform = re.search(r'C\d{1,2}D\d{1,2}', ID0).group()  # campaign and deployment identifier
    recorder = re.search(r'ST\d+', ID0).group().split('ST')[-1]  # instrument identifier
    ID_deploy = platform + ' ST' + recorder

    rank = deploy[deploy['ID deployment'] == ID_deploy].index.item()

    dt_deployment_beg = pd.Timestamp(pd.Timestamp.combine(deploy['date deployment'][rank], deploy['time deployment'][rank]), tz=tz)
    dt_deployment_end = pd.Timestamp(pd.Timestamp.combine(deploy['date recovery'][rank], deploy['time recovery'][rank]), tz=tz)

    n_instru = deploy['recorder number'][rank]

    wav_folder = os.path.join(Path(p).parents[3], 'wav')
    # wav_files = glob.glob(os.path.join(wav_folder, "*.wav"), recursive=True)
    # wav_names = [os.path.basename(file) for file in wav_files]

    # wav_dt_beg = [extract_datetime(var=file, tz=tz) for file in wav_files]
    # durations = [read_header(file)[-1] for file in wav_files]
    # wav_dt_end = [wav_dt_beg[i] + pd.Timedelta(value=durations[i], unit='s') for i in range(len(wav_files))]

    # index_wav = []
    # for j in range(len(wav_files)):
    #     # first and last file indexes corresponding to the deployment
    #     if wav_dt_beg[j] <= dt_deployment_beg <= wav_dt_end[j] or wav_dt_beg[j] <= dt_deployment_end <= wav_dt_end[j]:
    #         index_wav.append(j)

    # # filtering files
    # if len(index_wav) == 1:
    #     wav_names = wav_names[index_wav[0]]
    #     wav_files = wav_files[index_wav[0]]
    #     durations = durations[index_wav[0]]
    # elif len(index_wav) == 2:
    #     wav_names = wav_names[index_wav[0]:index_wav[1] + 1]
    #     wav_files = wav_files[index_wav[0]:index_wav[1] + 1]
    #     durations = durations[index_wav[0]:index_wav[1] + 1]
    # else:
    #     raise ValueError('index_wav error')

    metadata = {
        'project': 'APOCADO',
        'campaign': int(deploy['campaign'][rank]),
        'deployment': int(deploy['deployment'][rank]),
        # 'platform': platform,
        'recorder': recorder,
        'recorder number': int(deploy['recorder number'][rank]),

        'latitude': float(deploy['latitude'][rank]),
        'longitude': float(deploy['longitude'][rank]),
        'datetime deployment': dt_deployment_beg.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'datetime recovery': dt_deployment_end.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'duration': str(deploy['duration deployment'][rank]),

        'vessel': deploy['vessel'][rank],
        'port': deploy['port'][rank],
        'net': deploy['net'][rank],
        'net length': int(deploy['net length'][rank]),
        'species': deploy['species'][rank],

        'wav_folder': wav_folder,
        # 'first file': wav_names[0],
        # 'last file': wav_names[-1],

        'pamguard detection file': p,

        'thalassa detection file': t,
    }

    out_file = os.path.join(Path(p).parents[1], 'metadata.json')
    with open(out_file, 'w+') as f:
        json.dump(metadata, f, indent=1, ensure_ascii=False)

# %% reorganize folder structures of deployments
'''
import shutil
list_json = glob.glob(os.path.join(r'L:/acoustock2/Bioacoustique/APOCADO2/campagne 7', "**/metadata.json"), recursive=True)
list_folder = [os.path.dirname(json) for json in list_json]

for folder in tqdm(list_folder):

    # PG part
    if not os.path.exists(os.path.join(folder, 'pamguard')):
        os.mkdir(os.path.join(folder, 'pamguard'))

    source_dir = os.path.join(folder, 'PG Binary')
    destination_dir = os.path.join(folder, 'pamguard', 'PG Binary')
    try:
        os.rename(source_dir, destination_dir)
    except Exception as e:
        print(f"Failed to move {source_dir} to {destination_dir}: {e}")

    PG_files = glob.glob(os.path.join(folder, "PG_rawdata_**"), recursive=True)
    [shutil.move(PG, os.path.join(folder, 'pamguard')) for PG in PG_files]

    # result part
    source_dir = os.path.join(folder, "result")
    destination_dir = folder
    total_items = sum([len(files) + len(dirs) for root, dirs, files in os.walk(source_dir)])
    index = list_folder.index(folder) + 1
    f_name = os.path.basename(folder)
    with tqdm(total=total_items, desc=f"{f_name} -- Moving files and directories {index}/{len(list_folder)}", unit="item", unit_scale=True, ncols=100, leave=False, dynamic_ncols=True, position=0, file=None, miniters=1, mininterval=0.1, ascii=False, disable=False, unit_divisor=1000, gui=False) as pbar:
        # Walk through the source directory
        for root, dirs, files in os.walk(source_dir):
            for dir in dirs:
                source_path = os.path.join(root, dir)
                dest_path = os.path.join(destination_dir, os.path.relpath(source_path, source_dir))
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                pbar.update(1)  # Update progress bar for each directory
            for file in files:
                source_path = os.path.join(root, file)
                dest_path = os.path.join(destination_dir, os.path.relpath(source_path, source_dir))
                os.rename(source_path, dest_path)
                pbar.update(1)  # Update progress bar for each file

    shutil.rmtree(source_dir)

print('\n\ndone')
'''
