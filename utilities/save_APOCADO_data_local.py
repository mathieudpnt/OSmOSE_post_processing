import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from tqdm import tqdm
import numpy as np
import re
from collections import Counter
import pytz
import glob
import json

os.chdir(r'U:/Documents_U/Git/post_processing_detections')
from utilities.def_func import stat_box_day, stats_diel_pattern, sorting_detections, get_season

# %%
path_json = [r'L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO',
             r'Y:\Bioacoustique\APOCADO2',
             r'Z:\Bioacoustique\DATASETS\APOCADO3'
             ]

list_json = [file_path for path in path_json for file_path in glob.glob(os.path.join(path, "**/metadata.json"), recursive=True)]

destination_folder = r'C:\Users\dupontma2\Desktop\data_local'
os.makedirs(destination_folder, exist_ok=True)

# Iterate over each CSV file and copy it to the destination folder
for file_path in tqdm(list_json):
    
    r_file = open(file_path, 'r')
    metadata = json.load(r_file)
    r_file.close()

    file_name = metadata['deployment ID'] + ' - metadata.json'
    file_p = metadata['pamguard detection file']
    file_t = metadata['thalassa detection file']
    filename_p = metadata['deployment ID'] + ' - PG.csv'
    filename_t = metadata['deployment ID'] + ' - thalassa.csv'

    # Construct destination folder
    destination_folder1 = os.path.join(destination_folder, metadata['deployment ID'])
    os.makedirs(destination_folder1, exist_ok=True)
    
    # Construct destination path
    destination_path = os.path.join(destination_folder1, file_name)
    destination_path2 = os.path.join(destination_folder1, filename_p)
    destination_path3 = os.path.join(destination_folder1, filename_t)
    
    # Copy the file
    shutil.copyfile(file_path, destination_path)
    shutil.copyfile(file_p, destination_path2)
    shutil.copyfile(file_t, destination_path3)
print(f"{len(list_json)} files copied to {destination_path}")
