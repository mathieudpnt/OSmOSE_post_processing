"""
Copy all sud files from base folder to a specified folder and preserving the folder and subfolders architecture

"""

import glob
import os
from tqdm import tqdm
import shutil

# base_folder = r'L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO'
# base_folder = r'L:\acoustock2\Bioacoustique\APOCADO2'
base_folder = r"L:\acoustock3\Bioacoustique\DATASETS\APOCADO3"


campaign_folders = [
    r"Campagne 10 part1",
    r"Campagne 11",
]

destination_folder = r"D:\data"

files = [
    glob.glob(os.path.join(base_folder, f, "**/*.sud"), recursive=True)
    for f in campaign_folders
]
files = [item for sublist in files for item in sublist]
dirs = sorted(list(set([os.path.dirname(os.path.dirname(file)) for file in files])))


# define a function to ignore the 'thalassa/spectro' directory
def ignore_spectro_thalassa(dir, files):
    if "thalassa" in dir:
        return ["spectro"] if "spectro" in files else []
    else:
        return []


for d in tqdm(dirs):

    # sud folder
    sud_folder = os.path.join(d, "sud")
    if os.path.exists(sud_folder):
        dest_path = sud_folder.replace(base_folder, destination_folder, 1)
        shutil.copytree(sud_folder, dest_path)

    # # analysis folder
    # analysis_folder = os.path.join(d, 'analysis')
    # if os.path.exists(analysis_folder):
    #     dest_path = analysis_folder.replace(base_folder, destination_folder, 1)
    #     shutil.copytree(analysis_folder, dest_path, ignore=ignore_spectro_thalassa)

    # csv folder
    csv_folder = os.path.join(d, "csv")
    if os.path.exists(csv_folder):
        dest_path = csv_folder.replace(base_folder, destination_folder, 1)
        shutil.copytree(csv_folder, dest_path)

    # xml folder
    xml_folder = os.path.join(d, "xml")
    if os.path.exists(xml_folder):
        dest_path = xml_folder.replace(base_folder, destination_folder, 1)
        shutil.copytree(xml_folder, dest_path)

    # log folder
    log_folder = os.path.join(d, "log")
    if os.path.exists(log_folder):
        dest_path = log_folder.replace(base_folder, destination_folder, 1)
        shutil.copytree(log_folder, dest_path)

    print(f"{d} copied successfully")
