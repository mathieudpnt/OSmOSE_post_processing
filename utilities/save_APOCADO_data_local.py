import os
import shutil
from tqdm import tqdm
import glob
import json

# %%
path_json = [r'L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO',
             r'Y:\Bioacoustique\APOCADO2',
             r'Z:\Bioacoustique\DATASETS\APOCADO3'
             ]

list_json = [file_path for path in path_json for file_path in glob.glob(os.path.join(path, "**/metadata.json"), recursive=True)]

destination_folder = r'C:\Users\dupontma2\Desktop\data_local\files'
os.makedirs(destination_folder, exist_ok=True)

# Iterate over each CSV file and copy it to the destination folder
for file_path in tqdm(list_json):

    r_file = open(file_path, 'r')
    metadata = json.load(r_file)
    r_file.close()

    ID = 'C' + str(metadata['campaign']) + 'D' + str(metadata['deployment']) + ' ST' + str(metadata['recorder'])

    file_name = ID + ' - metadata.json'

    file_file_mt = metadata['origin file metadata']
    filename_file_mt = ID + ' - origin file_metadata.csv'

    file_mt1 = metadata['origin metadata file']
    filename_mt1 = ID + ' - origin metadata file.csv'

    file_ts1 = metadata['origin timestamp file']
    filename_ts1 = ID + ' - origin timestamp file.csv'

    file_mt2 = metadata['segment metadata file']
    filename_mt2 = ID + ' - segment metadata file.csv'

    file_ts2 = metadata['segment timestamp file']
    filename_ts2 = ID + ' - segment timestamp file.csv'

    file_p = metadata['pamguard detection file']
    filename_p = ID + ' - pamguard.csv'

    file_t = metadata['thalassa detection file']
    filename_t = ID + ' - thalassa.csv'

    if 'aplose file' in metadata:
        file_ap = metadata['aplose file']
        filename_ap = ID + ' - aplose.csv'

    # Construct destination folder
    destination_folder1 = os.path.join(destination_folder, ID)
    os.makedirs(destination_folder1, exist_ok=True)

    # Construct destination path
    destination_path = os.path.join(destination_folder1, file_name)
    destination_path2 = os.path.join(destination_folder1, filename_file_mt)
    destination_path3 = os.path.join(destination_folder1, filename_mt1)
    destination_path4 = os.path.join(destination_folder1, filename_ts1)
    destination_path5 = os.path.join(destination_folder1, filename_mt2)
    destination_path6 = os.path.join(destination_folder1, filename_ts2)
    destination_path7 = os.path.join(destination_folder1, filename_p)
    destination_path8 = os.path.join(destination_folder1, filename_t)
    if 'aplose file' in metadata:
        destination_path9 = os.path.join(destination_folder1, filename_ap)

    # Copy the file
    shutil.copyfile(file_path, destination_path)
    shutil.copyfile(file_file_mt, destination_path2)
    shutil.copyfile(file_mt1, destination_path3)
    shutil.copyfile(file_ts1, destination_path4)
    shutil.copyfile(file_mt2, destination_path5)
    shutil.copyfile(file_ts2, destination_path6)
    shutil.copyfile(file_p, destination_path7)
    shutil.copyfile(file_t, destination_path8)
    if 'aplose file' in metadata:
        shutil.copyfile(file_ap, destination_path9)

print(f"{len(list_json)} files copied to {destination_path}")
