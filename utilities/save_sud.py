import glob
from pathlib import Path
import os
from tqdm import tqdm
import shutil

#files = glob.glob(os.path.join('L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO', '**/*.sud'), recursive=True)\
            #+glob.glob(os.path.join('L:/acoustock2/Bioacoustique/APOCADO2', '**/*.sud'), recursive=True)

#%%

def get_total_size_in_gb(file_paths):
    total_size_bytes = sum(os.path.getsize(file_path) for file_path in tqdm(file_paths))
    total_size_gb = total_size_bytes / (1024 ** 3)  # Convert bytes to GB
    return total_size_gb


#total_size_gb = get_total_size_in_gb(files)
#total_size_tb = total_size_gb/1024
#print(f"Total size of files: {total_size_gb:.2f} GB")
#print(f"Total size of files: {total_size_tb:.2f} TB")

#%% Copy all sud files from base folder to a specified folder and preserving the folder and subfolders architecture

base_folder = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO'
destination_folder = 'D:/'

files = glob.glob(os.path.join(base_folder, '**/*.sud'), recursive=True)
dirs = [os.path.dirname(file) for file in files]

for i in tqdm(range(len(files))):
    dest_path = dirs[i].replace(base_folder, destination_folder, 1)
    os.makedirs(dest_path, exist_ok=True)
    shutil.copy2(files[i], dest_path)

print('Folder structure copied successfully')


