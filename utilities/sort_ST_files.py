from tkinter import filedialog
from tkinter import Tk
import glob
from pathlib import Path
import os
from tqdm import tqdm

root = Tk()
root.withdraw()
master_folder = filedialog.askdirectory(title="Select master folder")
folders = [ f.path for f in os.scandir(master_folder) if f.is_dir() ]

for folder in tqdm(folders) :

    files = glob.glob(os.path.join(folder, "**/*.*"), recursive=True)
    
    sorted_files = {}
    for file in files:
        ext = os.path.splitext(file)[1]
        if ext not in sorted_files:
            sorted_files[ext] = []
        sorted_files[ext].append(file)
    
    extensions = [list(sorted_files.keys())[i].split('.')[-1] for i in range(len(sorted_files.keys()))]
   
    [Path.mkdir(Path(folder, extension), parents=True, exist_ok=True) for extension in extensions];

    [[os.replace(file, os.path.join(folder, extensions[i], os.path.basename(file))) for file in sorted_files[list(sorted_files.keys())[i]]] for i in range(len(sorted_files))];
    
    [print('-{0}:'.format(extensions[i]), len(sorted_files[list(sorted_files.keys())[i]]), 'files' ) for i in range(len(list(sorted_files.keys())))];











