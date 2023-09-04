# This script is used to check whether or not the number of wav files and pgdf files are the same

# from post_processing_detections.utilities.def_func import find_files

# wav_files = find_files(f_type='file', ext='wav')
# pgdf_files = find_files(f_type='dir', ext='pgdf')

# print('\n{0} wav files selected'.format(len(wav_files)))
# print('{0} pgdf files selected'.format(len(pgdf_files)))

import os
import pandas as pd
from tkinter import filedialog
from tkinter import Tk

#wav files
root = Tk()
root.withdraw()
path = filedialog.askdirectory(title='Select wav folder')

folder_names = []
file_counts = []

for folder_name in os.listdir(path):
    folder_path = os.path.join(path, folder_name)
    if os.path.isdir(folder_path):
        wav_files = [file for file in os.listdir(folder_path) if file.endswith(".wav")]
        folder_names.append(folder_name)
        file_counts.append(len(wav_files))

data = {'Folder Name': folder_names, 'File Count': file_counts}
df = pd.DataFrame(data)

root = Tk()
root.withdraw()
path2 = filedialog.askdirectory(title='Select pgdf folder')

# pgdf files
folder_names2 = []
file_counts2 = []

for folder_name in os.listdir(path2):
    folder_path = os.path.join(path2, folder_name)
    if os.path.isdir(folder_path):
        pgdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pgdf")]
        folder_names2.append(folder_name)
        file_counts2.append(len(pgdf_files))

data2 = {'Folder Name': folder_names2, 'File Count': file_counts2}
df2 = pd.DataFrame(data2)


print('wav files : {0}'.format(sum(file_counts)))
print('pgdf files : {0}'.format(sum(file_counts2)))