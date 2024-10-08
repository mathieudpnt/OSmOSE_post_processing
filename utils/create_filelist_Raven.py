# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:49:24 2023

Create a text file with the name of the files contained in all the subfolders 
of your dataset for opening them in RavenPro

@author: torterma
"""
import glob
from tkfilebrowser import askopendirnames
from tkinter import *
from tkinter.filedialog import asksaveasfilename
import tkinter

# 1 - Ask user which folders they want to double check

list_dirs = askopendirnames(title="Select folders")


files_list = []
for directory in list_dirs:
    files_list.extend(glob.glob(directory + "\\*.wav"))

# %% 2 - Save the Raven txt at the root of wav files
date_first_dir = list_dirs[0].rpartition("\\")[-1]
date_last_dir = list_dirs[-1].rpartition("\\")[-1]
save_dir = (
    directory.rpartition("\\")[0] + "\\Raven_" + date_first_dir + date_last_dir + ".txt"
)
with open(save_dir, "w") as f:
    for item in files_list:
        f.write("%s\n" % item)

print("File saved in " + save_dir)


# 2 bis - Ask user where they want to save they Raven .txt
# file = asksaveasfilename(filetype = [("Text files", "*txt")], defaultextension=".txt", title='Save your Raven .txt')

# with open(file, 'w') as f:
#     for item in files_list:
#         f.write("%s\n" % item)
