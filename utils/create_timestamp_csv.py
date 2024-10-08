# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 08:52:30 2023

@author: torterma
"""

import glob
import pandas as pd
import os
import datetime
from tqdm import tqdm
import pytz
from tkinter import filedialog
from tkinter import Tk


############## Chose timezone ###################
tz = pytz.FixedOffset(120)

# Select wav folder (parent folder, can contain subfolders with wav data)
root = Tk()
root.withdraw()
path_raw_audio = filedialog.askdirectory(title="Select wav folder") + "/"

# get the list of the wav files
list_wav_file = glob.glob(path_raw_audio + "**/*.wav", recursive=True)

# Compute timestamp of each wav file
timestamp = []
filename_rawaudio = []
for name_file in tqdm(list_wav_file):

    dateds = os.path.basename(name_file)[:-4]
    #################### Change wav name format ###############################
    date_obj = datetime.datetime.strptime(dateds, "channelA_%Y-%m-%d_%H-%M-%S")
    date_obj = date_obj.replace(tzinfo=tz)
    dates = datetime.datetime.strftime(date_obj, "%Y-%m-%dT%H:%M:%S.%f%z")

    timestamp.append(dates)

    filename_rawaudio.append(os.path.basename(name_file))

df = pd.DataFrame({"filename": filename_rawaudio, "timestamp": timestamp})
df.sort_values(by=["timestamp"], inplace=True)
df.to_csv(
    os.path.join(path_raw_audio, "timestamp.csv"),
    index=False,
    na_rep="NaN",
    header=["filename", "timestamp"],
)
