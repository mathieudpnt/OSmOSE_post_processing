# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:20:49 2023

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
from utilities.def_func import read_header


############## Chose timezone ###################
tz = pytz.FixedOffset(120)

# Select wav folder (parent folder, can contain subfolders with wav data)
root = Tk()
root.withdraw()
path_raw_audio = filedialog.askdirectory(title='Select wav folder') + '/'

# get the list of the wav files
list_wav_file = glob.glob(path_raw_audio + '**/*.wav', recursive = True)

# Compute timestamp and nameof each wav file
timestamp=[]
filename=[]
duration = []
origin_sr = []
size = []
sampwidth = []
channel_count = []
status_read_header = []

for name_file in tqdm(list_wav_file):
    # filename
    filename.append(os.path.basename(name_file)) 
    
    # timestamp
    dateds = os.path.basename(name_file)[:-4]
    #################### Change wav name format ###############################        
    date_obj = datetime.datetime.strptime(dateds, 'channelA_%Y-%m-%d_%H-%M-%S') 
    date_obj = date_obj.replace(tzinfo=tz)      
    dates = datetime.datetime.strftime(date_obj, '%Y-%m-%dT%H:%M:%S.%f%z')

    [sampwidth_u, frames_u, samplerate_u, channels_u, duration_u] = read_header(name_file)


    timestamp.append(dates)    
    filename.append(os.path.basename(name_file))    
    sampwidth.append(sampwidth_u)
    duration.append(duration_u)
    origin_sr.append(samplerate_u)
    channel_count.append(channels_u)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    