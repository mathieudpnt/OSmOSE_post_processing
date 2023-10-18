# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:25:16 2023

@author: torterma
"""
import pandas as pd
from tkinter import filedialog
from tkinter import Tk
from utilities.def_func import get_csv_file, sorting_detections, t_rounder, get_timestamps, input_date, suntime_hour
import numpy as np

#%%

def readRaven(msg: str) -> pd.DataFrame:
    
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title = msg,
        filetypes=[('csv files', '*.csv')],
        parent=None
    )
    
    data = pd.read_csv(file_path, sep=";")
    #data.columns = ["a", "b", "c", "etc."]
    
    
    return data
    
    
RavenTable = readRaven('Select Raven Selection Table')
RavenTable_clean = RavenTable.loc[np.isnan(RavenTable['double_check'])]


dataset_name = 'CETIROISE_F1'
#filename = RavenTable_clean['filename']
end_time = [end-beg for end, beg in zip(RavenTable_clean['End Time (s)'], RavenTable_clean['Begin Time (s)'])]
nb_det = len(RavenTable_clean)
start_frequency = RavenTable_clean['Low Freq (Hz)']
end_frequency = RavenTable_clean['High Freq (Hz)']
species = 'Whistle and Moan'
annotator = 'PG double check'
#start_datetime = pd.to_datetime(RavenTAble['Begin Date Time'],format= '%H:%M:%S' ).dt.time
#end_datetime = start_datetime + pd.to_timedelta(end_time[0], unit='s')


data = {'dataset' : [dataset_name]*nb_det, 'filename' : ['']*nb_det, 'start_time' : [0]*nb_det, 'end_time' : end_time, 'start_frequency' :  start_frequency, 'end_frequency' :  end_frequency,'annotation' : [species]*nb_det, 'annotator' : annotator, 'start_datetime' : RavenTable_clean['Selection'],  'end_datetime' :  RavenTable_clean['Selection']}
df_APLOSE = pd.DataFrame(data)  
#▒ Save the csv file in the same folder as the FPOD results csv file
#df_APLOSE.to_csv('APLOSE.csv', index=False)
















