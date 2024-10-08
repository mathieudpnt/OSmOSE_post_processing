# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:25:16 2023

@author: torterma
"""
import pandas as pd
from tkinter import filedialog
from tkinter import Tk
from utilities.def_func import (
    get_csv_file,
    sorting_detections,
    t_rounder,
    get_timestamps,
    input_date,
    suntime_hour,
)
import numpy as np
import pytz
import datetime

# %%

tz = pytz.FixedOffset(120)


def readRaven(msg: str) -> (pd.DataFrame, str):

    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title=msg, filetypes=[("csv files", "*.csv")], parent=None
    )

    data = pd.read_csv(file_path, sep=";")
    # data.columns = ["a", "b", "c", "etc."]

    return data, filename


RavenTable, filename = readRaven("Select Raven Selection Table")
RavenTable_clean = RavenTable.loc[np.isnan(RavenTable["Double_Check"])]


dataset_name = "CETIROISE_F1"
filename = RavenTable_clean["Begin File"]
end_time = [
    end - beg
    for end, beg in zip(
        RavenTable_clean["End Time (s)"], RavenTable_clean["Begin Time (s)"]
    )
]
nb_det = len(RavenTable_clean)
start_frequency = RavenTable_clean["Low Freq (Hz)"]
end_frequency = RavenTable_clean["High Freq (Hz)"]
species = "Whistle and Moan"
annotator = "PG double check"
start_datetime = pd.to_datetime(
    RavenTable_clean["Begin Date Time"], format="%Y/%m/%d %H:%M:%S.0000"
).tolist()
start_datetime = [s.tz_localize(tz) for s in start_datetime]
end_datetime = [s + pd.Timedelta(seconds=end_time[0]) for s in start_datetime]

start_datetime = [
    datetime.datetime.strftime(s, "%Y-%m-%dT%H:%M:%S.%f%z") for s in start_datetime
]
end_datetime = [
    datetime.datetime.strftime(s, "%Y-%m-%dT%H:%M:%S.%f%z") for s in end_datetime
]

data = {
    "dataset": [dataset_name] * nb_det,
    "filename": [""] * nb_det,
    "start_time": [0] * nb_det,
    "end_time": end_time,
    "start_frequency": start_frequency,
    "end_frequency": end_frequency,
    "annotation": [species] * nb_det,
    "annotator": annotator,
    "start_datetime": start_datetime,
    "end_datetime": end_datetime,
}
df_APLOSE = pd.DataFrame(data)
# ▒ Save the csv file in the same folder as the FPOD results csv file
df_APLOSE.to_csv(
    "/".join(filename.split("/")[0:-1]) + "/" + dataset_name + "_APLOSE_clean.csv",
    index=False,
)
