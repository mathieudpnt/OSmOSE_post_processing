# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:11:48 2023

@author: torterma

Convert time zone of the detections contained in an APLOSE format csv to UTC
   Parameters :
       detection_file: csv file containing the detections
       new_tz : new time zone

   Returns :
       csv file containing the detections in the new time zone format
"""

import pandas as pd
import pytz
from post_processing_detections.utilities.def_func import get_detection_files

# Read csv file
detection_file = get_detection_files(1)
df = pd.read_csv(detection_file[0])
# Read start and end time of each detection
start_dt = pd.to_datetime(
    df["start_datetime"], format="%Y-%m-%dT%H:%M:%S.%f%z"
).tolist()
end_dt = pd.to_datetime(df["end_datetime"], format="%Y-%m-%dT%H:%M:%S.%f%z").tolist()

# Convert timezone to UTC and rewrite time to good timeformat
start_dt_utc = [s.tz_convert(pytz.utc) for s in start_dt]
start_dt_utc_str = [
    s.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + s.strftime("%z")[:-2] + ":00"
    for s in start_dt_utc
]

end_dt_utc = [s.tz_convert(pytz.utc) for s in end_dt]
end_dt_utc_str = [
    s.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + s.strftime("%z")[:-2] + ":00"
    for s in end_dt_utc
]
# Replace new start and end time in the Dataframe
df["start_datetime"] = start_dt_utc_str
df["end_datetime"] = end_dt_utc_str

# Write the new UTC dataframe in csv

new_filename = detection_file[0][:-4] + "_UTC.csv"
df.to_csv(new_filename, index=None)
