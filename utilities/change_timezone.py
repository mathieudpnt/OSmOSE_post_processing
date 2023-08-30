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
import datetime as dt
import pandas as pd
import numpy as np
import easygui
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
import seaborn as sns
from scipy import stats
import sys
import pytz
from collections import OrderedDict
from post_processing_detections.utilities.def_func import get_detection_files, extract_datetime, sorting_detections, t_rounder, get_timestamps, input_date

detection_file = get_detection_files(1)
df = pd.read_csv(detection_file[0])

start_dt = pd.to_datetime(df['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z').tolist()
end_dt = pd.to_datetime(df['end_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z').tolist()


start_dt_utc = [s.tz_convert(pytz.utc) for s in start_dt]
start_dt_utc_str = [s.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]+s.strftime('%z')[:-2]+':00' for s in start_dt_utc]

end_dt_utc = [s.tz_convert(pytz.utc) for s in end_dt]
end_dt_utc_str = [s.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]+s.strftime('%z')[:-2]+':00' for s in end_dt_utc]


# end_dt = df['end_datetime']
# b=end_dt[0]
# b_dt = dt.datetime.strptime(b, '%Y-%m-%dT%H:%M:%S.%f%z')
# #b_dt_utc = b.replace(tzinfo=pytz.utc)
# utc = pytz.timezone('UTC')
# b_dt_utc = utc.localize(b_dt)
# c=b_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]+b_dt.strftime('%z')[:-2]+':00'

#df_detections, t_detections = sorting_detections(detection_file)




