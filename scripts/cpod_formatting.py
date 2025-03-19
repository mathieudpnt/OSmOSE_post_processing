import pytz
import pandas as pd
from pathlib import Path

from fpod_utils import format_calendar, assign_phase_simple
from utils.fpod_utils import usable_data_phase, dpm_to_dph, assign_phase, generate_hourly_detections, merging_tab

from def_func import json2df

from premiers_resultats_utils import (
    load_parameters_from_yaml,
)

# %% 1 - Load parameters from the YAML file
yaml_file = Path(r".\scripts\yaml_example.yaml")

df_detections, time_bin, annotators, labels, fmax, datetime_begin, datetime_end, _ = (
    load_parameters_from_yaml(file=yaml_file)
)

json = Path(r"C:\Users\fouin\Downloads\deployment.json")
metadatax = json2df(json_path=json)

period = format_calendar(r"C:\Users\fouin\Downloads\Planning_records_deployment_Calais_Project.xlsx")

#%% 2 - Transform DPM per minute into DPM per h
tz = pytz.utc

df_detections['DPM'] = 1

dph = dpm_to_dph(df_detections, tz, 'Walde', "Marsouin", extra_columns=['DPM'])
#if needed, specify the columns you want to keep (extra_columns=['Name','of','your_columns'])

#%% 3a - If needed, reassign the name of your phases in your overall dataframe
df_period = assign_phase_simple(metadatax, period)
df_period = df_period.rename(columns={'start_datetime': 'deployment_date', 'end_datetime': 'recovery_date'})

#%% 3b -
df = assign_phase(df_period, dph, 'Walde')

#%% 4 - From the meta-dataframe (df_period), generate a table with one line per hour

meta_h = generate_hourly_detections(df_period, 'Walde')

#%% 5 - Merge "dph" and "meta_h" to obtain an R-compatible dataframe

data = merging_tab(meta_h, df)

#%% 6 - Export CSV

data.to_csv(r"path\to\folder\of_choice.csv", index=False)

####### To execute the code within the same campaign, you can start over from phase 3b #######



#%% Calculate the percentage of usable data in a phase
df1 = assign_phase(df_period, df_detections, 'Walde')

usable_data_phase(df_period, df1, 'Walde_Phase1')
#usable_data_phase(metadatax, df1, 'Ex : Walde_Phase4')