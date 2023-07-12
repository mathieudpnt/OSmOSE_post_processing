from tqdm import tqdm
import datetime as dt
from tkinter import filedialog
from tkinter import Tk
import pandas as pd
import numpy as np
import easygui
from post_processing_detections.utilities.def_func import sorting_annot_boxes, t_rounder

def reshape_timebin(df_detection_path):
    """Changes the timebin (time resolution) of a detection file
    ex :    -from a raw PAMGuard detection file to a detection file with 10s timebin
            -from an 10s detection file to a 1min / 1h / 24h detection file

    Parameter:
        df_detection: the dataframe corresponding to the detection file

    Returns:
        another dataframe with the new timebin and writes it to a csv"""
    
    #LOAD DATA
    
    root = Tk()
    root.withdraw()
    detections_file = filedialog.askopenfilename(title="Select APLOSE formatted detection file", filetypes=[("CSV files", "*.csv")])

    t_detections = sorting_annot_boxes(detections_file)
    df_detections = t_detections[-1]
    timebin_orig = t_detections[0]
    fmax = t_detections[1]
    annotators = t_detections[2]
    labels = t_detections[3]
    tz_data = df_detections['start_datetime'][0].tz

    while True:
        timebin_new = easygui.buttonbox('Select a new time resolution for the detection file', 'Select new timebin', ['10s','1min', '10min', '1h', '24h'])
        if timebin_new == '10s':
            f= timebin_new
            timebin_new=10
        elif timebin_new == '1min':
            f= timebin_new
            timebin_new=60
        elif timebin_new == '10min':
            f= timebin_new
            timebin_new=600
        elif timebin_new == '1h':
            f= timebin_new
            timebin_new=3600
        elif timebin_new == '24h':
            f= timebin_new
            timebin_new=86400
        
        if timebin_new > timebin_orig: break
        else: easygui.msgbox('New time resolution is equal or smaller than the original one', 'Warning', 'Ok')
                    
    df_new = pd.DataFrame()
    for annotator in annotators:
        for label in labels:
            
            df_detect_prov = sorting_annot_boxes(file=detections_file, annotator = annotator, label = label)[-1]

            t = t_rounder(df_detect_prov['start_datetime'].iloc[0], timebin_new)
            t2 = t_rounder(df_detect_prov['start_datetime'].iloc[-1], timebin_new) + dt.timedelta(seconds=timebin_new)
            time_vector = [ts.timestamp() for ts in pd.date_range(start=t, end=t2, freq=f)]
            # time_vector_str = [ts.timestamp() for ts in pd.date_range(start=t, end=t2, freq=f)]
            
            
            
            times_detect_beg = [detect.timestamp() for detect in df_detect_prov['start_datetime']]
            times_detect_end = [detect.timestamp() for detect in df_detect_prov['end_datetime']]
                
            detect_vec, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
            for i in range(len(times_detect_beg)):
                for j in range(k, len(time_vector)-1):
                    if int(times_detect_beg[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)) or int(times_detect_end[i]*1000) in range(int(time_vector[j]*1000), int(time_vector[j+1]*1000)):
                            ranks.append(j)
                            k=j
                            break
                    else: 
                        continue 
            ranks = sorted(list(set(ranks)))
            detect_vec[ranks] = 1
            detect_vec = list(detect_vec)
               
            
            start_datetime_str, end_datetime_str, filename = [],[],[]
            for i in range(len(time_vector)):
                if detect_vec[i] == 1:
                    start_datetime = pd.Timestamp(time_vector[i], unit='s', tz=tz_data)
                    start_datetime_str.append(start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[:-8]+ start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-5:-2] +':' + start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-2:])
                    end_datetime = pd.Timestamp(time_vector[i]+timebin_new, unit='s', tz=tz_data)
                    end_datetime_str.append(end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[:-8]+ end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-5:-2] +':' + end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-2:])
                    filename.append(str(pd.Timestamp(time_vector[i], unit='s', tz=tz_data)))
            
            
            df_new_prov = pd.DataFrame()
            dataset_str = list(set(df_detect_prov['dataset']))
            
            df_new_prov['dataset'] = dataset_str*len(start_datetime_str)
            df_new_prov['filename'] = filename
            df_new_prov['start_time'] = [0]*len(start_datetime_str)
            df_new_prov['end_time'] = [timebin_new]*len(start_datetime_str)
            df_new_prov['start_frequency'] = [0]*len(start_datetime_str)
            df_new_prov['end_frequency'] = [fmax]*len(start_datetime_str)
            
            df_new_prov['annotation'] = list(set(df_detect_prov['annotation']))*len(start_datetime_str)
            df_new_prov['annotator'] = list(set(df_detect_prov['annotator']))*len(start_datetime_str)
              
            df_new_prov['start_datetime'], df_new_prov['end_datetime'] = start_datetime_str, end_datetime_str
    
            df_new = pd.concat([df_new, df_new_prov])
            
        df_new = df_new.sort_values(by=['start_datetime'])
            
    return df_new
    
    
    
