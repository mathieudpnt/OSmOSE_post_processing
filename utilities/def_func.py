import struct
from typing import Tuple, List, Union
import pytz
import pandas as pd
import re
import datetime as dt
import random
import numpy as np
from tqdm import tqdm
import os
from tkinter import filedialog
from tkinter import Tk
import gzip
import math
import easygui
import glob
import bisect
from astral.sun import sun
import astral
import csv
import yaml


def get_csv_file(num_files: int, message='Select csv') -> List[str]:
    '''Opens a file dialog multiple times
    to get X csv files (APLOSE formatted detection file, task status file...).
    Parameters :
        num_files: The number of detection files the user needs to select.
    Returns :
        List of file paths selected by the user.
    '''
    root = Tk()
    root.withdraw()

    file_paths = []
    for _ in range(num_files):
        file_path = filedialog.askopenfilename(
            title=message + f' ({len(file_paths) + 1}/{num_files})',
            filetypes=[('CSV files', '*.csv')],
            parent=None
        )
        if not file_path:
            break  # User cancelled or closed the file dialog
        file_paths.append(file_path)

    return file_paths


def sorting_detections(file: List[str], tz: pytz._FixedOffset = None, date_begin: dt.datetime = None, date_end: dt.datetime = None, annotator: str = None, annotation: str = None, box: bool = False, timebin_new: int = None, user_sel: str = 'all', fmin_filter: int = None, fmax_filter: int = None) -> (pd.DataFrame, pd.DataFrame):
    ''' Filters an Aplose formatted detection file according to user specified filters
        Parameters :
            file : list of path(s) to the detection file(s), can be a str too
            tz : timezone info, to be specified if the user wants to change the TZ of the detections
            date_begin : datetime to be specified if the user wants to select detections after date_begin
            date_end : datetime to be specified if the user wants to select detections before date_end
            annotator : string to be specified if the user wants to select the detection of a particular annotator
            annotation : string to be specified if the user wants to select the detection of a particular label
            box : if set to True, keeps all the annotations, if False keeps only the absence/presence box (weak detection)
            timebin_new : integer to be specified if the user already know the new time resolution to set the detection file to
            user_sel: string to specify to filter detections of a file based on annotators
                'union' : the common detections of all annotators and the unique detections of each annotators are selected
                'intersection' : only the common detections of all annotators are selected
                'all' : all the detections are selected, default value
            fmin_filer/fmax_filer : integer, in the case where the user wants to filter out detections based on their frequency range
        Returns :
            max_time : spectrogram temporal length
            max_freq : sampling frequency *0.5
            annotators : list of annotators after filtering
            labels : list of labels after filtering
            result_df : dataFrame corresponding to the filters applied and containing all the detections
            info : DataFrame containing infos such as max_time/max_freq/annotators/labels corresponding to each detection file
    '''

    # find the proper delimiter for file
    with open(file, 'r', newline='') as csv_file:
        try:
            temp_lines = csv_file.readline() + '\n' + csv_file.readline()
            dialect = csv.Sniffer().sniff(temp_lines, delimiters=',;')
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ','

    df = pd.read_csv(file, sep=delimiter)
    list_annotators = list(df['annotator'].drop_duplicates())
    list_labels = list(df['annotation'].drop_duplicates())
    max_freq = int(max(df['end_frequency']))
    max_time = int(max(df['end_time']))

    df['start_datetime'] = pd.to_datetime(df['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z')
    df['end_datetime'] = pd.to_datetime(df['end_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z')
    df = df.sort_values('start_datetime')

    if tz is not None:
        df['start_datetime'] = [x.tz_convert(tz) for x in df['start_datetime']]
        df['end_datetime'] = [x.tz_convert(tz) for x in df['end_datetime']]

    tz_data = df['start_datetime'][0].tz

    if date_begin is not None:
        df = df[df['start_datetime'] >= date_begin]

    if date_end is not None:
        df = df[df['end_datetime'] <= date_end]

    if date_begin is not None and date_end is not None:
        if date_begin >= date_end:
            raise ValueError("Error: date_begin > date_end")

    if annotator is not None:
        df = df.loc[(df['annotator'] == annotator)]
        list_annotators = [annotator]

    if annotation is not None:
        df = df.loc[(df['annotation'] == annotation)]
        list_labels = [annotation]

    if fmin_filter is not None:
        df = df[df['start_frequency'] >= fmin_filter]
        if len(df) == 0:
            raise Exception("No detection found after fmin filtering, upload aborted")

    if fmax_filter is not None:
        df = df[df['end_frequency'] <= fmax_filter]
        if len(df) == 0:
            raise Exception("No detection found after fmax filtering, upload aborted")

    df_nobox = df.loc[(df['start_time'] == 0) & (df['end_time'] == max_time) & (df['end_frequency'] == max_freq)]
    if len(df_nobox) == 0:
        max_time = 0

    if box is False:
        if len(df_nobox) == 0:
            df = reshape_timebin(df=df.reset_index(drop=True), timebin_new=timebin_new)
            max_time = int(max(df['end_time']))
        else:
            if timebin_new is not None:
                df = reshape_timebin(df=df.reset_index(drop=True), timebin_new=timebin_new)
                max_time = int(max(df['end_time']))
            else:
                df = df_nobox

    if len(list_annotators) > 1:
        if user_sel == 'union' or user_sel == 'intersection':
            df_inter = pd.DataFrame()
            df_diff = pd.DataFrame()
            for label_sel in list_labels:
                df_label = df[df['annotation'] == label_sel]
                values = list(df_label['start_datetime'].drop_duplicates())
                common_values = []
                diff_values = []
                error_values = []
                for value in values:
                    if df_label['start_datetime'].to_list().count(value) == 2:
                        common_values.append(value)
                    elif df_label['start_datetime'].to_list().count(value) == 1:
                        diff_values.append(value)
                    else:
                        error_values.append(value)

                df_label_inter = df_label[df_label['start_datetime'].isin(common_values)].reset_index(drop=True)
                df_label_inter = df_label_inter.drop_duplicates(subset='start_datetime')
                df_inter = pd.concat([df_inter, df_label_inter]).reset_index(drop=True)

                df_label_diff = df_label[df_label['start_datetime'].isin(diff_values)].reset_index(drop=True)
                df_diff = pd.concat([df_diff, df_label_diff]).reset_index(drop=True)

            if user_sel == 'intersection':
                df = df_inter
                list_annotators = [' ∩ '.join(list_annotators)]
            elif user_sel == 'union':
                df = pd.concat([df_diff, df_inter]).reset_index(drop=True)
                df = df.sort_values('start_datetime')
                list_annotators = [' ∪ '.join(list_annotators)]

            df['annotator'] = list_annotators[0]

    columns = ['file', 'max_time', 'max_freq', 'annotators', 'labels', 'tz_data']
    info = pd.DataFrame([[file, int(max_time), max_freq, list_annotators, list_labels, tz_data]], columns=columns)

    return df.reset_index(drop=True), info


def reshape_timebin(df: pd.DataFrame, timebin_new: int = None) -> pd.DataFrame:
    ''' Changes the timebin (time resolution) of a detection dataframe
    ex :    -from a raw PAMGuard detection file to a detection file with 10s timebin
            -from an 10s detection file to a 1min / 1h / 24h detection file
    Parameter:
        df : detection dataframe
        timebin_new : Time resolution to base the detections on, if not provided it is asked to the user
    Returns:
        df_new : detection dataframe with the new timebin
    '''
    if isinstance(df, pd.DataFrame) is False:
        raise Exception("Not a dataframe passed, reshape aborted")

    annotators = list(df['annotator'].drop_duplicates())
    labels = list(df['annotation'].drop_duplicates())

    df_nobox = df.loc[(df['start_time'] == 0) & (df['end_time'] == max(df['end_time'])) & (df['end_frequency'] == max(df['end_frequency']))]
    max_time = 0 if len(df_nobox) == 0 else int(max(df['end_time']))
    max_freq = int(max(df['end_frequency']))

    tz_data = df['start_datetime'][0].tz

    if timebin_new is None:
        while True:
            timebin_new = easygui.buttonbox('Select a new time resolution for the detections', df['dataset'][0], ['10s', '1min', '10min', '1h', '24h'])
            if timebin_new == '10s':
                f = timebin_new
                timebin_new = 10
            elif timebin_new == '1min':
                f = timebin_new
                timebin_new = 60
            elif timebin_new == '10min':
                f = timebin_new
                timebin_new = 600
            elif timebin_new == '1h':
                f = timebin_new
                timebin_new = 3600
            elif timebin_new == '24h':
                f = timebin_new
                timebin_new = 86400

            if timebin_new > max_time: break
            else: easygui.msgbox('New time resolution is equal or smaller than the original one', 'Warning', 'Ok')
    else: f = str(timebin_new) + 's'

    df_new = pd.DataFrame()
    if isinstance(annotators, str): annotators = [annotators]
    if isinstance(labels, str): labels = [labels]
    for annotator in annotators:
        for label in labels:

            df_detect_prov = df[(df['annotator'] == annotator) & (df['annotation'] == label)]

            if len(df_detect_prov) == 0:
                continue

            t = t_rounder(t=df_detect_prov['start_datetime'].iloc[0], res=timebin_new)
            t2 = t_rounder(df_detect_prov['start_datetime'].iloc[-1], timebin_new) + dt.timedelta(seconds=timebin_new)

            time_vector = [ts.timestamp() for ts in pd.date_range(start=t, end=t2, freq=f)]

            # #here test to find for each time vector value which filename corresponds
            filenames = sorted(list(set(df_detect_prov['filename'])))
            if not all(isinstance(filename, str) for filename in filenames):
                if all(math.isnan(filename) for filename in filenames):
                    # FPOD case: the filenames of a FPOD csv file are NaN values
                    filenames = [i.strftime('%Y-%m-%dT%H:%M:%S%z') for i in df_detect_prov['start_datetime']]

            ts_filenames = [extract_datetime(var=filename, tz=tz_data).timestamp()for filename in filenames]

            filename_vector = []
            for ts in time_vector:
                index = bisect.bisect_left(ts_filenames, ts)
                if index == 0:
                    filename_vector.append(filenames[index])
                elif index == len(ts_filenames):
                    filename_vector.append(filenames[index - 1])
                else:
                    filename_vector.append(filenames[index - 1])

            times_detect_beg = [detect.timestamp() for detect in df_detect_prov['start_datetime']]
            times_detect_end = [detect.timestamp() for detect in df_detect_prov['end_datetime']]

            detect_vec, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
            for i in range(len(times_detect_beg)):
                for j in range(k, len(time_vector) - 1):
                    if int(times_detect_beg[i] * 1e7) in range(int(time_vector[j] * 1e7), int(time_vector[j + 1] * 1e7)) or int(times_detect_end[i] * 1e7) in range(int(time_vector[j] * 1e7), int(time_vector[j + 1] * 1e7)):
                        ranks.append(j)
                        k = j
                        break
                    else:
                        continue

            ranks = sorted(list(set(ranks)))
            detect_vec[ranks] = 1
            detect_vec = list(detect_vec)

            start_datetime_str, end_datetime_str, filename = [], [], []
            for i in range(len(time_vector)):
                if detect_vec[i] == 1:
                    start_datetime = pd.Timestamp(time_vector[i], unit='s', tz=tz_data)
                    start_datetime_str.append(start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[:-8] + start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-5:-2] + ':' + start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-2:])
                    end_datetime = pd.Timestamp(time_vector[i] + timebin_new, unit='s', tz=tz_data)
                    end_datetime_str.append(end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[:-8] + end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-5:-2] + ':' + end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-2:])
                    filename.append(filename_vector[i])

            df_new_prov = pd.DataFrame()
            dataset_str = list(set(df_detect_prov['dataset']))

            df_new_prov['dataset'] = dataset_str * len(start_datetime_str)
            df_new_prov['filename'] = filename
            df_new_prov['start_time'] = [0] * len(start_datetime_str)
            df_new_prov['end_time'] = [timebin_new] * len(start_datetime_str)
            df_new_prov['start_frequency'] = [0] * len(start_datetime_str)
            df_new_prov['end_frequency'] = [max_freq] * len(start_datetime_str)
            df_new_prov['annotation'] = list(set(df_detect_prov['annotation'])) * len(start_datetime_str)
            df_new_prov['annotator'] = list(set(df_detect_prov['annotator'])) * len(start_datetime_str)
            df_new_prov['start_datetime'], df_new_prov['end_datetime'] = start_datetime_str, end_datetime_str

            df_new = pd.concat([df_new, df_new_prov])

        df_new['start_datetime'] = [pd.to_datetime(d, format='%Y-%m-%dT%H:%M:%S.%f%z') for d in df_new['start_datetime']]
        # df_new['start_datetime'] = pd.to_datetime(df_new['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z')
        df_new['end_datetime'] = [pd.to_datetime(d, format='%Y-%m-%dT%H:%M:%S.%f%z') for d in df_new['end_datetime']]
        # df_new['end_datetime'] = pd.to_datetime(df_new['end_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z')
        df_new = df_new.sort_values(by=['start_datetime'])

    return df_new


def read_param(file: str):
    ''' Reads parameters from a yaml file for importing detections from an APLOSE formatted csv file with sorting_detection
        Parameters :
            file : path to the yaml file, str
        Returns :
            arguments_list : list of dict containing a set of parameters for each csv file
    '''
    # TODO : Add warnings if the parameters are not properly formatted

    with open(file, 'r') as yaml_file:
        parameters = yaml.safe_load(yaml_file)

    arguments_list = []

    for param in parameters:
        argument = {'file': param['file']}

        if 'timebin_new' in param:
            argument['timebin_new'] = param['timebin_new']
        if 'tz' in param:
            offset_string = param['tz']
            hours, minutes = map(int, offset_string.lstrip('+').split(':'))
            total_offset_minutes = (hours * 60) + minutes
            argument['tz'] = pytz.FixedOffset(total_offset_minutes)
        if 'fmin_filter' in param:
            argument['fmin_filter'] = param['fmin_filter']
        if 'fmax_filter' in param:
            argument['fmax_filter'] = param['fmax_filter']
        if 'date_begin' in param:
            argument['date_begin'] = pd.Timestamp(param['date_begin'])
        if 'date_end' in param:
            argument['date_end'] = pd.Timestamp(param['date_end'])
        if 'annotator' in param:
            argument['annotator'] = param['annotator']
        if 'annotation' in param:
            argument['annotation'] = param['annotation']
        if 'box' in param:
            box_string = param['box']
            argument['box'] = box_string.lower() != 'false'
        if 'user_sel' in param:
            argument['user_sel'] = param['user_sel']

        arguments_list.append(argument)

    return arguments_list


def task_status_selection(files: List[str], df_detections: pd.DataFrame, user: Union[str, List[str]] = 'all') -> pd.DataFrame:
    ''' Filters a detection DataFrame to select only the segments that all annotator have completed (i.e. status == 'FINISHED')
        Parameters :
            file : list of path(s) to the status file(s), can be a str too
            df_detections : df of the detections (output of sorting_detections)
            user : string of list of strings, this argument is used to select the annotator to take into consideration
                - user='all', then all the annotators of the status file are used
                - user='annotator_name', then only one annotator is used
                - user=[list of annotators], then only the annotators present in the list are used
        Returns :
            df_kept : df of the detections sorted according to the selected annotators
    '''

    if isinstance(files, str):
        files = [files]  # Convert the single string to a list with one element

    result_df = pd.DataFrame()
    for file in files:

        # find the proper delimiter for file
        with open(file, 'r', newline='') as csv_file:
            try:
                temp_lines = csv_file.readline() + '\n' + csv_file.readline()
                dialect = csv.Sniffer().sniff(temp_lines, delimiters=',;')
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ','

        df = pd.read_csv(file, sep=delimiter)
        annotators_df = [i for i in list(df.columns) if i != 'dataset' and i != 'filename']

        # selection of the annotators according to the user argument
        if user == 'all':
            list_annotators = [i for i in list(df.columns) if i != 'dataset' and i != 'filename']
        elif isinstance(user, list):
            for u in user:
                if u not in annotators_df:
                    raise Exception(f"'{u}' not present in the task satuts file")
            list_annotators = user
        elif isinstance(user, str) and user != 'all':
            if user not in annotators_df:
                raise Exception(f"'{user}' not present in the task satuts file")
            list_annotators = [user]

        df_users = df_detections[df_detections['annotator'].isin(list_annotators)]

        filename_list = list(df[df[list_annotators].eq('FINISHED').all(axis=1)]['filename'])
        ignored_list = list(df[~df[list_annotators].eq('FINISHED').all(axis=1)]['filename'])

        df_kept = df_users[df_users['filename'].isin(filename_list)]
        # df_ignored = df_users[df_users['filename'].isin(ignored_list)]

        print(f'\n{os.path.basename(file)}: {len(ignored_list)} files ignored', end='\n')
        result_df = pd.concat([result_df, df_kept]).reset_index(drop=True)

    return result_df


# def read_header(file:str) -> Tuple[int, int, int, int, int]:
#     #reads header of a wav file to get info such as duration, samplerate etc...
#     with open(file, 'rb') as fh:
#        _, size, _ = struct.unpack('<4sI4s', fh.read(12))
#        chunk_header = fh.read(8)
#        subchunkid, _ = struct.unpack('<4sI', chunk_header)
#        if (subchunkid == b'fmt '):
#            _, channels, samplerate, _, _, sampwidth = struct.unpack('HHIIHH', fh.read(16))
#        sampwidth = (sampwidth + 7) // 8
#        framesize = channels * sampwidth
#        frames = size // framesize
#        return sampwidth, frames, samplerate, channels, frames/samplerate


def read_header(file: str) -> Tuple[int, int, int, int]:
    ''' Reads header of a wav file to get info such as duration, samplerate etc...
    Parameter :
        file : path to the wav file
    Returns :
        sampwidth
        frames
        samplerate
        channels
        frames/samplerate
    '''

    with open(file, 'rb') as fh:
        _, size, _ = struct.unpack('<4sI4s', fh.read(12))
        chunk_header = fh.read(8)
        subchunkid, _ = struct.unpack('<4sI', chunk_header)

        if (subchunkid == b'fmt '):
            _, channels, samplerate, _, _, sampwidth = struct.unpack('HHIIHH', fh.read(16))

        chunkOffset = fh.tell()
        found_data = False
        while (chunkOffset < size and not found_data):
            fh.seek(chunkOffset)
            subchunk2id, subchunk2size = struct.unpack('<4sI', fh.read(8))
            if (subchunk2id == b'data'):
                found_data = True

            chunkOffset = chunkOffset + subchunk2size + 8

        if not found_data:
            print('No data chunk found while reading the header. Will fallback on the header size.')
            subchunk2size = (size - 36)

        sampwidth = (sampwidth + 7) // 8
        framesize = channels * sampwidth
        frames = subchunk2size // framesize

        if (size - 36) != subchunk2size:
            print(f'Warning : the size indicated in the header is not the same as the actual file size. This might mean that the file is truncated or otherwise corrupt.\
                \nSupposed size: {size} bytes \nActual size: {subchunk2size} bytes.')

        return sampwidth, frames, samplerate, channels, frames / samplerate

# def get_wav_info(folder):
#     durations=[]
#     wav_files = glob.glob(os.path.join(folder, '**/*.wav'), recursive=True)
#     for file in tqdm(wav_files, 'Getting wav durations...', position=0, leave=True):
#         try:
#             with wave.open(file, 'r') as wav_files:
#                 frames = wav_files.getnframes()
#                 rate = wav_files.getframerate()
#                 durations.append(frames / float(rate))
#         except Exception as e:
#             print(f'An error occured while reading the file {file} : {e}')
#     return durations


def extract_datetime(var: str, tz: pytz._FixedOffset = None, formats=None) -> Union[dt.datetime, str]:
    ''' Extracts datetime from filename based on the date format
        Parameters :
            var : name of the wav file
            tz : timezone info
            formats : the date template in strftime format. For example, `2017/02/24` has the template `%Y/%m/%d`
                        For more information on strftime template, see https://strftime.org/
        Returns :
            date_obj : datetime corresponding to the datetime found in var
    '''

    if formats is None:
        # add more format if necessary
        formats = [r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}',
                   r'\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}',
                   r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}',
                   r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
                   r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
                   r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}',
                   r'\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2}',
                   r'\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}']
    match = None
    for f in formats:
        match = re.search(f, var)
        if match:
            break
    if match:
        dt_string = match.group()
        if f == r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}':
            dt_format = '%Y-%m-%dT%H-%M-%S'
        elif f == r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}':
            dt_format = '%Y-%m-%d_%H-%M-%S'
        elif f == r'\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}':
            dt_format = '%y%m%d%H%M%S'
        elif f == r'\d{2}\d{2}\d{2}_\d{2}\d{2}\d{2}':
            dt_format = '%y%m%d_%H%M%S'
        elif f == r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}':
            dt_format = '%Y-%m-%d %H:%M:%S'
        elif f == r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}':
            dt_format = '%Y-%m-%dT%H:%M:%S'
        elif f == r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}':
            dt_format = '%Y_%m_%d_%H_%M_%S'
        elif f == r'\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2}':
            dt_format = '%Y_%m_%dT%H_%M_%S'
        elif f == r'\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}':
            dt_format = '%Y%m%dT%H%M%S'

        date_obj = pd.to_datetime(dt_string, format=dt_format)

        if tz is None:
            return date_obj
        elif type(tz) is dt.timezone:
            offset_minutes = tz.utcoffset(None).total_seconds() / 60
            pytz_fixed_offset = pytz.FixedOffset(int(offset_minutes))
            date_obj = pytz_fixed_offset.localize(date_obj)
        else:
            date_obj = tz.localize(date_obj)

        return date_obj
    else:
        raise ValueError(f'{var}: No datetime found')


def t_rounder(t: dt.datetime, res: int):
    ''' Rounds a Timestamp according to the user specified resolution : 10s / 1min / 10 min / 1h / 24h
    Parameter :
        t: Timestamp to round
        res: integer corresponding to the new resolution in seconds
    Returns :
        t: rounded Timestamp
    '''

    if res == 600:  # 10min
        minute = t.minute
        minute = math.floor(minute / 10) * 10
        t = t.replace(minute=minute, second=0, microsecond=0)
    elif res == 10:  # 10s
        seconde = t.second
        seconde = math.floor(seconde / 10) * 10
        t = t.replace(second=seconde, microsecond=0)
    elif res == 60:  # 1min
        t = t.replace(second=0, microsecond=0)
    elif res == 3600:  # 1h
        t = t.replace(minute=0, second=0, microsecond=0)
    elif res == 86400:  # 24h
        t = t.replace(hour=0, minute=0, second=0, microsecond=0)
    elif res == 3:
        t = t.replace(microsecond=0)
    else:
        raise ValueError(f'res={res}s: Resolution not available')
    return t


def oneday_per_month(time_vector_ts, time_vector_str, vec) -> Tuple[list, list, list, list]:
    # select a random day for each months in input datetimes list and returns all the datetimes of those randomly selected days

    time_vector = [dt.datetime.fromtimestamp(time_vector_ts[i]) for i in range(len(time_vector_ts))]

    datetimes_by_month = {}
    for i in range(len(time_vector)):
        key = (time_vector[i].year, time_vector[i].month)
        if key not in datetimes_by_month:
            datetimes_by_month[key] = []
        datetimes_by_month[key].append((time_vector[i], vec[i], time_vector_str[i]))

    # randomly select one day for each month
    selected_datetimes = []
    selected_vec = []
    selected_str = []
    for dt_by_month in datetimes_by_month.values():
        if len(dt_by_month) > 0:
            month_days = list(set(list_dt[0].day for list_dt in dt_by_month))  # get all unique days in the month
            selected_day = random.choice(month_days)  # randomly select one day
            for i, PG, time_str in dt_by_month:
                if i.day == selected_day:
                    selected_datetimes.append(i)
                    selected_vec.append(PG)
                    selected_str.append(time_str)

    unique_dates = sorted(list(set(i.strftime('%d/%m/%Y') for i in selected_datetimes)), key=lambda x: dt.datetime.strptime(x, '%d/%m/%Y'))
    return [selected_datetimes[i].timestamp() for i in range(len(selected_datetimes))], [selected_vec[i] for i in range(len(selected_vec))], [selected_str[i] for i in range(len(selected_str))], unique_dates


def n_random_hour(time_vector_ts, time_vector_str, vec, n_hour: int, tz, time_step: int) -> Tuple[list, list, list, list]:
    ''' Randomly select n non-overlapping hours from the time vector
    Parameter :
        time_vector_ts : vector of timestamps
        time_vector_str : vector of strings corresponding to the timestamps
        vec: vector of 0/1 representing the absense/presence of a detection at the corresponding timestamp of the time_vector
        n_hour: number of hours to select
        tz : timezone object
        time_step: time bin of the time vector
    Returns :
        t: rounded Timestamp'''

    if type(tz) is not pytz._FixedOffset and tz is not pytz.utc: tz = pytz.timezone(tz)

    if not isinstance(n_hour, int):
        print('n_hour is not an integer')
        return

    selected_time_vector_ts, selected_dates = [], []
    while len(selected_dates) < n_hour:
        # choose a random datetime from the time vector
        rand_idx = random.randrange(len(time_vector_ts))
        rand_datetime = time_vector_ts[rand_idx]

        rand_datetime = np.round(rand_datetime / 3600) * 3600

        selected_dates.append(rand_datetime)

        # select all datetimes that fall within the hour following this datetime
        possible_datetimes = time_vector_ts[rand_idx:rand_idx + round(3600 / time_step) + 1]

        # check if any of the selected datetimes overlap with the previously selected datetimes
        overlap = False
        for i in selected_time_vector_ts:
            if any(i <= time < i + 3600 for time in possible_datetimes):
                overlap = True
                break

        if overlap: continue

        # add the selected datetimes to the list
        selected_time_vector_ts.extend(possible_datetimes)

        # sort the selected datetimes in chronological order
        selected_time_vector_ts.sort()
        selected_dates.sort()

    # extract the corresponding vectors and time strings
    selected_vec = [vec[time_vector_ts.index(i)] for i in tqdm(selected_time_vector_ts, position=0, leave=True)]
    selected_time_vector_str = [time_vector_str[time_vector_ts.index(i)] for i in tqdm(selected_time_vector_ts, position=0, leave=True)]
    selected_dates = [dt.datetime.fromtimestamp(i, tz).strftime('%d/%m/%Y %H:%M:%S') for i in selected_dates]

    return selected_time_vector_ts, selected_time_vector_str, selected_vec, selected_dates


def pick_datetimes(time_vector_ts, time_vector_str, vec, selected_dates, selected_durations, TZ) -> Tuple[list, list, list, list]:
    # user-selected datetimes from the time vector

    selected_df_out = pd.DataFrame({'datetimes': selected_dates, 'durations': selected_durations})

    # format the datetimes and durations from strings to datetimes/timedeltas
    # selected_dates = [dt.datetime.strptime(i, '%d/%m/%Y %H:%M:%S').timestamp() for i in selected_dates]
    selected_dates = [pd.to_datetime(i, format='%d/%m/%Y %H:%M:%S').timestamp() for i in selected_dates]
    timedeltas = []
    for i in selected_durations:
        if i.endswith('h'):
            timedeltas.append(dt.timedelta(hours=int(i[:-1])).total_seconds())
        elif i.endswith('m'):
            timedeltas.append(dt.timedelta(minutes=int(i[:-1])).total_seconds())
        elif i.endswith('s'):
            timedeltas.append(dt.timedelta(seconds=int(i[:-1])).total_seconds())
        elif i.endswith('d'):
            timedeltas.append(dt.timedelta(days=int(i[:-1])).total_seconds())
        else:
            print('incorrect duration format')
            return
    selected_durations = timedeltas

    selected_time_vector_ts = []

    for i in range(len(selected_dates)):
        # select all datetimes that fall within the durations following this datetime
        possible_datetimes = [time for time in time_vector_ts if selected_dates[i] <= time <= selected_dates[i] + selected_durations[i]]

        # add the selected datetimes to the list
        selected_time_vector_ts.extend(possible_datetimes)

    # sort the selected datetimes in chronological order
    selected_time_vector_ts.sort()

    # extract the corresponding vectors and time strings
    selected_vec = [vec[time_vector_ts.index(i)] for i in tqdm(selected_time_vector_ts, position=0, leave=True)]
    selected_time_vector_str = [time_vector_str[time_vector_ts.index(i)] for i in tqdm(selected_time_vector_ts, position=0, leave=True)]
    selected_dates = [dt.datetime.fromtimestamp(i, pytz.timezone(TZ)).strftime('%d/%m/%Y %H:%M:%S') for i in selected_dates]

    return selected_time_vector_ts, selected_time_vector_str, selected_vec, selected_df_out


def export2Raven(tuple_info, timestamps, df, timebin_new, bin_height, selection_vec: bool = False, offset: bool = False) -> pd.DataFrame:
    ''' Export a given vector to Raven formatted table
        Parameters :
            df : dataframe of the detections
            timebin_new : int, duration of the detection boxes to export, if set to 0, the original detections are exported
            bin_height : the maximum frequency of the exported timebins
            tuple_info : tuple containing info such as the filenames of the wav files, their durations and datetimes
            selection_vec : if it is set to False, all the timebins are exported, else the selection_vec is used to selec the wanted timebins to export, for instance it corresponds to all the positives timebins, containing detections
    '''

    file_list = list(tuple_info[0])
    file_datetimes = tuple_info[1]
    dur = list(tuple_info[2])

    offsets = [(file_datetimes[i] + dt.timedelta(seconds=dur[i])).timestamp() - (file_datetimes[i + 1]).timestamp() for i in range(len(file_datetimes) - 1)]
    offsets_cumsum = (list(np.cumsum([offsets[i] for i in range(len(offsets))])))
    offsets_cumsum.insert(0, 0)
    idx_wav_df = [file_list.index(df['filename'][i]) for i in range(len(df))]

    if timebin_new > 0:

        # time_vec = []
        # for i in range(len(file_list)):
        #     timestamp = file_datetimes[i].timestamp() + offsets_cumsum[i]
        #     durations = np.arange(0, dur[i], timebin_new).astype(int)

        #     for elem in durations:
        #         time_vec.append(timestamp + elem)
        time_vec = np.arange(t_rounder(file_datetimes[0], res=timebin_new).timestamp(), file_datetimes[-1].timestamp() + dur[-1], timebin_new).astype(int)

        if selection_vec is True:
            times_det_beg = [df['start_datetime'][i].timestamp() + offsets_cumsum[idx_wav_df[i]] + 1e-8 * timebin_new for i in range(len(df))]
            times_det_end = [df['end_datetime'][i].timestamp() + offsets_cumsum[idx_wav_df[i]] - 1e-8 * timebin_new for i in range(len(df))]

            det_vec, ranks, k = np.zeros(len(time_vec) - 1, dtype=int), [], 0
            for i in range(len(times_det_beg)):
                for j in range(k, len(time_vec) - 0):
                    if int(times_det_beg[i] * 1e8) in range(int(time_vec[j] * 1e8), int(time_vec[j + 1] * 1e8)) or int(times_det_end[i] * 1e8) in range(int(time_vec[j] * 1e7), int(time_vec[j + 1] * 1e7)):
                        ranks.append(j)
                        k = j
                        break
                    else: continue
            ranks = sorted(list(set(ranks)))
            det_vec[np.isin(range(len(time_vec) - 1), ranks)] = 1

        else:
            det_vec = [1] * (len(time_vec) - 1)

        start_time = [int(time_vec[i] - file_datetimes[0].timestamp()) for i in range(0, len(time_vec) - 1)]
        end_time = [int(time_vec[i] - file_datetimes[0].timestamp()) for i in range(1, len(time_vec))]
        delta = [end_time[i] - start_time[i] for i in range(len(start_time))]
        df_time = pd.DataFrame({'start': start_time, 'end': end_time, 'd': delta, 'vec': det_vec})
        df_time_sorted = df_time[(df_time['d'] == timebin_new) & (df_time['vec'] == 1)].reset_index(drop=True)

        df_PG2Raven = pd.DataFrame()
        df_PG2Raven['Selection'] = np.arange(1, len(df_time_sorted) + 1)
        df_PG2Raven['View'], df_PG2Raven['Channel'] = [1] * len(df_time_sorted), [1] * len(df_time_sorted)
        df_PG2Raven['Begin Time (s)'] = df_time_sorted['start']
        df_PG2Raven['End Time (s)'] = df_time_sorted['end']
        df_PG2Raven['Low Freq (Hz)'] = [0] * len(df_time_sorted)
        df_PG2Raven['High Freq (Hz)'] = [bin_height] * len(df_time_sorted)

    else:
        start_time = [df['start_time'][i] + offsets_cumsum[idx_wav_df[i]] for i in range(len(df))]
        end_time = [df['end_time'][i] + offsets_cumsum[idx_wav_df[i]] for i in range(len(df))]

        df_PG2Raven = pd.DataFrame()
        df_PG2Raven['Selection'] = np.arange(1, len(df) + 1)
        df_PG2Raven['View'], df_PG2Raven['Channel'] = [1] * len(df), [1] * len(df)
        df_PG2Raven['Begin Time (s)'] = start_time
        df_PG2Raven['End Time (s)'] = end_time
        df_PG2Raven['Low Freq (Hz)'] = df['start_frequency']
        df_PG2Raven['High Freq (Hz)'] = df['end_frequency']

    if offset is True:
        df_offset = pd.DataFrame({'filename': file_list, 'offset_cumsum': offsets_cumsum})
        return df_PG2Raven, df_offset
    else:
        return df_PG2Raven, None


def get_season(ts: dt.datetime) -> str:
    ''' 'day of year' ranges for the northern hemisphere
        Parameter :
            ts : datetime
        Returns :
            season : string corresponding to the season and year of the datetime (ex : if datetime is 01/01/2023, returns 'winter 2022')
    '''
    winter1 = range(1, 80)
    spring = range(80, 172)
    summer = range(172, 264)
    autumn = range(264, 355)
    winter2 = range(355, 367)

    if ts.dayofyear in spring: season = 'spring' + ' ' + str(ts.year)
    elif ts.dayofyear in summer: season = 'summer' + ' ' + str(ts.year)
    elif ts.dayofyear in autumn: season = 'autumn' + ' ' + str(ts.year)
    elif ts.dayofyear in winter1: season = 'winter' + ' ' + str(ts.year - 1)
    elif ts.dayofyear in winter2: season = 'winter' + ' ' + str(ts.year)

    return season


def load_glider_nav():
    ''' Load the navigation data from glider output files
        Parameter :
        Returns :
            df : dataframe with glider navigation data
    '''
    root = Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title='Select master folder')

    all_rows = []  # Initialize an empty list to store the contents of all CSV files
    yo = []  # List to store the file numbers
    file = []

    first_file = True
    file_number = 1  # Initialize the file number

    for filename in os.listdir(directory):
        if filename.endswith('.gz'):
            file_path = os.path.join(directory, filename)

            with gzip.open(file_path, 'rt') as gz_file:
                delimiter = ';'  # Specify the desired delimiter
                gz_reader = pd.read_csv(gz_file, delimiter=delimiter)
                # If it's the first file, append the header row
                if first_file:
                    all_rows.append(gz_reader.columns.tolist())
                    first_file = False
                # Add the rows from the current CSV file to the all_rows list
                all_rows.extend(gz_reader.values.tolist())
                # Add yo number to the yo list
                yo.extend([filename.split('.')[-2]] * len(gz_reader))
                file.extend([filename] * len(gz_reader))
                file_number += 1  # Increment the file number for the next file

    # Create a DataFrame from the combined data
    df = pd.DataFrame(all_rows)
    df.columns = df.iloc[0]  # set 1st row as headers
    df = df.iloc[1:, 0:-1]  # delete last column and 1st row

    # Add the yo number to the DataFrame
    df['yo'] = [int(x) for x in yo]

    df['file'] = file
    df = df.sort_values(by=['Timestamp'])
    df = df.drop(df[(df['Lat'] == 0) & (df['Lon'] == 0)].index).reset_index(drop=True)
    df['Lat DD'] = [int(x) + (((x - int(x)) / 60) * 100) for x in df['Lat'] / 100]
    df['Lon DD'] = [int(x) + (((x - int(x)) / 60) * 100) for x in df['Lon'] / 100]
    # df['Datetime'] = [dt.datetime.strptime(x, '%d/%m/%Y %H:%M:%S') for x in df['Timestamp']]
    df['Datetime'] = [pd.to_datetime(x, format='%d/%m/%Y %H:%M:%S') for x in df['Timestamp']]
    df['Depth'] = -df['Depth']

    return df


__converter = {
    '%Y': r'[12][0-9]{3}',
    '%y': r'[0-9]{2}',
    '%m': r'(0[1-9]|1[0-2])',
    '%d': r'([0-2][0-9]|3[0-1])',
    '%H': r'([0-1][0-9]|2[0-4])',
    '%I': r'(0[1-9]|1[0-2])',
    '%p': r'(AM|PM)',
    '%M': r'[0-5][0-9]',
    '%S': r'[0-5][0-9]',
    '%f': r'[0-9]{6}',
}


def convert_template_to_re(date_template: str) -> str:
    ''' Converts a template in strftime format to a matching regular expression
    Parameter :
        date_template: the template in strftime format
    Returns :
        The regular expression matching the template
    '''

    res = ''
    i = 0
    while i < len(date_template):
        if date_template[i: i + 2] in __converter:
            res += __converter[date_template[i: i + 2]]
            i += 1
        else:
            res += date_template[i]
        i += 1

    return res


def get_timestamps() -> pd.DataFrame:
    '''
    Read infos from APLOSE files timestamps.csv OR file_metadata.csv
    Parameters :
    Returns
        df_timestamps : TYPE
            DESCRIPTION.
    '''

    root = Tk()
    root.withdraw()
    timestampcsv_path = filedialog.askopenfilename(title='Select the timestamp.csv file', filetypes=[('CSV files', '*.csv')])
    root = Tk()
    root.withdraw()
    df_timestamps = pd.read_csv(timestampcsv_path)

    return df_timestamps


def find_files(f_type: str, ext: str, path: str = None, msg: str = None, n_dir: int = 1) -> list:
    ''' Based on selection_type, ask the user a folder and yields all the wav files inside it or ask the user multiple wav files
    Parameters :
        f_type : str, either 'dir' or 'file'
        ext : str, ex: 'wav'
        path : string, the user can specify the path of the askfolder dialog to open
        msg : string, the user can specify a message to display on the askfolder dialog
    Returns :
        selected_files : list of the paths of the wav files
    '''
    root = Tk()
    root.withdraw()

    # Define the file types to display in the dialog

    selected_files = []

    if f_type == 'dir':

        directory = []
        if path is None:
            for i in range(n_dir):
                directory.append(filedialog.askdirectory(initialdir=path, title='Select {0} folder {1}'.format(ext, i + 1)))

        else:
            directory = os.path.join(path, 'wav')

        if directory:
            [selected_files.extend(glob.glob(os.path.join(d, '**/*.{0}'.format(ext)), recursive=True)) for d in directory]

    elif f_type == 'file':
        # If the user wants to select multiple files, show the file dialog
        file_paths = filedialog.askopenfilenames(initialdir=path, title='Select {0} files {1}'.format(ext, msg), filetypes=[('{0} files'.format(ext), '*.{0}'.format(ext))])
        selected_files.extend(file_paths)

    return selected_files


def get_tz(file):
    '''Extract the tz from a detection file list
    if more than one tz is present UTC is chosen by default
    Parameters :
        file : list of APLOSE formatted detection files
    Returns:
        tz: pytz.tz object
    '''

    tz = []
    if isinstance(file, list):
        if len(file) == 1: [file] = file  # Convert the single string to a list with one element
        else:
            for i in file:
                dt = pd.to_datetime(pd.read_csv(i, usecols=['start_datetime'])['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z').tolist()
                tz.append(list(set([x.tz for x in dt]))[0])
            tz = list(set(tz))
            if len(tz) == 1: tz = tz[0]
            else:
                print('More than one tz present on detection files, UTC is selected')
                tz = pytz.UTC
        return tz
    else:
        dt = pd.to_datetime(pd.read_csv(file, usecols=['start_datetime'])['start_datetime'].tolist(), format='%Y-%m-%dT%H:%M:%S.%f%z')
        tz = list(set([x.tz for x in dt]))

        if len(tz) == 1:
            return tz[0]
        elif len(tz) > 1:
            print('More than one timezone present in file')
        else: print('error tz')


# def input_date(msg):

#     title = 'Date'
#     fieldNames = ['Year [YYYY]', 'Month [m]', 'Day [d]', 'Hour [H]', 'Minute [M]', 'Second [S]', 'Timezone [+/-HHMM]']
#     fieldValues = []  # we start with blanks for the values
#     fieldValues = easygui.multenterbox(msg, title, fieldNames)

#     # make sure that none of the fields was left blank
#     while 1:
#         if fieldValues is None: break
#         errmsg = ''
#         for i in range(len(fieldNames)):
#             if fieldValues[i].strip() == '':
#                 errmsg = errmsg + ("'%s' is a required field.\n\n" % fieldNames[i])
#         if errmsg == '': break  # no problems found
#         fieldValues = easygui.multenterbox(errmsg, title, fieldNames, fieldValues)
#     print('Reply was:', fieldValues)

#     hours_offset = int(fieldValues[-1][:3])
#     minutes_offset = int(fieldValues[-1][3:])
#     tz = pytz.FixedOffset(hours_offset * 60 + minutes_offset)

#     date_dt = dt.datetime(*map(int, fieldValues[:-1]), 0, tz)

#     return date_dt

def input_date(msg):
    ''' Based on selection_type, ask the user a folder and yields all the wav files inside it or ask the user multiple wav files
        Parameters :
            msg : Message to tell the user what date they have to enter (begin, end...)
        Returns :
            date_dt : aware datetime entered by the user
    '''

    title = 'Date'
    fieldNames = ['Year [YYYY]', 'Month [m]', 'Day [d]', 'Hour [H]', 'Minute [M]', 'Second [S]', 'Timezone [+/-HHMM]']
    fieldValues = []  # Initialize with empty values

    while True:
        fieldValues = easygui.multenterbox(msg, title, fieldNames, fieldValues)

        if fieldValues is None:
            # User canceled the input
            return None

        errmsg = ''
        for i in range(len(fieldNames)):
            if fieldValues[i].strip() == '':
                errmsg += f"'{fieldNames[i]}' is a required field.\n"

        if errmsg == '':
            break  # No validation errors

        easygui.msgbox(errmsg, title)

    year, month, day, hour, minute, second = map(int, fieldValues[:-1])
    hours_offset = int(fieldValues[-1][:3])
    minutes_offset = int(fieldValues[-1][3:])
    tz = pytz.FixedOffset(hours_offset * 60 + minutes_offset)

    date_dt = dt.datetime(year, month, day, hour, minute, second, tzinfo=tz)
    return date_dt


def suntime_hour(begin_deploy, end_deploy, timeZ, lat, lon):
    """ Fetch sunrise and sunset hours for dates between date_beg and date_end
    Parameters :
        date_beg : str Date in format 'YYYY-mm-dd'. Start date of when to fetch sun hour
        date_end : str Date in format 'YYYY-mm-dd'. End date of when to fetch sun hour
        timeZ : tz_data, FixedOffset object of pytz module
        lat : str latitude in Decimal Degrees
        lon : str longitude in Decimal Degrees
    Returns :
        hour_sunrise : list of float with sunrise decimal hours for each day between date_beg and date_end
        hour_sunset : list of float with sunset decimal hours for each day between date_beg and date_end
    """

    # Infos sur la localisation
    gps = astral.LocationInfo(timezone=timeZ, latitude=lat, longitude=lon)
    # List of days during when the data were recorded
    list_time = pd.date_range(begin_deploy.replace(tzinfo=pytz.UTC), end_deploy.replace(tzinfo=pytz.UTC))
    h_sunrise = []
    h_sunset = []
    dt_dusk = []
    dt_dawn = []
    dt_day = []
    dt_night = []
    astral.Depression = 12  # nautical twilight see def here : https://www.timeanddate.com/astronomy/nautical-twilight.html

    # For each day : find time of sunset, sun rise, begin dawn and dusk
    for day in list_time:
        suntime = sun(gps.observer, date=day, dawn_dusk_depression=astral.Depression)
        #suntime = sun(gps.observer, date=day)
        
        
        
        dawn_dt=((pd.to_datetime(suntime['dawn'])).tz_convert(timeZ)).to_pydatetime()
        dusk_dt=((pd.to_datetime(suntime['dusk'])).tz_convert(timeZ)).to_pydatetime()
        day_dt = ((pd.to_datetime(suntime['sunrise'])).tz_convert(timeZ)).to_pydatetime()
        night_dt = ((pd.to_datetime(suntime['sunset'])).tz_convert(timeZ)).to_pydatetime()

        day_hour = day_dt.hour + day_dt.minute / 60
        night_hour = night_dt.hour + night_dt.minute / 60
        h_sunrise.append(day_hour)
        h_sunset.append(night_hour)
        dt_dusk.append(dusk_dt)
        dt_dawn.append(dawn_dt)
        dt_day.append(day_dt)
        dt_night.append(night_dt)
    return h_sunrise[0:-1], h_sunset[0:-1], dt_dusk, dt_dawn, dt_day, dt_night


def stats_diel_pattern(df_detections: pd.DataFrame, begin_date: dt.datetime, end_date: dt.datetime, lat: float = None, lon: float = None):
    """ Plot detection proportions for each light regime (night/dawn/day/dawn)
    Parameters :
        begin_date : begin datetime of data to analyse
        end_date : end datetime of data to analyse
        lat : float latitude in Decimal Degrees
        lon : float longitude in Decimal Degrees
    Returns :
        lr : df used to plot the detections
        BoxName : list of light regimes
    """

    tz_data = df_detections['start_datetime'][0].tz

    if not isinstance(lat, float) and not isinstance(lat, int) and lat is not None:
        raise ValueError('Invalid latitude')
    elif not isinstance(lon, float) and not isinstance(lon, int) and lon is not None:
        raise ValueError('Invalid longitude')
    elif lat is None or lon is None:
        # User input : gps coordinates in Decimal Degrees
        title = "Coordinates in degree° minute' "
        msg = "Latitudes (N/S) and longitudes (E/W)"
        fieldNames = ["Lat Decimal Degree", "Lon Decimal Degree"]
        fieldValues = easygui.multenterbox(msg, title, fieldNames)

        # make sure that none of the fields was left blank
        while 1:
            if fieldValues is None: break
            errmsg = ""
            for i in range(len(fieldNames)):
                value = fieldValues[i]
                if not value.strip():
                    errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
                elif not isinstance(value, float) and not isinstance(value, int):
                    errmsg = errmsg + ('"%s" must be a valid number.\n\n' % fieldNames[i])
            if errmsg == "": break  # no problems found
            fieldValues = easygui.multpasswordbox(errmsg, title, fieldNames, fieldValues)
            print("Reply was:", fieldValues)

            lat = fieldValues[0]
            lon = fieldValues[1]

    # Compute sunrise and sunset decimal hour at the dataset location
    # Seems to only work with UTC data ?
    [_, _, dt_dusk, dt_dawn, dt_day, dt_night] = suntime_hour(begin_date, end_date, tz_data, lat, lon)

    # List of days in the dataset
    list_days = [dt.date(d.year, d.month, d.day) for d in dt_day]
    # Compute dusk_duration, dawn_duration, day_duration, night_duration
    dawn_duration = [b - a for a, b in zip(dt_dawn, dt_day)]
    day_duration = [b - a for a, b in zip(dt_day, dt_night)]
    dusk_duration = [b - a for a, b in zip(dt_night, dt_dusk)]
    night_duration = [dt.timedelta(hours=24) - dawn - day - dusk for dawn, day, dusk in zip(dawn_duration, day_duration, dusk_duration)]
    # Convert to decimal
    dawn_duration_dec = [dawn_d.total_seconds() / 3600 for dawn_d in dawn_duration]
    day_duration_dec = [day_d.total_seconds() / 3600 for day_d in day_duration]
    dusk_duration_dec = [dusk_d.total_seconds() / 3600 for dusk_d in dusk_duration]
    night_duration_dec = [night_d.total_seconds() / 3600 for night_d in night_duration]

    # Assign a light regime to each detection
    # : 1 = night ; 2 = dawn ; 3 = day ; 4 = dusk
    day_det = [start_datetime.date() for start_datetime in df_detections['start_datetime']]
    light_regime = []
    for idx_day, day in enumerate(list_days):
        for idx_det, d in enumerate(day_det):
            # If the detection occured during 'day'
            if d == day:
                if df_detections['start_datetime'][idx_det] > dt_dawn[idx_day] and df_detections['start_datetime'][idx_det] < dt_day[idx_day]:
                    lr = 2
                    light_regime.append(lr)
                elif df_detections['start_datetime'][idx_det] > dt_day[idx_day] and df_detections['start_datetime'][idx_det] < dt_night[idx_day]:
                    lr = 3
                    light_regime.append(lr)
                elif df_detections['start_datetime'][idx_det] > dt_night[idx_day] and df_detections['start_datetime'][idx_det] < dt_dusk[idx_day]:
                    lr = 4
                    light_regime.append(lr)
                else:
                    lr = 1
                    light_regime.append(lr)

    # For each day, count the number of detection per light regime
    nb_det_night = []
    nb_det_dawn = []
    nb_det_day = []
    nb_det_dusk = []
    for idx_day, day in enumerate(list_days):
        # Find index of detections that occured during 'day'
        idx_det = [idx for idx, det in enumerate(day_det) if det == day]
        if idx_det == []:
            lr = 0
            nb_det_night.append(lr)
            nb_det_dawn.append(lr)
            nb_det_day.append(lr)
            nb_det_dusk.append(lr)
        else:
            nb_det_night.append(light_regime[idx_det[0]:idx_det[-1]].count(1))
            nb_det_dawn.append(light_regime[idx_det[0]:idx_det[-1]].count(2))
            nb_det_day.append(light_regime[idx_det[0]:idx_det[-1]].count(3))
            nb_det_dusk.append(light_regime[idx_det[0]:idx_det[-1]].count(4))

    # For each day :  compute number of detection per light regime corrected by ligh regime duration
    nb_det_night_corr = [(nb / d) for nb, d in zip(nb_det_night, night_duration_dec)]
    nb_det_dawn_corr = [(nb / d) for nb, d in zip(nb_det_dawn, dawn_duration_dec)]
    nb_det_day_corr = [(nb / d) for nb, d in zip(nb_det_day, day_duration_dec)]
    nb_det_dusk_corr = [(nb / d) for nb, d in zip(nb_det_dusk, dusk_duration_dec)]

    # Normalize by daily average number of detection per hour
    av_daily_nbdet = []
    nb_det_night_corr_norm = []
    nb_det_dawn_corr_norm = []
    nb_det_day_corr_norm = []
    nb_det_dusk_corr_norm = []

    for idx_day, day in enumerate(list_days):
        # Find index of detections that occured during 'day'
        idx_det = [idx for idx, det in enumerate(day_det) if det == day]
        # Compute daily average number of detections per hour
        a = len(idx_det) / 24
        av_daily_nbdet.append(a)
        if a == 0:
            nb_det_night_corr_norm.append(0)
            nb_det_dawn_corr_norm.append(0)
            nb_det_day_corr_norm.append(0)
            nb_det_dusk_corr_norm.append(0)
        else:
            nb_det_night_corr_norm.append(nb_det_night_corr[idx_day] - a)
            nb_det_dawn_corr_norm.append(nb_det_dawn_corr[idx_day] - a)
            nb_det_day_corr_norm.append(nb_det_day_corr[idx_day] - a)
            nb_det_dusk_corr_norm.append(nb_det_dusk_corr[idx_day] - a)

    LIGHTR = [nb_det_night_corr_norm, nb_det_dawn_corr_norm, nb_det_day_corr_norm, nb_det_dusk_corr_norm]
    BoxName = ['Night', 'Dawn', 'Day', 'Dusk']

    lr = pd.DataFrame(LIGHTR, index=BoxName).transpose()

    return lr, BoxName


def stat_box_day(data_test: pd.DataFrame, df_detections: pd.DataFrame) -> pd.DataFrame:
    """ Plot detection proportions for each hour of the day
    Parameters :
        data_test : df with data infos
        df_detections : APLOSE formatted df of the detections
    Returns :
        result : df used to plot the detections
    """

    hour_list = ['{:02d}:00'.format(i) for i in range(24)]
    hour_list.append('00:00')

    df_detections['date'] = [dt.datetime.strftime(i.date(), '%d/%m/%Y') for i in df_detections['start_datetime']]
    df_detections['season'] = [get_season(i) for i in df_detections['start_datetime']]
    df_detections['dataset'] = [i.replace('_', ' ') for i in df_detections['dataset']]

    vec1 = [[data_test['beg_deployment'][i]] * len(data_test['df_detections'][i]) for i in data_test.index]
    vec2 = [[data_test['end_deployment'][i]] * len(data_test['df_detections'][i]) for i in data_test.index]
    start_deploy, end_deploy = [], []
    [start_deploy.extend(inner_list) for inner_list in vec1]
    [end_deploy.extend(inner_list) for inner_list in vec2]
    df_detections['start_deploy'] = [pd.to_datetime(d) for d in start_deploy]
    df_detections['end_deploy'] = [pd.to_datetime(d) for d in end_deploy]

    result = {}
    list_dates = sorted(list(set(df_detections['date'])))  # list of dates
    for date in list_dates:
        detection_bydate = df_detections[df_detections['date'] == date]  # sub-dataframe : per date
        list_datasets = sorted(list(set(detection_bydate['dataset'])))  # dataset list for date=date

        for dataset in list_datasets:
            df = detection_bydate[detection_bydate['dataset'] == dataset].set_index('start_datetime')  # sub-dataframe : per date & per dataset

            # number of detections per hour of the day at date and at dataset
            detection_per_dataset = [len(df.between_time(hour_list[j], hour_list[j + 1], inclusive='left')) for j in (range(len(hour_list) - 1))]

            deploy_beg_ts, deploy_end_ts = int(df['start_deploy'][0].timestamp()), int(df['end_deploy'][0].timestamp())

            list_present_h = [dt.datetime.fromtimestamp(i) for i in list(range(deploy_beg_ts, deploy_end_ts, 3600))]
            list_present_h2 = [dt.datetime.strftime(list_present_h[i], '%d/%m/%Y %H') for i in range(len(list_present_h))]

            list_deploy_d = sorted(list(set([dt.datetime.strftime(dt.datetime.fromtimestamp(i), '%d/%m/%Y') for i in list(range(deploy_beg_ts, deploy_end_ts, 3600))])))
            list_deploy_d2 = [d for i, d in enumerate(list_deploy_d) if d in date][0]

            list_present_h3 = []
            for item in list_present_h2:
                if item.startswith(list_deploy_d2):
                    list_present_h3.append(item)

            list_deploy = [df['date'][0] + ' ' + n for n in [f'{i:02}' for i in range(0, 24)]]

            for i, h in enumerate(list_deploy):
                if h not in list_present_h3:
                    detection_per_dataset[i] = np.nan

            result[dataset, date] = detection_per_dataset

    return pd.DataFrame(result).T
