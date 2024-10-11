import struct
import pytz
import pandas as pd
import re
import datetime as dt
import numpy as np
import math
import easygui
import bisect
from astral.sun import sun
import astral
import csv
import yaml
from pathlib import Path
import matplotlib.dates as mdates
from pandas.tseries.frequencies import to_offset



def reshape_timebin2(
    df: pd.DataFrame,
    timebin_new: int = None,
    timestamp: str = None,
) -> pd.DataFrame:

    df = df.sort_values("start_datetime").reset_index(drop=True)
    annotators = df["annotator"].drop_duplicates().to_list()
    labels = df["annotation"].drop_duplicates().to_list()

    df_nobox = df.loc[
        (df["start_time"] == 0)
        & (df["end_time"] == max(df["end_time"]))
        & (df["end_frequency"] == max(df["end_frequency"]))
    ]

    max_time = 1 if len(df_nobox) == 0 else int(max(df["end_time"]))
    max_freq = int(max(df["end_frequency"]))

    tz_data = pytz.FixedOffset(
        df["start_datetime"][0].utcoffset().total_seconds() // 60
    )

    if timebin_new is None:

        while True:
            timebin_new = easygui.buttonbox(
                "Select a new time resolution for the detections",
                df["dataset"][0],
                ["10s", "1min", "10min", "1h", "24h"],
            )
            if timebin_new == "10s":
                f = timebin_new
                timebin_new = 10
            elif timebin_new == "1min":
                f = timebin_new
                timebin_new = 60
            elif timebin_new == "10min":
                f = timebin_new
                timebin_new = 600
            elif timebin_new == "1h":
                f = timebin_new
                timebin_new = 3600
            elif timebin_new == "24h":
                f = timebin_new
                timebin_new = 86400

            if timebin_new > max_time:
                break
            else:
                easygui.msgbox(
                    "New time resolution is equal or smaller than the original one",
                    "Warning",
                    "Ok",
                )
    else:
        f = str(timebin_new) + "s"

    if isinstance(annotators, str):
        annotators = [annotators]
    if isinstance(labels, str):
        labels = [labels]

    df_new = pd.DataFrame()
    for annotator in annotators:
        for label in labels:

            df_detect_prov = df[
                (df["annotator"] == annotator) & (df["annotation"] == label)
            ]

            if len(df_detect_prov) == 0:
                continue
            
            
            if timestamp:
                # timestamp_csv = pd.read_csv(timestamp_file, parse_dates=['timestamp'])
                # timestamp_range = timestamp_csv['timestamp'].to_list()
                
                origin_timebin = (timestamp[1] - timestamp[0]).total_seconds()
                time_vector = [
                    ts.timestamp()
                    for ts in timestamp[0 :: int(timebin_new / origin_timebin)]
                ]
            else:
                t = t_rounder(t=df_detect_prov['start_datetime'].iloc[0], res=timebin_new)
                t2 = t_rounder(df_detect_prov['start_datetime'].iloc[-1], timebin_new) + pd.Timedelta(seconds=timebin_new)
                time_vector = [ts.timestamp() for ts in pd.date_range(start=t, end=t2, freq=f)]
                
            # test to find for each time vector value which filename corresponds
            filenames = sorted(list(set(df_detect_prov["filename"])))
            if not all(isinstance(filename, str) for filename in filenames):
                if all(math.isnan(filename) for filename in filenames):
                    # FPOD case: the filenames of a FPOD csv file are NaN values
                    filenames = [
                        i.strftime("%Y-%m-%dT%H:%M:%S%z")
                        for i in df_detect_prov["start_datetime"]
                    ]

            ts_filenames = [
                extract_datetime(var=filename, tz=tz_data).timestamp()
                for filename in filenames
            ]

            filename_vector = []
            for ts in time_vector:
                index = bisect.bisect_left(ts_filenames, ts)
                if index == 0:
                    filename_vector.append(filenames[index])
                elif index == len(ts_filenames):
                    filename_vector.append(filenames[index - 1])
                else:
                    # filename_vector.append(filenames[index - 1]) # pb sur cette ligne, à creuser
                    filename_vector.append(filenames[index])

            times_detect_beg = [
                detect.timestamp() for detect in df_detect_prov["start_datetime"]
            ]
            times_detect_end = [
                detect.timestamp() for detect in df_detect_prov["end_datetime"]
            ]

            detect_vec, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
            for i in range(len(times_detect_beg)):
                for j in range(k, len(time_vector) - 1):
                    if (
                        times_detect_beg[i] >= time_vector[j]
                        and times_detect_beg[i] < time_vector[j + 1]
                    ) or (
                        times_detect_end[i] > time_vector[j]
                        and times_detect_end[i] <= time_vector[j + 1]
                    ):
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
                    start_datetime = pd.Timestamp(time_vector[i], unit="s", tz=tz_data)
                    start_datetime_str.append(
                        start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f%z")[:-8]
                        + start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f%z")[-5:-2]
                        + ":"
                        + start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f%z")[-2:]
                    )
                    end_datetime = pd.Timestamp(
                        time_vector[i] + timebin_new, unit="s", tz=tz_data
                    )
                    end_datetime_str.append(
                        end_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f%z")[:-8]
                        + end_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f%z")[-5:-2]
                        + ":"
                        + end_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f%z")[-2:]
                    )
                    filename.append(filename_vector[i])

            df_new_prov = pd.DataFrame()
            dataset_str = list(set(df_detect_prov["dataset"]))

            df_new_prov["dataset"] = dataset_str * len(start_datetime_str)
            df_new_prov["filename"] = filename
            df_new_prov["start_time"] = [0] * len(start_datetime_str)
            df_new_prov["end_time"] = [timebin_new] * len(start_datetime_str)
            df_new_prov["start_frequency"] = [0] * len(start_datetime_str)
            df_new_prov["end_frequency"] = [max_freq] * len(start_datetime_str)
            df_new_prov["annotation"] = list(set(df_detect_prov["annotation"])) * len(
                start_datetime_str
            )
            df_new_prov["annotator"] = list(set(df_detect_prov["annotator"])) * len(
                start_datetime_str
            )
            df_new_prov["start_datetime"], df_new_prov["end_datetime"] = (
                start_datetime_str,
                end_datetime_str,
            )

            df_new = pd.concat([df_new, df_new_prov])

        df_new["start_datetime"] = [
            pd.to_datetime(d, format="%Y-%m-%dT%H:%M:%S.%f%z")
            for d in df_new["start_datetime"]
        ]
        df_new["end_datetime"] = [
            pd.to_datetime(d, format="%Y-%m-%dT%H:%M:%S.%f%z")
            for d in df_new["end_datetime"]
        ]
        df_new = df_new.sort_values(by=["start_datetime"])

    return df_new


def sort_detections(
    file: Path,
    timebin_new: int = None,
    tz: pytz._FixedOffset = None,
    datetime_begin: dt.datetime = None,
    datetime_end: dt.datetime = None,
    annotator: str = None,
    annotation: str = None,
    box: bool = False,
    timestamp_file: str = None,
    user_sel: str = "all",
    fmin_filter: int = None,
    fmax_filter: int = None,
) -> pd.DataFrame:
    """Filters an Aplose formatted detection file according to user specified filters

    Parameters
    ----------
        file : Path to the detection file
        timebin_new : new time resolution to set the detection file to (in seconds)
        tz : timezone info, timezone object to change the TZ of the detections
        datetime_begin : datetime to be specified if the user wants to select detections after date_begin
        datetime_end : datetime to be specified if the user wants to select detections before date_end
        annotator : string to be specified if the user wants to select the detection of a particular annotator
        annotation : string to be specified if the user wants to select the detection of a particular label
        box : if set to True, keeps all the annotations, if False keeps only the absence/presence box (weak detection)
        timestamp_file : path to the an APLOSE formatted timestamp file.
                        It is used to create a reshaped detection file with timestamps that matches the APLOSE annotations.
        user_sel: string to specify to filter detections of a file based on annotators
            'union' : the common detections of all annotators and the unique detections of each annotators are selected
            'intersection' : only the common detections of all annotators are selected
            'all' : all the detections are selected, default value
        fmin_filter : integer to filter out detections based on a minimum frequency
        fmax_filter : integer to filter out detections based on a maximum frequency

    Returns
    -------
        result_df : dataFrame corresponding to the filters applied and containing all the detections
    """

    delimiter = find_delimiter(file)

    df = pd.read_csv(
        file, sep=delimiter, parse_dates=["start_datetime", "end_datetime"]
    ).sort_values("start_datetime").reset_index(drop=True)

    df = df.dropna(subset=["annotation"])  # drop lines with only comments
    
    list_annotators = df["annotator"].drop_duplicates().to_list()
    list_labels = df["annotation"].drop_duplicates().to_list()
    max_freq = int(max(df["end_frequency"]))
    max_time = int(max(df["end_time"]))

    if tz:
        df["start_datetime"] = [x.tz_convert(tz) for x in df["start_datetime"]]
        df["end_datetime"] = [x.tz_convert(tz) for x in df["end_datetime"]]

    tz_data = pytz.FixedOffset(
        df["start_datetime"][0].utcoffset().total_seconds() // 60
    )
            
    if datetime_begin:
        df = df[df["start_datetime"] >= datetime_begin]
        if len(df) == 0:
            raise Exception(f"No detection found after 'datetime_begin' filtering at '{datetime_begin}', upload aborted")

    if datetime_end:
        df = df[df["end_datetime"] <= datetime_end]
        if len(df) == 0:
            raise Exception(f"No detection found after 'datetime_end' filtering at '{datetime_end}', upload aborted")

    if annotator:
        if annotator not in list_annotators:
            raise ValueError(f"Annotator '{annotator}' is not present in result file annotators, upload aborted")
        df = df.loc[(df["annotator"] == annotator)]
        list_annotators = [annotator]

    if annotation:
        if annotation not in list_labels:
            raise ValueError(f"Annotation '{annotation}' is not present in result file labels, upload aborted")
        df = df.loc[(df["annotation"] == annotation)]
        list_labels = [annotation]

    if fmin_filter:
        df = df[df["start_frequency"] >= fmin_filter]
        if len(df) == 0:
            raise Exception(f"No detection found after fmin filtering at {fmin_filter}Hz, upload aborted")

    if fmax_filter:
        df = df[df["end_frequency"] <= fmax_filter]
        if len(df) == 0:
            raise Exception(f"No detection found after fmax filtering at {fmax_filter}Hz, upload aborted")

    df_nobox = df.loc[
        (df["start_time"] == 0)
        & (df["end_time"] == max_time)
        & (df["end_frequency"] == max_freq)
    ]
    
    if len(df_nobox) == 0:
        max_time = 1

    if box is False:
        if len(df_nobox) == 0:
            df = reshape_timebin2(
                df=df.reset_index(drop=True),
                timebin_new=timebin_new,
                timestamp=timestamp_file,
            )
            max_time = int(max(df["end_time"]))
        else:
            if timebin_new is not None:
                df = reshape_timebin2(
                    df=df.reset_index(drop=True),
                    timebin_new=timebin_new,
                    timestamp=timestamp_file,
                )
                max_time = int(max(df["end_time"]))
            else:
                df = df_nobox

    if len(list_annotators) > 1:
        if user_sel == "union" or user_sel == "intersection":
            df_inter = pd.DataFrame()
            df_diff = pd.DataFrame()
            for label_sel in list_labels:
                df_label = df[df["annotation"] == label_sel]
                values = list(df_label["start_datetime"].drop_duplicates())
                common_values = []
                diff_values = []
                error_values = []
                for value in values:
                    if df_label["start_datetime"].to_list().count(value) == 2:
                        common_values.append(value)
                    elif df_label["start_datetime"].to_list().count(value) == 1:
                        diff_values.append(value)
                    else:
                        error_values.append(value)

                df_label_inter = df_label[
                    df_label["start_datetime"].isin(common_values)
                ].reset_index(drop=True)
                df_label_inter = df_label_inter.drop_duplicates(subset="start_datetime")
                df_inter = pd.concat([df_inter, df_label_inter]).reset_index(drop=True)

                df_label_diff = df_label[
                    df_label["start_datetime"].isin(diff_values)
                ].reset_index(drop=True)
                df_diff = pd.concat([df_diff, df_label_diff]).reset_index(drop=True)

            if user_sel == "intersection":
                df = df_inter
                list_annotators = [" ∩ ".join(list_annotators)]
            elif user_sel == "union":
                df = pd.concat([df_diff, df_inter]).reset_index(drop=True)
                df = df.sort_values("start_datetime")
                list_annotators = [" ∪ ".join(list_annotators)]

            df["annotator"] = list_annotators[0]

    columns = [
        "file",
        "max_time",
        "max_freq",
        "annotators",
        "labels",
        "tz_data",
        "timestamp_file",
    ]

    return df.reset_index(drop=True)


# def reshape_timebin(df: pd.DataFrame,
#                     timebin_new: int = None,
#                     timestamp_file: str = None,
#                     reshape_method: str = 'timestamp') -> pd.DataFrame:
#     ''' Changes the timebin (time resolution) of a detection dataframe
#     ex :    -from a raw PAMGuard detection file to a detection file with 10s timebin
#             -from an 10s detection file to a 1min / 1h / 24h detection file
#     Parameter:
#         df : detection dataframe
#         timebin_new : time resolution to base the detections on, if not provided it is asked to the user
#         timestamp_file : path to the an APLOSE formatted timestamp file.
#                         It is used to create a reshaped detection file with timestamps that matches the APLOSE annotations.
#     Returns:
#         df_new : detection dataframe with the new timebin
#     '''
#     if isinstance(df, pd.DataFrame) is False:
#         raise Exception("Not a dataframe passed, reshape aborted")

#     annotators = list(df['annotator'].drop_duplicates())
#     labels = list(df['annotation'].drop_duplicates())

#     df_nobox = df.loc[(df['start_time'] == 0) & (df['end_time'] == max(df['end_time'])) & (df['end_frequency'] == max(df['end_frequency']))]
#     max_time = 1 if len(df_nobox) == 0 else int(max(df['end_time']))
#     max_freq = int(max(df['end_frequency']))

#     tz_data = pytz.FixedOffset(df['start_datetime'][0].utcoffset().total_seconds() // 60)

#     if timebin_new is None:
#         while True:
#             timebin_new = easygui.buttonbox('Select a new time resolution for the detections', df['dataset'][0], ['10s', '1min', '10min', '1h', '24h'])
#             if timebin_new == '10s':
#                 f = timebin_new
#                 timebin_new = 10
#             elif timebin_new == '1min':
#                 f = timebin_new
#                 timebin_new = 60
#             elif timebin_new == '10min':
#                 f = timebin_new
#                 timebin_new = 600
#             elif timebin_new == '1h':
#                 f = timebin_new
#                 timebin_new = 3600
#             elif timebin_new == '24h':
#                 f = timebin_new
#                 timebin_new = 86400

#             if timebin_new > max_time: break
#             else: easygui.msgbox('New time resolution is equal or smaller than the original one', 'Warning', 'Ok')
#     else: f = str(timebin_new) + 's'

#     df_new = pd.DataFrame()
#     if isinstance(annotators, str): annotators = [annotators]
#     if isinstance(labels, str): labels = [labels]
#     for annotator in annotators:
#         for label in labels:

#             df_detect_prov = df[(df['annotator'] == annotator) & (df['annotation'] == label)]

#             if len(df_detect_prov) == 0:
#                 continue

#             if not timestamp_file:
#                 t = t_rounder(t=df_detect_prov['start_datetime'].iloc[0], res=timebin_new)
#                 t2 = t_rounder(df_detect_prov['start_datetime'].iloc[-1], timebin_new) + dt.timedelta(seconds=timebin_new)
#                 time_vector = [ts.timestamp() for ts in pd.date_range(start=t, end=t2, freq=f)]
#             else:
#                 timestamp_csv = pd.read_csv(timestamp_file, parse_dates=['timestamp'])
#                 timestamp_range = timestamp_csv['timestamp'].to_list()
#                 '''
#                 original_timebin = int(timebin_new / max_time)
#                 timestamp_csv2 = timestamp_csv[0::original_timebin] # a verif
#                 # timestamp_range = timestamp_csv['timestamp'].to_list()
#                 timestamp_range = timestamp_csv2['timestamp'].to_list()
#                 timestamp_range.append(timestamp_range[-1] + pd.Timedelta(timebin_new, unit='second'))

#                 # time_vector_raw = [ts.timestamp() for ts in timestamp_range]
#                 # time_vector = time_vector_raw[0::int(timebin_new / (time_vector_raw[1] - time_vector_raw[0]))]
#                 time_vector = [ts.timestamp() for ts in timestamp_range]
#                 '''
#                 if reshape_method == 'timebin':
#                     t = t_rounder(t=timestamp_range[0], res=timebin_new)
#                     t2 = t_rounder(timestamp_range[-1], timebin_new) + pd.Timedelta(seconds=timebin_new)
#                     time_vector = [ts.timestamp() for ts in pd.date_range(start=t, end=t2, freq=f)]
#                 elif reshape_method == 'timestamp':
#                     time_vector = [ts.timestamp() for ts in timestamp_range]

#             # here test to find for each time vector value which filename corresponds
#             filenames = sorted(list(set(df_detect_prov['filename'])))
#             if not all(isinstance(filename, str) for filename in filenames):
#                 if all(math.isnan(filename) for filename in filenames):
#                     # FPOD case: the filenames of a FPOD csv file are NaN values
#                     filenames = [i.strftime('%Y-%m-%dT%H:%M:%S%z') for i in df_detect_prov['start_datetime']]

#             ts_filenames = [extract_datetime(var=filename, tz=tz_data).timestamp()for filename in filenames]

#             filename_vector = []
#             for ts in time_vector:
#                 index = bisect.bisect_left(ts_filenames, ts)
#                 if index == 0:
#                     filename_vector.append(filenames[index])
#                 elif index == len(ts_filenames):
#                     filename_vector.append(filenames[index - 1])
#                 else:
#                     # filename_vector.append(filenames[index - 1]) # pb sur cette ligne, à creuser
#                     filename_vector.append(filenames[index])

#             times_detect_beg = [detect.timestamp() for detect in df_detect_prov['start_datetime']]
#             times_detect_end = [detect.timestamp() for detect in df_detect_prov['end_datetime']]

#             detect_vec, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
#             for i in range(len(times_detect_beg)):
#                 for j in range(k, len(time_vector) - 1):
#                     if int(times_detect_beg[i] * 1e7) in range(int(time_vector[j] * 1e7), int(time_vector[j + 1] * 1e7)) or int(times_detect_end[i] * 1e7) in range(int(time_vector[j] * 1e7), int(time_vector[j + 1] * 1e7)):
#                         ranks.append(j)
#                         k = j
#                         break
#                     else:
#                         continue

#             ranks = sorted(list(set(ranks)))
#             detect_vec[ranks] = 1
#             detect_vec = list(detect_vec)

#             start_datetime_str, end_datetime_str, filename = [], [], []
#             for i in range(len(time_vector)):
#                 if detect_vec[i] == 1:
#                     start_datetime = pd.Timestamp(time_vector[i], unit='s', tz=tz_data)
#                     start_datetime_str.append(start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[:-8] + start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-5:-2] + ':' + start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-2:])
#                     end_datetime = pd.Timestamp(time_vector[i] + timebin_new, unit='s', tz=tz_data)
#                     end_datetime_str.append(end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[:-8] + end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-5:-2] + ':' + end_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f%z')[-2:])
#                     filename.append(filename_vector[i])

#             df_new_prov = pd.DataFrame()
#             dataset_str = list(set(df_detect_prov['dataset']))

#             df_new_prov['dataset'] = dataset_str * len(start_datetime_str)
#             df_new_prov['filename'] = filename
#             df_new_prov['start_time'] = [0] * len(start_datetime_str)
#             df_new_prov['end_time'] = [timebin_new] * len(start_datetime_str)
#             df_new_prov['start_frequency'] = [0] * len(start_datetime_str)
#             df_new_prov['end_frequency'] = [max_freq] * len(start_datetime_str)
#             df_new_prov['annotation'] = list(set(df_detect_prov['annotation'])) * len(start_datetime_str)
#             df_new_prov['annotator'] = list(set(df_detect_prov['annotator'])) * len(start_datetime_str)
#             df_new_prov['start_datetime'], df_new_prov['end_datetime'] = start_datetime_str, end_datetime_str

#             df_new = pd.concat([df_new, df_new_prov])

#         df_new['start_datetime'] = [pd.to_datetime(d, format='%Y-%m-%dT%H:%M:%S.%f%z') for d in df_new['start_datetime']]
#         df_new['end_datetime'] = [pd.to_datetime(d, format='%Y-%m-%dT%H:%M:%S.%f%z') for d in df_new['end_datetime']]
#         df_new = df_new.sort_values(by=['start_datetime'])

#     return df_new


def read_yaml(file: Path) -> dict:
    """Reads yaml file to extract detection parameters. The extracted parameters
    are then used to import detections using the 'sorting_detection' function.
    Parameters
    ----------
        file: path to the yaml file
    Returns
    -------
        parameters: Dictionary containing a set of parameters for each csv file
    """
    with open(file, "r") as yaml_file:
        parameters = yaml.safe_load(yaml_file)

    for filename in parameters.keys():

        if not Path(filename).exists():
            raise FileNotFoundError(f"{filename} does not exist")
        else:
            parameters[filename]['file'] = Path(filename)

        if parameters[filename]["timebin_new"] and not isinstance(parameters[filename]["timebin_new"], int):
            raise ValueError(f"An integer must be passed to 'timebin_new', {parameters[filename]['timebin_new']} not a valid value.")

        if parameters[filename]["tz"] is not None:
            if isinstance(parameters[filename]["tz"], int):
                parameters[filename]["tz"] = pytz.FixedOffset(parameters[filename]["tz"])
            else:
                raise ValueError(f"An integer must be passed to 'tz', {parameters[filename]['tz']} not a valid value.")

        if parameters[filename]["fmin_filter"] and not isinstance(parameters[filename]["fmin_filter"], int):
            raise ValueError(f"An integer must be passed to 'fmin_filter', {parameters[filename]['fmin_filter']} not a valid value.")

        if parameters[filename]["fmax_filter"] and not isinstance(parameters[filename]["fmax_filter"], int):
            raise ValueError(f"An integer must be passed to 'fmax_filter', {parameters[filename]['fmax_filter']} not a valid value.")

        if parameters[filename]["datetime_begin"]:
            try:
                parameters[filename]["datetime_begin"] = pd.Timestamp(parameters[filename]["datetime_begin"])
            except ValueError as e:
                raise ValueError(f"Invalid date format for 'datetime_begin': {parameters[filename]['datetime_begin']}") from e

        if parameters[filename]["datetime_end"]:
            try:
                parameters[filename]["datetime_end"] = pd.Timestamp(parameters[filename]["datetime_end"])
            except ValueError as e:
                raise ValueError(f"Invalid date format for 'datetime_end': {parameters[filename]['datetime_end']}") from e

        if all([parameters[filename]["datetime_begin"], parameters[filename]["datetime_end"]]) and parameters[filename]["datetime_begin"] >= parameters[filename]["datetime_end"]:
            raise ValueError(f'{parameters[filename]["datetime_begin"]} >= {parameters[filename]["datetime_end"]}')

        if parameters[filename]["annotator"] and not isinstance(parameters[filename]["annotator"], str):
            raise ValueError(f"A string must be passed to 'annotator', {parameters[filename]['annotator']} not a valid value.")

        if parameters[filename]["annotation"] and not isinstance(parameters[filename]["annotation"], str):
            raise ValueError(f"A string must be passed to 'annotation', {parameters[filename]['annotation']} not a valid value.")

        if parameters[filename]["box"] and not isinstance(parameters[filename]["box"], bool):
            raise ValueError(f"A boolean must be passed to 'box', {parameters[filename]['box']} not a valid value.")

        if parameters[filename]["user_sel"] and parameters[filename]["user_sel"] not in ['union', 'intersection', 'all']:
            raise ValueError(f"Either 'union', 'intersection' or 'all' must be passed to 'user_sel', {parameters[filename]['user_sel']} not a valid value.")

        if parameters[filename]["timestamp_file"]:
            if not Path(parameters[filename]["timestamp_file"]).exists():
                raise FileNotFoundError(f"{parameters[filename]['timestamp_file']} does not exist")
            else:
                parameters[filename]["timestamp_file"] = Path(parameters[filename]["timestamp_file"])

        # if len(parameters.keys()) == 1:
        #     parameters = parameters[list(parameters.keys())[0]]

    return parameters


def find_delimiter(file: str | Path) -> str:
    """Finds the proper delimiter for a csv file

    Parameters
    ----------
    file: Path to the csv file

    Returns
    -------
    delimiter: The delimiter to use to read the file
    """
    with open(file, "r", newline="") as csv_file:
        try:
            temp_lines = csv_file.readline() + "\n" + csv_file.readline()
            dialect = csv.Sniffer().sniff(temp_lines, delimiters=",;")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        return delimiter

"""
utile ??

def task_status_selection(
    files: List[str], df_detections: pd.DataFrame, user: Union[str, List[str]] = "all"
) -> pd.DataFrame:
    Filters a detection DataFrame to select only the segments that all annotator have completed (i.e. status == 'FINISHED')
    Parameters :
        file : list of path(s) to the status file(s), can be a str too
        df_detections : df of the detections (output of sorting_detections)
        user : string of list of strings, this argument is used to select the annotator to take into consideration
            - user='all', then all the annotators of the status file are used
            - user='annotator_name', then only one annotator is used
            - user=[list of annotators], then only the annotators present in the list are used
    Returns :
        df_kept : df of the detections sorted according to the selected annotators


    if isinstance(files, str):
        files = [files]  # Convert the single string to a list with one element

    result_df = pd.DataFrame()
    for file in files:
        delimiter = find_delimiter(file)
        df = pd.read_csv(file, sep=delimiter)
        annotators_df = [
            i for i in list(df.columns) if i != "dataset" and i != "filename"
        ]

        # selection of the annotators according to the user argument
        if user == "all":
            list_annotators = [
                i for i in list(df.columns) if i != "dataset" and i != "filename"
            ]
        elif isinstance(user, list):
            for u in user:
                if u not in annotators_df:
                    raise Exception(f"'{u}' not present in the task satuts file")
            list_annotators = user
        elif isinstance(user, str) and user != "all":
            if user not in annotators_df:
                raise Exception(f"'{user}' not present in the task satuts file")
            list_annotators = [user]

        df_users = df_detections[df_detections["annotator"].isin(list_annotators)]

        filename_list = list(
            df[df[list_annotators].eq("FINISHED").all(axis=1)]["filename"]
        )
        ignored_list = list(
            df[~df[list_annotators].eq("FINISHED").all(axis=1)]["filename"]
        )

        df_kept = df_users[df_users["filename"].isin(filename_list)]
        # df_ignored = df_users[df_users['filename'].isin(ignored_list)]

        print(
            f"\n{os.path.basename(file)}: {len(ignored_list)} files ignored", end="\n"
        )
        result_df = pd.concat([result_df, df_kept]).reset_index(drop=True)

    return result_df
"""

def extract_datetime(
    var: str, tz: pytz._FixedOffset = None, formats=None
) -> pd.Timestamp | str:
    """Extracts datetime from filename based on the date format
    Parameters :
        var : name of the wav file
        tz : timezone info
        formats : the date template in strftime format. For example, `2017/02/24` has the template `%Y/%m/%d`
                    For more information on strftime template, see https://strftime.org/
    Returns :
        date_obj : pd.Timestamp object corresponding to the datetime found in var
    """

    if formats is None:
        # add more format if necessary
        formats = [
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}",
            r"\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}",
            r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}",
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
            r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}",
            r"\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2}",
            r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}",
            r"\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}",
        ]
    match = None
    for f in formats:
        match = re.search(f, var)
        if match:
            break
    if not match:
        raise ValueError(f"{var}: No datetime found")

    dt_string = match.group()
    if f == r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}":
        dt_format = "%Y-%m-%dT%H-%M-%S"
    elif f == r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}":
        dt_format = "%Y-%m-%d_%H-%M-%S"
    elif f == r"\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}":
        dt_format = "%y%m%d%H%M%S"
    elif f == r"\d{2}\d{2}\d{2}_\d{2}\d{2}\d{2}":
        dt_format = "%y%m%d_%H%M%S"
    elif f == r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}":
        dt_format = "%Y-%m-%d %H:%M:%S"
    elif f == r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}":
        dt_format = "%Y-%m-%dT%H:%M:%S"
    elif f == r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}":
        dt_format = "%Y_%m_%d_%H_%M_%S"
    elif f == r"\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2}":
        dt_format = "%Y_%m_%dT%H_%M_%S"
    elif f == r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}":
        dt_format = "%Y-%m-%dT%H_%M_%S"
    elif f == r"\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}":
        dt_format = "%Y%m%dT%H%M%S"

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


def t_rounder(t: pd.Timestamp, res: int):
    """Rounds a Timestamp according to the user specified resolution : 10s / 1min / 10 min / 1h / 24h
    Parameter :
        t: Timestamp to round
        res: integer corresponding to the new resolution in seconds
    Returns :
        t: rounded Timestamp
    """
    if res == 600:  # 10min
        minute = t.minute
        minute = round(minute / 10) * 10
        hour = t.hour
        if minute < 60:
            t = t.replace(minute=minute, second=0, microsecond=0)
        else:
            if hour < 23:
                hour += 1
            else:
                hour = 0
                t += pd.Timedelta(days=1)
        t = t.replace(hour=hour, minute=0, second=0, microsecond=0)
    elif res == 10:  # 10s
        seconde = t.second
        seconde = round(seconde / 10) * 10
        t = t.replace(second=seconde, microsecond=0)
    elif res == 60:  # 1min
        t = t.replace(second=0, microsecond=0)
    elif res == 3600:  # 1h
        t = t.replace(minute=0, second=0, microsecond=0)
    elif res == 86400:  # 24h
        if t > t.replace(hour=12, minute=0, second=0, microsecond=0):
            t = t.replace(hour=0, minute=0, second=0, microsecond=0)
            t += pd.Timedelta(days=1)
        else:
            t = t.replace(hour=0, minute=0, second=0, microsecond=0)
    elif res == 3:
        t = t.replace(microsecond=0)
    else:
        raise ValueError(f"res={res}s: Resolution not available")
    return t


def export2Raven(
    tuple_info,
    timestamps,
    df,
    timebin_new,
    bin_height,
    selection_vec: bool = False,
    offset: bool = False,
) -> pd.DataFrame:
    """Export a given vector to Raven formatted table
    Parameters :
        df : dataframe of the detections
        timebin_new : int, duration of the detection boxes to export, if set to 0, the original detections are exported
        bin_height : the maximum frequency of the exported timebins
        tuple_info : tuple containing info such as the filenames of the wav files, their durations and datetimes
        selection_vec : if it is set to False, all the timebins are exported, else the selection_vec is used to selec the wanted timebins to export, for instance it corresponds to all the positives timebins, containing detections
    """

    file_list = list(tuple_info[0])
    file_datetimes = tuple_info[1]
    dur = list(tuple_info[2])

    offsets = [
        (file_datetimes[i] + dt.timedelta(seconds=dur[i])).timestamp()
        - (file_datetimes[i + 1]).timestamp()
        for i in range(len(file_datetimes) - 1)
    ]
    offsets_cumsum = list(np.cumsum([offsets[i] for i in range(len(offsets))]))
    offsets_cumsum.insert(0, 0)
    idx_wav_df = [file_list.index(df["filename"][i]) for i in range(len(df))]

    if timebin_new > 0:

        # time_vec = []
        # for i in range(len(file_list)):
        #     timestamp = file_datetimes[i].timestamp() + offsets_cumsum[i]
        #     durations = np.arange(0, dur[i], timebin_new).astype(int)

        #     for elem in durations:
        #         time_vec.append(timestamp + elem)
        time_vec = np.arange(
            t_rounder(file_datetimes[0], res=timebin_new).timestamp(),
            file_datetimes[-1].timestamp() + dur[-1],
            timebin_new,
        ).astype(int)

        if selection_vec is True:
            times_det_beg = [
                df["start_datetime"][i].timestamp()
                + offsets_cumsum[idx_wav_df[i]]
                + 1e-8 * timebin_new
                for i in range(len(df))
            ]
            times_det_end = [
                df["end_datetime"][i].timestamp()
                + offsets_cumsum[idx_wav_df[i]]
                - 1e-8 * timebin_new
                - 1e-8 * timebin_new
                for i in range(len(df))
            ]

            det_vec, ranks, k = np.zeros(len(time_vec) - 1, dtype=int), [], 0
            for i in range(len(times_det_beg)):
                for j in range(k, len(time_vec) - 0):
                    if int(times_det_beg[i] * 1e8) in range(
                        int(time_vec[j] * 1e8), int(time_vec[j + 1] * 1e8)
                    ) or int(times_det_end[i] * 1e8) in range(
                        int(time_vec[j] * 1e7), int(time_vec[j + 1] * 1e7)
                    ):
                        ranks.append(j)
                        k = j
                        break
                    else:
                        continue
            ranks = sorted(list(set(ranks)))
            det_vec[np.isin(range(len(time_vec) - 1), ranks)] = 1

        else:
            det_vec = [1] * (len(time_vec) - 1)

        start_time = [
            int(time_vec[i] - file_datetimes[0].timestamp())
            for i in range(0, len(time_vec) - 1)
        ]
        end_time = [
            int(time_vec[i] - file_datetimes[0].timestamp())
            for i in range(1, len(time_vec))
        ]
        delta = [end_time[i] - start_time[i] for i in range(len(start_time))]
        df_time = pd.DataFrame(
            {"start": start_time, "end": end_time, "d": delta, "vec": det_vec}
        )
        df_time_sorted = df_time[
            (df_time["d"] == timebin_new) & (df_time["vec"] == 1)
        ].reset_index(drop=True)

        df_PG2Raven = pd.DataFrame()
        df_PG2Raven["Selection"] = np.arange(1, len(df_time_sorted) + 1)
        df_PG2Raven["View"], df_PG2Raven["Channel"] = [1] * len(df_time_sorted), [
            1
        ] * len(df_time_sorted)
        df_PG2Raven["Begin Time (s)"] = df_time_sorted["start"]
        df_PG2Raven["End Time (s)"] = df_time_sorted["end"]
        df_PG2Raven["Low Freq (Hz)"] = [0] * len(df_time_sorted)
        df_PG2Raven["High Freq (Hz)"] = [bin_height] * len(df_time_sorted)

    else:
        start_time = [
            df["start_time"][i] + offsets_cumsum[idx_wav_df[i]] for i in range(len(df))
        ]
        end_time = [
            df["end_time"][i] + offsets_cumsum[idx_wav_df[i]] for i in range(len(df))
        ]

        df_PG2Raven = pd.DataFrame()
        df_PG2Raven["Selection"] = np.arange(1, len(df) + 1)
        df_PG2Raven["View"], df_PG2Raven["Channel"] = [1] * len(df), [1] * len(df)
        df_PG2Raven["Begin Time (s)"] = start_time
        df_PG2Raven["End Time (s)"] = end_time
        df_PG2Raven["Low Freq (Hz)"] = df["start_frequency"]
        df_PG2Raven["High Freq (Hz)"] = df["end_frequency"]

    if offset is True:
        df_offset = pd.DataFrame(
            {"filename": file_list, "offset_cumsum": offsets_cumsum}
        )
        return df_PG2Raven, df_offset
    else:
        return df_PG2Raven, None


def get_season(ts: pd.Timestamp) -> str:
    """'day of year' ranges for the northern hemisphere
    Parameter :
        ts : Timestamp
    Returns :
        season : string corresponding to the season and year of the datetime (ex : if datetime is 01/01/2023, returns 'winter 2022')
    """

    winter = [1, 2, 12]
    spring = [3, 4, 5]
    summer = [6, 7, 8]
    autumn = [9, 10, 11]

    if ts.month in spring:
        season = "spring" + " " + str(ts.year)
    elif ts.month in summer:
        season = "summer" + " " + str(ts.year)
    elif ts.month in autumn:
        season = "autumn" + " " + str(ts.year)
    elif ts.month in winter and ts.month != 12:
        season = "winter" + " " + str(ts.year - 1)
    elif ts.month in winter and ts.month == 12:
        season = "winter" + " " + str(ts.year)

    return season


__converter = {
    "%Y": r"[12][0-9]{3}",
    "%y": r"[0-9]{2}",
    "%m": r"(0[1-9]|1[0-2])",
    "%d": r"([0-2][0-9]|3[0-1])",
    "%H": r"([0-1][0-9]|2[0-4])",
    "%I": r"(0[1-9]|1[0-2])",
    "%p": r"(AM|PM)",
    "%M": r"[0-5][0-9]",
    "%S": r"[0-5][0-9]",
    "%f": r"[0-9]{6}",
}


def convert_template_to_re(date_template: str) -> str:
    """Converts a template in strftime format to a matching regular expression
    Parameter :
        date_template: the template in strftime format
    Returns :
        The regular expression matching the template
    """

    res = ""
    i = 0
    while i < len(date_template):
        if date_template[i : i + 2] in __converter:
            res += __converter[date_template[i : i + 2]]
            i += 1
        else:
            res += date_template[i]
        i += 1

    return res


def input_date(msg):
    """Based on selection_type, ask the user a folder and yields all the wav files inside it or ask the user multiple wav files
    Parameters :
        msg : Message to tell the user what date they have to enter (begin, end...)
    Returns :
        date_dt : aware datetime entered by the user
    """

    title = "Date"
    fieldNames = [
        "Year [YYYY]",
        "Month [m]",
        "Day [d]",
        "Hour [H]",
        "Minute [M]",
        "Second [S]",
        "Timezone [+/-HHMM]",
    ]
    fieldValues = []  # Initialize with empty values

    while True:
        fieldValues = easygui.multenterbox(msg, title, fieldNames, fieldValues)

        if fieldValues is None:
            # User canceled the input
            return None

        errmsg = ""
        for i in range(len(fieldNames)):
            if fieldValues[i].strip() == "":
                errmsg += f"'{fieldNames[i]}' is a required field.\n"

        if errmsg == "":
            break  # No validation errors

        easygui.msgbox(errmsg, title)

    year, month, day, hour, minute, second = map(int, fieldValues[:-1])
    hours_offset = int(fieldValues[-1][:3])
    minutes_offset = int(fieldValues[-1][3:])
    tz = pytz.FixedOffset(hours_offset * 60 + minutes_offset)

    date_dt = pd.Timestamp(year, month, day, hour, minute, second, tzinfo=tz)
    return date_dt


def suntime_hour(start: pd.Timestamp, stop: pd.Timestamp, lat: float, lon: float):
    """Fetch sunrise and sunset hours for dates between date_beg and date_end
    Parameters :
        start : pd.Timestamp, start datetime of when to fetch sun hour
        stop : pd.Timestamp, end datetime of when to fetch sun hour
        lat : float, latitude in decimal degrees
        lon : float, longitude in decimal degrees
    Returns :
        hour_sunrise : list of float with sunrise decimal hours for each day between date_beg and date_end
        hour_sunset : list of float with sunset decimal hours for each day between date_beg and date_end
    """
    # timezone
    tz = start.tz

    # localisation info
    gps = astral.LocationInfo(timezone=tz, latitude=lat, longitude=lon)

    # List of days during when the data were recorded
    h_sunrise, h_sunset, dt_dusk, dt_dawn, dt_day, dt_night = [], [], [], [], [], []

    # For each day : find time of sunset, sun rise, begin dawn and dusk
    for date in [ts.date() for ts in pd.date_range(start.normalize(), stop.normalize(), freq='D')]:

        # nautical twilight = 12, see def here : https://www.timeanddate.com/astronomy/nautical-twilight.html
        suntime = sun(gps.observer, date=date, dawn_dusk_depression=12)
        dawn, day, _, dusk, night  = [pd.Timestamp(suntime[period]).tz_convert(tz) for period in suntime]

        for lst, period in zip([h_sunrise, h_sunset], [day, night]):
            lst.append(period.hour + period.minute / 60 + period.second / 3600)

        for lst, period in zip([dt_dawn, dt_day, dt_dusk, dt_night], [dawn, day, dusk, night]):
            lst.append(period)

    return h_sunrise, h_sunset, dt_dusk, dt_dawn, dt_day, dt_night

def get_coordinates():
    """Ask user input to get GPS coordinates.

    Returns
    -------
    latitude : float
    longitude : float
    """
    title = "Coordinates in degree° minute'"
    msg = "latitude (N/S) and longitude (E/W)"
    field_names = ["lat decimal degree", "lon decimal degree"]
    field_values = easygui.multenterbox(msg, title, field_names)

    # make sure that none of the fields was left blank
    while True:
        if field_values is None:
            raise TypeError("'get_coordinates()' was cancelled")

        lat, lon = field_values
        errmsg = ""
        try:
            lat_val = float(lat.strip())  # Convert to float for latitude
            if lat_val < -90 or lat_val > 90:
                errmsg += f"'{lat}' is not a valid latitude. It must be between -90 and 90.\n"
        except ValueError:
            errmsg += f"'{lat}' is not a valid entry for latitude.\n"

        try:
            lon_val = float(lon.strip())  # Convert to float for longitude
            if lon_val < -180 or lon_val > 180:
                errmsg += f"'{lon}' is not a valid longitude. It must be between -180 and 180.\n"
        except ValueError:
            errmsg += f"'{lon}' is not a valid entry for longitude.\n"

        if errmsg == "":
            break

        field_values = easygui.multenterbox(errmsg, title, field_names)

    lat = float(lat.strip())
    lon = float(lon.strip())

    return lat, lon

def get_duration(title: str = 'Get duration', msg: str = 'Enter a time alias', default: str = '10min', base: bool = False):
    """Ask user input to get time duration.
    Offset aliases are to be used,
    e.g.: '5D' => 432_000s
    '2h' => 7_200s
    See https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    Parameters
    ----------
    title : str
    msg : str
    default : '10min' => 600s
    base : bool, optional, default False, if True, return the base of the value.
        For instance, '10min" => '<Minute>'
    Returns
    -------
    The total number of seconds of the entered time alias.
    The smallest duration is 1s.
    """
    value = easygui.enterbox(msg=f'{msg}', title=f'{title}', default=f'{default}', strip=True)

    while True:
        if value is None:
            raise TypeError("'get_duration()' was cancelled")

        errmsg = ""
        try:
            base_str = to_offset(value).base.freqstr
            value = int(pd.Timedelta(to_offset(value)).total_seconds())
        except ValueError:
            errmsg = f"'{value}' is not a valid time alias."

        if errmsg == "":
            break

        value = easygui.enterbox(msg=errmsg, title=f'{title}', strip=True)

    if base:
        return value, base_str
    else:
        return value


def get_datetime_format(title: str = 'Get datetime format', msg: str = 'Enter a datetime format code', default: str = '%d/%m/%Y\n%H:%M') -> str:
    """Ask user input to get datetime format.
    Datetime format codes are to be used,
    See https://docs.python.org/fr/3/library/datetime.html
    Parameters
    ----------
    title : str
    msg : str
    default : '%d/%m/%Y\n%H:%M'
    """
    fmt = easygui.enterbox(msg=f'{msg}', title=f'{title}', default=f'{default}', strip=True)

    while True:
        if fmt is None:
            raise TypeError("'get_duration()' was cancelled")

        errmsg = ""
        datetime_test = pd.Timestamp('now')
        try:
            datetime_test.strftime(format=fmt)
        except ValueError:
            errmsg = f"'{fmt}' is not a valid datetime format code."

        if errmsg == "":
            break

        fmt = easygui.enterbox(msg=errmsg, title=f'{title}', strip=True)

    return fmt


def stats_diel_pattern(
    df_detections: pd.DataFrame,
    begin_date: dt.datetime,
    end_date: dt.datetime,
    lat: float = None,
    lon: float = None,
):
    """Plot detection proportions for each light regime (night/dawn/day/dawn)
    Parameters :
        begin_date : begin datetime of data to analyse
        end_date : end datetime of data to analyse
        lat : float latitude in Decimal Degrees
        lon : float longitude in Decimal Degrees
    Returns :
        lr : df used to plot the detections
        BoxName : list of light regimes
    """

    tz_data = df_detections["start_datetime"][0].tz

    if not isinstance(lat, float) and not isinstance(lat, int) and lat is not None:
        raise ValueError("Invalid latitude")
    elif not isinstance(lon, float) and not isinstance(lon, int) and lon is not None:
        raise ValueError("Invalid longitude")
    elif lat is None or lon is None:
        # User input : gps coordinates in Decimal Degrees
        title = "Coordinates in degree° minute' "
        msg = "Latitudes (N/S) and longitudes (E/W)"
        fieldNames = ["Lat Decimal Degree", "Lon Decimal Degree"]
        fieldValues = easygui.multenterbox(msg, title, fieldNames)

        # make sure that none of the fields was left blank
        while 1:
            if fieldValues is None:
                break
            errmsg = ""
            for i in range(len(fieldNames)):
                value = fieldValues[i]
                if not value.strip():
                    errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
                elif not isinstance(value, float) and not isinstance(value, int):
                    errmsg = errmsg + (
                        '"%s" must be a valid number.\n\n' % fieldNames[i]
                    )
            if errmsg == "":
                break  # no problems found
            fieldValues = easygui.multpasswordbox(
                errmsg, title, fieldNames, fieldValues
            )
            print("Reply was:", fieldValues)

            lat = fieldValues[0]
            lon = fieldValues[1]

    # Compute sunrise and sunset decimal hour at the dataset location
    # Seems to only work with UTC data ?
    [_, _, dt_dusk, dt_dawn, dt_day, dt_night] = suntime_hour(
        begin_date, end_date, tz_data, lat, lon
    )

    # List of days in the dataset
    list_days = [dt.date(d.year, d.month, d.day) for d in dt_day]
    # Compute dusk_duration, dawn_duration, day_duration, night_duration
    dawn_duration = [b - a for a, b in zip(dt_dawn, dt_day)]
    day_duration = [b - a for a, b in zip(dt_day, dt_night)]
    dusk_duration = [b - a for a, b in zip(dt_night, dt_dusk)]
    night_duration = [
        dt.timedelta(hours=24) - dawn - day - dusk
        for dawn, day, dusk in zip(dawn_duration, day_duration, dusk_duration)
    ]
    # Convert to decimal
    dawn_duration_dec = [dawn_d.total_seconds() / 3600 for dawn_d in dawn_duration]
    day_duration_dec = [day_d.total_seconds() / 3600 for day_d in day_duration]
    dusk_duration_dec = [dusk_d.total_seconds() / 3600 for dusk_d in dusk_duration]
    night_duration_dec = [night_d.total_seconds() / 3600 for night_d in night_duration]

    # Assign a light regime to each detection
    # : 1 = night ; 2 = dawn ; 3 = day ; 4 = dusk
    day_det = [
        start_datetime.date() for start_datetime in df_detections["start_datetime"]
    ]
    light_regime = []
    for idx_day, day in enumerate(list_days):
        for idx_det, d in enumerate(day_det):
            # If the detection occured during 'day'
            if d == day:
                if (
                    df_detections["start_datetime"][idx_det] > dt_dawn[idx_day]
                    and df_detections["start_datetime"][idx_det] < dt_day[idx_day]
                ):
                    lr = 2
                    light_regime.append(lr)
                elif (
                    df_detections["start_datetime"][idx_det] > dt_day[idx_day]
                    and df_detections["start_datetime"][idx_det] < dt_night[idx_day]
                ):
                    lr = 3
                    light_regime.append(lr)
                elif (
                    df_detections["start_datetime"][idx_det] > dt_night[idx_day]
                    and df_detections["start_datetime"][idx_det] < dt_dusk[idx_day]
                ):
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
            nb_det_night.append(light_regime[idx_det[0] : idx_det[-1]].count(1))
            nb_det_dawn.append(light_regime[idx_det[0] : idx_det[-1]].count(2))
            nb_det_day.append(light_regime[idx_det[0] : idx_det[-1]].count(3))
            nb_det_dusk.append(light_regime[idx_det[0] : idx_det[-1]].count(4))

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

    LIGHTR = [
        nb_det_night_corr_norm,
        nb_det_dawn_corr_norm,
        nb_det_day_corr_norm,
        nb_det_dusk_corr_norm,
    ]
    BoxName = ["Night", "Dawn", "Day", "Dusk"]

    lr = pd.DataFrame(LIGHTR, index=BoxName).transpose()

    return lr, BoxName


def stat_box_day(
    data_test: pd.DataFrame, df_detections: pd.DataFrame, detector: str
) -> pd.DataFrame:
    """Plot detection proportions for each hour of the day
    Parameters :
        data_test : df with data infos
        df_detections : APLOSE formatted df of the detections
        detector : name of the automatic detector to use
    Returns :
        result : df used to plot the detections
    """

    hour_list = ["{:02d}:00".format(i) for i in range(24)]
    hour_list.append("00:00")

    df_detections["date"] = [
        dt.datetime.strftime(i.date(), "%d/%m/%Y")
        for i in df_detections["start_datetime"]
    ]
    df_detections["season"] = [get_season(i) for i in df_detections["start_datetime"]]
    df_detections["dataset"] = [i.replace("_", " ") for i in df_detections["dataset"]]

    vec1 = [
        [data_test["datetime deployment"][i]] * len(data_test[f"df {detector}"][i])
        for i in data_test.index
    ]
    vec2 = [
        [data_test["datetime recovery"][i]] * len(data_test[f"df {detector}"][i])
        for i in data_test.index
    ]
    start_deploy, end_deploy = [], []
    [start_deploy.extend(inner_list) for inner_list in vec1]
    [end_deploy.extend(inner_list) for inner_list in vec2]
    df_detections["start_deploy"] = [pd.to_datetime(d) for d in start_deploy]
    df_detections["end_deploy"] = [pd.to_datetime(d) for d in end_deploy]

    result = {}
    list_dates = sorted(list(set(df_detections["date"])))  # list of dates
    for date in list_dates:
        detection_bydate = df_detections[
            df_detections["date"] == date
        ]  # sub-dataframe : per date
        list_datasets = sorted(
            list(set(detection_bydate["dataset"]))
        )  # dataset list for date=date

        for dataset in list_datasets:
            df = detection_bydate[detection_bydate["dataset"] == dataset].set_index(
                "start_datetime"
            )  # sub-dataframe : per date & per dataset

            # number of detections per hour of the day at date and at dataset
            detection_per_dataset = [
                len(df.between_time(hour_list[j], hour_list[j + 1], inclusive="left"))
                for j in (range(len(hour_list) - 1))
            ]

            deploy_beg_ts, deploy_end_ts = int(df["start_deploy"][0].timestamp()), int(
                df["end_deploy"][0].timestamp()
            )

            list_present_h = [
                dt.datetime.fromtimestamp(i)
                for i in list(range(deploy_beg_ts, deploy_end_ts, 3600))
            ]
            list_present_h2 = [
                dt.datetime.strftime(list_present_h[i], "%d/%m/%Y %H")
                for i in range(len(list_present_h))
            ]

            list_deploy_d = sorted(
                list(
                    set(
                        [
                            dt.datetime.strftime(
                                dt.datetime.fromtimestamp(i), "%d/%m/%Y"
                            )
                            for i in list(range(deploy_beg_ts, deploy_end_ts, 3600))
                        ]
                    )
                )
            )
            list_deploy_d2 = [d for i, d in enumerate(list_deploy_d) if d in date][0]

            list_present_h3 = []
            for item in list_present_h2:
                if item.startswith(list_deploy_d2):
                    list_present_h3.append(item)

            list_deploy = [
                df["date"][0] + " " + n for n in [f"{i:02}" for i in range(0, 24)]
            ]

            for i, h in enumerate(list_deploy):
                if h not in list_present_h3:
                    detection_per_dataset[i] = np.nan

            result[dataset, date] = detection_per_dataset

    return pd.DataFrame(result).T
