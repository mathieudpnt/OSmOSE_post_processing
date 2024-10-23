import pytz
import pandas as pd
import numpy as np
import easygui
import bisect
from astral.sun import sun
import astral
import csv
import yaml
from pathlib import Path
from pandas.tseries.frequencies import to_offset
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

from OSmOSE.utils.audio_utils import is_supported_audio_format
from OSmOSE.utils.timestamp_utils import is_datetime_template_valid
from OSmOSE.config import TIMESTAMP_FORMAT_AUDIO_FILE


def reshape_timebin(
    df: pd.DataFrame,
    timebin_new: int = None,
    timestamp: list[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Reshapes an APLOSE result DataFrame according to a new time bin

    Parameters
    ----------
    df: pd.DataFrame
        An APLOSE result DataFrame
    timebin_new: int
        The size of the new time bin in seconds
    timestamp: list(pd.Timestamp)
        A list of Timestamp objects

    Returns
    -------
    df_new_timebin: pd.DataFrame
        The reshaped DataFrame

    """
    df = df.sort_values("start_datetime").reset_index(drop=True)
    annotators = df["annotator"].drop_duplicates().to_list()
    labels = df["annotation"].drop_duplicates().to_list()
    max_freq = int(max(df["end_frequency"]))
    tz_data = df["start_datetime"][0].tz

    if not timebin_new:
        frequency = (
            str(
                get_duration(
                    title="Get duration", msg="Enter a new time bin", default="1min"
                )
            )
            + "s"
        )
    else:
        frequency = str(timebin_new) + "s"

    df_new_timebin = pd.DataFrame()
    for annotator in annotators:
        for label in labels:

            df_1annot_1label = df[
                (df["annotator"] == annotator) & (df["annotation"] == label)
            ]

            if len(df_1annot_1label) == 0:
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
                t1 = t_rounder(
                    t=df_1annot_1label["start_datetime"].iloc[0], res=timebin_new
                )
                t2 = t_rounder(
                    t=df_1annot_1label["end_datetime"].iloc[-1], res=timebin_new
                )
                time_vector = [
                    ts.timestamp()
                    for ts in pd.date_range(start=t1, end=t2, freq=frequency)
                ]

            ts_detect_beg = [
                ts.timestamp() for ts in df_1annot_1label["start_datetime"]
            ]
            ts_detect_end = [ts.timestamp() for ts in df_1annot_1label["end_datetime"]]

            # filenames = sorted(list(set(df_1annot_1label["filename"])))
            filenames = df_1annot_1label["filename"]
            # FPOD case: the filenames of a FPOD csv file are NaN values
            if all(pd.isna(filename) for filename in filenames):
                filenames = [
                    ts.strftime(TIMESTAMP_FORMAT_AUDIO_FILE) for ts in filenames
                ]

            filename_vector = []
            for ts in time_vector:
                # insertion of ts in ts_detect_beg, `bisect_left` provides
                # the index of the element in ts_detect_beg that is closest
                # to ts (left element if between 2 elements of the list).
                index = bisect.bisect_left(ts_detect_beg, ts)
                if index == 0:
                    filename_vector.append(filenames.iloc[index])
                else:
                    (
                        filename_vector.append(filenames.iloc[index])
                        if ts in ts_detect_beg
                        else filename_vector.append(filenames.iloc[index - 1])
                    )

            ranks1, ranks2 = [], []
            for i in range(len(df_1annot_1label)):
                idx1 = bisect.bisect_left(time_vector, ts_detect_beg[i])
                idx2 = bisect.bisect_left(time_vector, ts_detect_end[i])
                (
                    ranks1.append(idx1)
                    if ts_detect_beg[i] in time_vector
                    else ranks1.append(idx1 - 1)
                )
                (
                    ranks2.append(idx2)
                    if ts_detect_end[i] in time_vector
                    else ranks2.append(idx2 - 1)
                )

            detect_vec = [0] * len(time_vector)
            for start, end in zip(ranks1, ranks2):
                detect_vec[start : end + 1] = [1] * (end - start + 1)

            start_datetime, end_datetime, filename = [], [], []
            for i in range(len(time_vector)):
                if detect_vec[i] == 1:
                    start_datetime.append(
                        pd.Timestamp(time_vector[i], unit="s", tz=tz_data)
                    )
                    end_datetime.append(
                        pd.Timestamp(time_vector[i] + timebin_new, unit="s", tz=tz_data)
                    )
                    filename.append(filename_vector[i])

            df_1annot_1label_new_timebin = pd.DataFrame()
            df_1annot_1label_new_timebin["dataset"] = [
                df_1annot_1label["dataset"].iloc[0]
            ] * len(start_datetime)
            df_1annot_1label_new_timebin["filename"] = filename
            df_1annot_1label_new_timebin["start_time"] = [0] * len(start_datetime)
            df_1annot_1label_new_timebin["end_time"] = [timebin_new] * len(
                start_datetime
            )
            df_1annot_1label_new_timebin["start_frequency"] = [0] * len(start_datetime)
            df_1annot_1label_new_timebin["end_frequency"] = [max_freq] * len(
                start_datetime
            )
            df_1annot_1label_new_timebin["annotation"] = [label] * len(start_datetime)
            df_1annot_1label_new_timebin["annotator"] = [annotator] * len(
                start_datetime
            )
            df_1annot_1label_new_timebin["start_datetime"] = start_datetime
            df_1annot_1label_new_timebin["end_datetime"] = end_datetime

            df_new_timebin = pd.concat([df_new_timebin, df_1annot_1label_new_timebin])

    return df_new_timebin.sort_values(by=["start_datetime"])


def sort_detections(
    file: Path,
    timebin_new: int = None,
    datetime_begin: pd.Timestamp = None,
    datetime_end: pd.Timestamp = None,
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
    file : Path
        A Path to the detection file
    timebin_new: int
        The new time resolution to set the detections to (in seconds)
    datetime_begin: pd.Timestamp
        A datetime to filter out detections anterior to the datetime
    datetime_end: pd.Timestamp
        A datetime to filter out detections posterior to the datetime
    annotator: str
        A string to filter only detections of a particular annotator
    annotation: str
        A string to filter only detections of a particular annotation
    box: bool, default False
        if True, all annotations are kept, else keeps only absence/presence boxes (weak detection)
    timestamp_file: Path
        A Path to an APLOSE formatted timestamp file.
        It is used to create a reshaped detection file with timestamps that matches the APLOSE annotations.
    user_sel: str, default "all"
        A string to filter detections of a file based on annotators
            'union': the common detections of all annotators and the unique detections of each annotator are selected
            'intersection': only the common detections of all annotators are selected
            'all': all the detections are selected
    fmin_filter: int
        An integer to filter out detections based on a minimum frequency
    fmax_filter: int
        An integer to filter out detections based on a maximum frequency

    Returns
    -------
    result_df: pd.DataFrame
        A DataFrame corresponding to the selected filters and containing all the corresponding detections
    """
    delimiter = find_delimiter(file)

    df = (
        pd.read_csv(file, sep=delimiter, parse_dates=["start_datetime", "end_datetime"])
        .sort_values("start_datetime")
        .reset_index(drop=True)
    )

    df = df.dropna(subset=["annotation"])  # drop lines with only comments

    list_annotators = df["annotator"].drop_duplicates().to_list()
    list_labels = df["annotation"].drop_duplicates().to_list()
    max_freq = int(max(df["end_frequency"]))
    max_time = int(max(df["end_time"]))

    if datetime_begin:
        df = df[df["start_datetime"] >= datetime_begin]
        if len(df) == 0:
            raise Exception(
                f"No detection found after 'datetime_begin' filtering at '{datetime_begin}', upload aborted"
            )

    if datetime_end:
        df = df[df["end_datetime"] <= datetime_end]
        if len(df) == 0:
            raise Exception(
                f"No detection found after 'datetime_end' filtering at '{datetime_end}', upload aborted"
            )

    if annotator:
        if annotator not in list_annotators:
            raise ValueError(
                f"Annotator '{annotator}' is not present in result file annotators, upload aborted"
            )
        df = df.loc[(df["annotator"] == annotator)]
        list_annotators = [annotator]

    if annotation:
        if annotation not in list_labels:
            raise ValueError(
                f"Annotation '{annotation}' is not present in result file labels, upload aborted"
            )
        df = df.loc[(df["annotation"] == annotation)]

    if fmin_filter:
        df = df[df["start_frequency"] >= fmin_filter]
        if len(df) == 0:
            raise Exception(
                f"No detection found after fmin filtering at {fmin_filter}Hz, upload aborted"
            )

    if fmax_filter:
        df = df[df["end_frequency"] <= fmax_filter]
        if len(df) == 0:
            raise Exception(
                f"No detection found after fmax filtering at {fmax_filter}Hz, upload aborted"
            )

    df_no_box = df.loc[
        (df["start_time"] == 0)
        & (df["end_time"] == max_time)
        & (df["end_frequency"] == max_freq)
    ]

    if box is False:
        if len(df_no_box) == 0 or timebin_new is not None:
            df = reshape_timebin(
                df=df,
                timebin_new=timebin_new,
                timestamp=timestamp_file,
            )
        else:
            df = df_no_box

    if len(list_annotators) > 1 and user_sel in ["union", "intersection"]:
        df = intersection_or_union(df=df, user_sel=user_sel)

    return df.sort_values("start_datetime").reset_index(drop=True)


def intersection_or_union(df: pd.DataFrame, user_sel: str) -> pd.DataFrame:
    annotators = df["annotator"].drop_duplicates().to_list()
    if not len(annotators) > 1:
        raise ValueError('Not enough annotators detected')

    if user_sel not in ['intersection', 'union']:
        raise ValueError("'user_sel' must be either 'intersection' or 'union'")

    labels = df["annotation"].drop_duplicates().to_list()

    df_inter = pd.DataFrame()
    df_diff = pd.DataFrame()
    for label in labels:
        df_label = df[df["annotation"] == label]
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
        df_inter['annotator'] = [" ∩ ".join(annotators)] * len(df_inter)
        return df_inter.sort_values("start_datetime").reset_index(drop=True)
    elif user_sel == "union":
        df_union = pd.concat([df_diff, df_inter]).reset_index(drop=True)
        df_union['annotator'] = [" ∪ ".join(annotators)] * len(df_union)
        return df_union.sort_values("start_datetime").reset_index(drop=True)


def read_yaml(file: Path) -> dict:
    """Reads yaml file to extract detection parameters. The extracted parameters
    are then used to import detections using 'sorting_detection'.

    Parameters
    ----------
        file: Path
            A path to the yaml file

    Returns
    -------
        parameters: dict
            Dictionary containing a set of parameters for each csv file
    """
    with open(file, "r") as yaml_file:
        parameters = yaml.safe_load(yaml_file)

    for filename in parameters.keys():

        if not Path(filename).exists():
            raise FileNotFoundError(f"'{filename}' does not exist")
        else:
            parameters[filename]["file"] = Path(filename)

        if parameters[filename]["timebin_new"] and not isinstance(
            parameters[filename]["timebin_new"], int
        ):
            raise ValueError(
                f"An integer must be passed to 'timebin_new', '{parameters[filename]['timebin_new']}' not a valid value."
            )

        if parameters[filename]["datetime_format"] and not is_datetime_template_valid(
            parameters[filename]["datetime_format"]
        ):
            raise ValueError(
                f"'{parameters[filename]['timebin_new']}' must be a valid datetime format."
            )

        if parameters[filename]["fmin_filter"] and not isinstance(
            parameters[filename]["fmin_filter"], int
        ):
            raise ValueError(
                f"An integer must be passed to 'fmin_filter', '{parameters[filename]['fmin_filter']}' not a valid value."
            )

        if parameters[filename]["fmax_filter"] and not isinstance(
            parameters[filename]["fmax_filter"], int
        ):
            raise ValueError(
                f"An integer must be passed to 'fmax_filter', '{parameters[filename]['fmax_filter']}' not a valid value."
            )

        if parameters[filename]["datetime_begin"]:
            try:
                parameters[filename]["datetime_begin"] = pd.Timestamp(
                    parameters[filename]["datetime_begin"]
                )
            except ValueError as e:
                raise ValueError(
                    f"Invalid date format for 'datetime_begin': '{parameters[filename]['datetime_begin']}'"
                ) from e

        if parameters[filename]["datetime_end"]:
            try:
                parameters[filename]["datetime_end"] = pd.Timestamp(
                    parameters[filename]["datetime_end"]
                )
            except ValueError as e:
                raise ValueError(
                    f"Invalid date format for 'datetime_end': {parameters[filename]['datetime_end']}"
                ) from e

        if (
            all(
                [
                    parameters[filename]["datetime_begin"],
                    parameters[filename]["datetime_end"],
                ]
            )
            and parameters[filename]["datetime_begin"]
            >= parameters[filename]["datetime_end"]
        ):
            raise ValueError(
                f'{parameters[filename]["datetime_begin"]} >= {parameters[filename]["datetime_end"]}'
            )

        if parameters[filename]["annotator"] and not isinstance(
            parameters[filename]["annotator"], str
        ):
            raise ValueError(
                f"A string must be passed to 'annotator', '{parameters[filename]['annotator']}' not a valid value."
            )

        if parameters[filename]["annotation"] and not isinstance(
            parameters[filename]["annotation"], str
        ):
            raise ValueError(
                f"A string must be passed to 'annotation', '{parameters[filename]['annotation']}' not a valid value."
            )

        if parameters[filename]["box"] and not isinstance(
            parameters[filename]["box"], bool
        ):
            raise ValueError(
                f"A boolean must be passed to 'box', '{parameters[filename]['box']}' not a valid value."
            )

        if parameters[filename]["user_sel"] and parameters[filename][
            "user_sel"
        ] not in ["union", "intersection", "all"]:
            raise ValueError(
                f"Either 'union', 'intersection' or 'all' must be passed to 'user_sel', '{parameters[filename]['user_sel']}' not a valid value."
            )

        if parameters[filename]["timestamp_file"]:
            if not Path(parameters[filename]["timestamp_file"]).exists():
                raise FileNotFoundError(
                    f"'{parameters[filename]['timestamp_file']}' does not exist"
                )
            else:
                parameters[filename]["timestamp_file"] = Path(
                    parameters[filename]["timestamp_file"]
                )

    return parameters


def find_delimiter(file: str | Path) -> str:
    """Finds the proper delimiter for a csv file

    Parameters
    ----------
    file: Path
        A Path to a csv file

    Returns
    -------
    delimiter: str
        The delimiter to use to read the file
    """
    with open(file, "r", newline="") as csv_file:
        try:
            temp_lines = csv_file.readline() + "\n" + csv_file.readline()
            dialect = csv.Sniffer().sniff(temp_lines, delimiters=",;")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        return delimiter


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
        second = t.second
        second = round(second / 10) * 10
        t = t.replace(second=second, microsecond=0)
    elif res == 60:  # 1min
        second = round(t.second / 10) * 10
        if second < 60:
            t = t.replace(second=0, microsecond=0)
        else:
            t = t + pd.Timedelta(minutes=1)
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


def export2raven(
    df,
    tuple_info,
    timebin_new,
    bin_height,
    selection_vec: bool = False,
    offset: bool = False,
) -> pd.DataFrame:
    """Export a given vector to Raven formatted table

    Parameters
    ----------
    offset
    df : dataframe of the detections
    timebin_new : int, duration of the detection boxes to export, if set to 0, the original detections are exported
    bin_height : the maximum frequency of the exported time bins
    tuple_info : tuple containing info such as the filenames of the wav files, their durations and datetimes
    selection_vec : if it is set to False, all the time bins are exported, else the selection_vec is used to select the wanted time bins to export, for instance it corresponds to all the positives time bins, containing detections
    """
    file_list = list(tuple_info[0])
    file_datetimes = tuple_info[1]
    dur = list(tuple_info[2])

    offsets = [
        (file_datetimes[i] + pd.Timedelta(seconds=dur[i])).timestamp()
        - (file_datetimes[i + 1]).timestamp()
        for i in range(len(file_datetimes) - 1)
    ]
    offsets_cumsum = list(np.cumsum([offsets[i] for i in range(len(offsets))]))
    offsets_cumsum.insert(0, 0)
    idx_wav_df = [file_list.index(df["filename"][i]) for i in range(len(df))]

    if timebin_new > 0:
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

        df_pg2raven = pd.DataFrame()
        df_pg2raven["Selection"] = np.arange(1, len(df_time_sorted) + 1)
        df_pg2raven["View"], df_pg2raven["Channel"] = [1] * len(df_time_sorted), [
            1
        ] * len(df_time_sorted)
        df_pg2raven["Begin Time (s)"] = df_time_sorted["start"]
        df_pg2raven["End Time (s)"] = df_time_sorted["end"]
        df_pg2raven["Low Freq (Hz)"] = [0] * len(df_time_sorted)
        df_pg2raven["High Freq (Hz)"] = [bin_height] * len(df_time_sorted)

    else:
        start_time = [
            df["start_time"][i] + offsets_cumsum[idx_wav_df[i]] for i in range(len(df))
        ]
        end_time = [
            df["end_time"][i] + offsets_cumsum[idx_wav_df[i]] for i in range(len(df))
        ]

        df_pg2raven = pd.DataFrame()
        df_pg2raven["Selection"] = np.arange(1, len(df) + 1)
        df_pg2raven["View"], df_pg2raven["Channel"] = [1] * len(df), [1] * len(df)
        df_pg2raven["Begin Time (s)"] = start_time
        df_pg2raven["End Time (s)"] = end_time
        df_pg2raven["Low Freq (Hz)"] = df["start_frequency"]
        df_pg2raven["High Freq (Hz)"] = df["end_frequency"]

    if offset is True:
        df_offset = pd.DataFrame(
            {"filename": file_list, "offset_cumsum": offsets_cumsum}
        )
        return df_pg2raven, df_offset
    else:
        return df_pg2raven, None


def get_season(ts: pd.Timestamp) -> str:
    """'day of year' ranges for the northern hemisphere

    Parameter
    ---------
        ts: pd.Timestamp

    Returns
    -------
        season: string
            The season and year of ts

    Example
    -------
    get_season(pd.Timestamp("01/01/2023"))
    >>> 'winter 2022'
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
    else:
        raise ValueError("Invalid timestamp")

    return season


def input_date(msg):
    """Based on selection_type, ask the user a folder and yields all the wav files inside it or ask the user multiple wav files
    Parameters :
        msg : Message to tell the user what date they have to enter (begin, end...)
    Returns :
        date_dt : aware datetime entered by the user
    """

    title = "Date"
    field_names = [
        "Year [YYYY]",
        "Month [m]",
        "Day [d]",
        "Hour [H]",
        "Minute [M]",
        "Second [S]",
        "Timezone [+/-HHMM]",
    ]
    field_values = []  # Initialize with empty values

    while True:
        field_values = easygui.multenterbox(msg, title, field_names, field_values)

        if field_values is None:
            # User canceled the input
            return None

        errmsg = ""
        for i in range(len(field_names)):
            if field_values[i].strip() == "":
                errmsg += f"'{field_names[i]}' is a required field.\n"

        if errmsg == "":
            break  # No validation errors

        easygui.msgbox(errmsg, title)

    year, month, day, hour, minute, second = map(int, field_values[:-1])
    hours_offset = int(field_values[-1][:3])
    minutes_offset = int(field_values[-1][3:])
    tz = pytz.FixedOffset(hours_offset * 60 + minutes_offset)

    date_dt = pd.Timestamp(year, month, day, hour, minute, second, tzinfo=tz)
    return date_dt


def suntime_hour(start: pd.Timestamp, stop: pd.Timestamp, lat: float, lon: float):
    """Fetch sunrise and sunset hours for dates between date_beg and date_end

    Parameters
    ----------
        start: pd.Timestamp
            start datetime of when to fetch sun hour
        stop: pd.Timestamp
            end datetime of when to fetch sun hour
        lat: float
            latitude in decimal degrees
        lon: float
            longitude in decimal degrees

    Returns
    -------
        hour_sunrise: list
            A list of float with sunrise decimal hours for each day between date_beg and date_end
        hour_sunset: list
            A List of float with sunset decimal hours for each day between date_beg and date_end
    """
    tz = start.tz

    # localisation info
    gps = astral.LocationInfo(timezone=tz, latitude=lat, longitude=lon)

    # List of days during when the data were recorded
    h_sunrise, h_sunset, dt_dusk, dt_dawn, dt_day, dt_night = [], [], [], [], [], []

    # For each day : find time of sunset, sun rise, begin dawn and dusk
    for date in [
        ts.date() for ts in pd.date_range(start.normalize(), stop.normalize(), freq="D")
    ]:

        # nautical twilight = 12, see def here : https://www.timeanddate.com/astronomy/nautical-twilight.html
        suntime = sun(gps.observer, date=date, dawn_dusk_depression=12)
        dawn, day, _, dusk, night = [
            pd.Timestamp(suntime[period]).tz_convert(tz) for period in suntime
        ]

        for lst, period in zip([h_sunrise, h_sunset], [day, night]):
            lst.append(period.hour + period.minute / 60 + period.second / 3600)

        for lst, period in zip(
            [dt_dawn, dt_day, dt_dusk, dt_night], [dawn, day, dusk, night]
        ):
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
                errmsg += (
                    f"'{lat}' is not a valid latitude. It must be between -90 and 90.\n"
                )
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


def get_duration(
    title: str = "Get duration",
    msg: str = "Enter a time alias",
    default: str = "10min",
    base: bool = False,
):
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
        For instance, "10min" => '<Minute>'
    Returns
    -------
    The total number of seconds of the entered time alias.
    The smallest duration is 1s.
    """
    value = easygui.enterbox(
        msg=f"{msg}", title=f"{title}", default=f"{default}", strip=True
    )

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

        value = easygui.enterbox(msg=errmsg, title=f"{title}", strip=True)

    if base:
        return value, base_str
    else:
        return value


def get_datetime_format(
    title: str = "Get datetime format",
    msg: str = "Enter a datetime format code",
    default: str = "%d/%m/%Y\n%H:%M",
) -> str:
    """Ask user input to get datetime format.
    Datetime format codes are to be used,
    See https://docs.python.org/fr/3/library/datetime.html
    Parameters
    ----------
    title : str
    msg : str
    default : '%d/%m/%Y\n%H:%M'
    """
    fmt = easygui.enterbox(
        msg=f"{msg}", title=f"{title}", default=f"{default}", strip=True
    )

    while True:
        if fmt is None:
            raise TypeError("'get_duration()' was cancelled")

        errmsg = ""
        datetime_test = pd.Timestamp("now")
        try:
            datetime_test.strftime(format=fmt)
        except ValueError:
            errmsg = f"'{fmt}' is not a valid datetime format code."

        if errmsg == "":
            break

        fmt = easygui.enterbox(msg=errmsg, title=f"{title}", strip=True)

    return fmt


def stats_diel_pattern(
    df_detections: pd.DataFrame,
    begin_date: pd.Timestamp,
    end_date: pd.Timestamp,
    lat: float = None,
    lon: float = None,
):
    """Plots detection proportions for each light regime (night/dawn/day/dawn)

    Parameters
    ----------
        df_detections: pd.DataFrame
            An APLOSE result DataFrame
        begin_date: pd.Timestamp
            A beginning datetime of data to analyse
        end_date: pd.Timestamp
            An end datetime of data to analyse
        lat: float
            A latitude in Decimal Degrees
        lon: float
            A longitude in Decimal Degrees

    Returns
    -------
        lr: pd.DataFrame
            df used to plot the detections
        box_name: list
            A list of light regimes
    """
    if not isinstance(lat, float) and not isinstance(lat, int) and lat is not None:
        raise ValueError("Invalid latitude")
    elif not isinstance(lon, float) and not isinstance(lon, int) and lon is not None:
        raise ValueError("Invalid longitude")
    else:
        lat, lon = get_coordinates()

    # Compute sunrise and sunset decimal hour at the dataset location
    # Seems to only work with UTC data ?
    [_, _, dt_dusk, dt_dawn, dt_day, dt_night] = suntime_hour(begin_date, end_date, lat, lon)

    # List of days in the dataset
    list_days = [d.date() for d in dt_day]

    # Compute dusk_duration, dawn_duration, day_duration, night_duration
    dawn_duration = [b - a for a, b in zip(dt_dawn, dt_day)]
    day_duration = [b - a for a, b in zip(dt_day, dt_night)]
    dusk_duration = [b - a for a, b in zip(dt_night, dt_dusk)]
    night_duration = [
        pd.Timedelta(hours=24) - dawn - day - dusk
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
            # If the detection occurred during 'day'
            if d == day:
                if (
                        dt_dawn[idx_day] < df_detections["start_datetime"][idx_det] < dt_day[idx_day]
                ):
                    lr = 2
                    light_regime.append(lr)
                elif (
                        dt_day[idx_day] < df_detections["start_datetime"][idx_det] < dt_night[idx_day]
                ):
                    lr = 3
                    light_regime.append(lr)
                elif (
                        dt_night[idx_day] < df_detections["start_datetime"][idx_det] < dt_dusk[idx_day]
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
        # Find index of detections that occurred during 'day'
        idx_det = [idx for idx, det in enumerate(day_det) if det == day]
        if not idx_det:
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

    # For each day :  compute number of detection per light regime corrected by light regime duration
    nb_det_night_corr = [(nb / d) for nb, d in zip(nb_det_night, night_duration_dec)]
    nb_det_dawn_corr = [(nb / d) for nb, d in zip(nb_det_dawn, dawn_duration_dec)]
    nb_det_day_corr = [(nb / d) for nb, d in zip(nb_det_day, day_duration_dec)]
    nb_det_dusk_corr = [(nb / d) for nb, d in zip(nb_det_dusk, dusk_duration_dec)]

    # Normalize by daily average number of detection per hour
    av_daily_nb_det = []
    nb_det_night_corr_norm = []
    nb_det_dawn_corr_norm = []
    nb_det_day_corr_norm = []
    nb_det_dusk_corr_norm = []

    for idx_day, day in enumerate(list_days):
        # Find index of detections that occurred during 'day'
        idx_det = [idx for idx, det in enumerate(day_det) if det == day]
        # Compute daily average number of detections per hour
        a = len(idx_det) / 24
        av_daily_nb_det.append(a)
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

    light_regime = [
        nb_det_night_corr_norm,
        nb_det_dawn_corr_norm,
        nb_det_day_corr_norm,
        nb_det_dusk_corr_norm,
    ]
    box_name = ["Night", "Dawn", "Day", "Dusk"]

    lr = pd.DataFrame(light_regime, index=box_name).transpose()

    return lr, box_name


def stat_box_day(
    data_test: pd.DataFrame, df_detections: pd.DataFrame, detector: str
) -> pd.DataFrame:
    """Plot detection proportions for each hour of the day

    Parameters
    ----------
        data_test: df with data infos
        df_detections: APLOSE formatted df of the detections
        detector: name of the automatic detector to use

    Returns
    -------
        result: df used to plot the detections
    """
    hour_list = ["{:02d}:00".format(i) for i in range(24)]
    hour_list.append("00:00")

    df_detections["date"] = [date.strftime("%d/%m/%Y") for date in df_detections["start_datetime"]]
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
        detection_by_date = df_detections[
            df_detections["date"] == date
        ]  # sub-dataframe : per date
        list_datasets = sorted(
            list(set(detection_by_date["dataset"]))
        )  # dataset list for date=date

        for dataset in list_datasets:
            df = detection_by_date[detection_by_date["dataset"] == dataset].set_index(
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

            # list_present_h = [
            #     dt.datetime.fromtimestamp(i)
            #     for i in list(range(deploy_beg_ts, deploy_end_ts, 3600))
            # ]
            list_present_h = pd.date_range(start=pd.to_datetime(deploy_beg_ts, unit='s'),
                                           end=pd.to_datetime(deploy_end_ts, unit='s'),
                                           freq='H').tolist()
            list_present_h2 = [
                list_present_h[i].strftime("%d/%m/%Y %H")
                for i in range(len(list_present_h))
            ]

            # list_deploy_d = sorted(
            #     list(
            #         set(
            #             [
            #                 dt.datetime.strftime(
            #                     dt.datetime.fromtimestamp(i), "%d/%m/%Y"
            #                 )
            #                 for i in list(range(deploy_beg_ts, deploy_end_ts, 3600))
            #             ]
            #         )
            #     )
            # )
            list_deploy_d = sorted(
                pd.date_range(start=pd.to_datetime(deploy_beg_ts, unit='s'),
                              end=pd.to_datetime(deploy_end_ts, unit='s'),
                              freq='H')
                .strftime("%d/%m/%Y")
                .unique()
                .tolist()
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


def print_spectro_from_audio(
    file: Path,
    nfft: int = 1024,
    window_size: int = 1024,
    overlap: int = 20,
    ax: bool = True,
):
    """Computes and prints a spectrogram from an audio file.

    Parameters
    ----------
    file: Path to the audio file
    nfft
    window_size
    overlap
    ax: bool, show axes based on this value

    Examples
    --------
    audio_file = Path(r'path/to/file')
    print_spectro_from_audio(audio_file)
    """
    if not is_supported_audio_format(file):
        raise ValueError("Audio file format is not supported")

    try:
        sr, data = wavfile.read(file)
    except ValueError as e:
        print(e)

    overlap_samples = int(overlap / 100 * window_size)  # overlap in samples

    frequencies, times, sxx = spectrogram(
        data, fs=sr, nperseg=window_size, noverlap=overlap_samples, nfft=nfft
    )

    my_dpi = 200
    fact_x = 1.3
    fact_y = 1.3
    fig, _ = plt.subplots(
        figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
        dpi=my_dpi,
    )

    plt.pcolormesh(times, frequencies, 10 * np.log10(sxx))

    if not ax:
        plt.axis("off")
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
        )  # delete white borders
    else:
        plt.tight_layout()

    plt.show()

    ech = len(data)
    size_x = (ech - window_size) / overlap_samples
    size_y = nfft / 2
    print(f"X: {size_x:.3f}\nY: {size_y:.3f}")

    return


def print_spectro_from_npz(file: Path, ax: bool = True):
    """Computes and prints a spectrogram from a npz file.

    Parameters
    ----------
    file: Path to the npz file
    ax: bool, show axes based on this value

    Examples
    --------
    npz_file = Path(r'path/to/file')
    print_spectro_from_npz(npz_file)
    """
    if not file.suffix == ".npz":
        raise ValueError("NPZ file format must be provided")

    try:
        with np.load(file, allow_pickle=True) as data:
            sxx = data["Sxx"]
            freq = data["Freq"]
            time = data["Time"]
    except ValueError as e:
        print(e)

    my_dpi = 200
    fact_x = 1.3
    fact_y = 1.3
    fig, _ = plt.subplots(
        figsize=(fact_x * 1800 / my_dpi, fact_y * 512 / my_dpi),
        dpi=my_dpi,
    )

    plt.pcolormesh(time, freq, 10 * np.log10(sxx))

    if not ax:
        plt.xticks([], [])
        plt.yticks([], [])
        plt.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    else:
        plt.tight_layout()

    plt.show()

    return
