from __future__ import annotations

import bisect
import csv
import json
from pathlib import Path

import astral
import easygui
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from astral.sun import sun
from matplotlib.axes import Axes
from OSmOSE.config import TIMESTAMP_FORMAT_AUDIO_FILE
from OSmOSE.utils.timestamp_utils import strptime_from_text
from pandas import DateOffset, Timestamp, date_range
from pandas.tseries.frequencies import to_offset
from scipy.signal import spectrogram


def reshape_timebin(
    df: pd.DataFrame,
    timebin_new: int | None = None,
    timestamp: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Reshape an APLOSE result DataFrame according to a new time bin.

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
    df_sorted = df.sort_values("start_datetime").reset_index(drop=True)
    annotators = df_sorted["annotator"].drop_duplicates().to_list()
    labels = df_sorted["annotation"].drop_duplicates().to_list()
    max_freq = int(max(df_sorted["end_frequency"]))
    tz_data = df_sorted["start_datetime"][0].tz

    if not timebin_new:
        timebin_new = get_duration(
            title="Get duration",
            msg="Enter a new time bin",
            default="1min",
        )
    frequency = str(timebin_new) + "s"

    df_new_timebin = pd.DataFrame()
    for annotator in annotators:
        for label in labels:
            df_1annot_1label = df_sorted[
                (df_sorted["annotator"] == annotator)
                & (df_sorted["annotation"] == label)
            ]

            if len(df_1annot_1label) == 0:
                continue

            if timestamp is not None:
                origin_timebin = (timestamp[1] - timestamp[0]).total_seconds()
                time_vector = timestamp[0 :: int(timebin_new / origin_timebin)]
            else:
                t1 = t_rounder(
                    t=min(df_1annot_1label["start_datetime"]),
                    res=timebin_new,
                )
                t2 = t_rounder(
                    t=max(df_1annot_1label["end_datetime"]),
                    res=timebin_new,
                )
                time_vector = pd.date_range(start=t1, end=t2, freq=frequency)

            ts_detect_beg = list(df_1annot_1label["start_datetime"])
            ts_detect_end = list(df_1annot_1label["end_datetime"])
            filenames = list(df_1annot_1label["filename"])

            filename_vector = []
            for ts in time_vector:
                """
                insertion of ts in ts_detect_beg, `bisect_left` provides
                the index of the element in ts_detect_beg that is closest
                to ts (left element if between 2 elements of the list).
                """
                index = bisect.bisect_left(ts_detect_beg, ts)
                if index == 0:
                    filename_vector.append(filenames[index])
                else:
                    (
                        filename_vector.append(filenames[index])
                        if ts in ts_detect_beg
                        else filename_vector.append(filenames[index - 1])
                    )

            detect_vec = [0] * len(time_vector)
            for i in range(len(df_1annot_1label)):
                idx = bisect.bisect_left(time_vector, ts_detect_beg[i])

                if ts_detect_beg[i] in time_vector:
                    rank = idx
                else:
                    rank = max(0, idx - 1)

                inc = 0
                while (rank + inc) < len(time_vector) and time_vector[rank + inc] < ts_detect_end[i]:
                    detect_vec[rank + inc] = 1
                    inc += 1

            start_datetime, end_datetime, filename = [], [], []
            for i in range(len(time_vector)):
                if detect_vec[i] == 1:
                    start_datetime.append(time_vector[i])
                    end_datetime.append(time_vector[i] + pd.Timedelta(timebin_new, 's'))
                    filename.append(filename_vector[i])

            df_1annot_1label_new_timebin = pd.DataFrame()
            df_1annot_1label_new_timebin["dataset"] = [
                df_1annot_1label["dataset"].iloc[0],
            ] * len(start_datetime)
            df_1annot_1label_new_timebin["filename"] = filename
            df_1annot_1label_new_timebin["start_time"] = [0] * len(start_datetime)
            df_1annot_1label_new_timebin["end_time"] = [timebin_new] * len(
                start_datetime,
            )
            df_1annot_1label_new_timebin["start_frequency"] = [0] * len(start_datetime)
            df_1annot_1label_new_timebin["end_frequency"] = [max_freq] * len(
                start_datetime,
            )
            df_1annot_1label_new_timebin["annotation"] = [label] * len(start_datetime)
            df_1annot_1label_new_timebin["annotator"] = [annotator] * len(
                start_datetime,
            )
            df_1annot_1label_new_timebin["start_datetime"] = start_datetime
            df_1annot_1label_new_timebin["end_datetime"] = end_datetime

            df_new_timebin = pd.concat([df_new_timebin, df_1annot_1label_new_timebin])
            df_new_timebin["is_box"] = [0] * len(df_new_timebin)

    return df_new_timebin.sort_values(by=["start_datetime"])


def load_detections(
    file: Path,
    timebin_new: int | None = None,
    datetime_begin: pd.Timestamp = None,
    datetime_end: pd.Timestamp = None,
    annotator: str | None = None,
    annotation: str | None = None,
    timestamp_file: Path | None = None,
    user_sel: str = "all",
    f_min: int | None = None,
    f_max: int | None = None,
    score: float | None = None,
    *,
    box: bool = False,
) -> pd.DataFrame:
    """Load and filter an APLOSE-formatted detection file.

    Parameters
    ----------
    file : Path
        Detection file.

    timebin_new: int
        The new time resolution to set the detections to, in seconds.

    datetime_begin: pd.Timestamp
        To filter out detections anterior to the argument.

    datetime_end: pd.Timestamp
        To filter out detections posterior to the argument.

    annotator: str
       To filter only detections of an annotator.

    annotation: str
        To filter only detections of an annotation.

    box: bool, default False
        if True, all annotations are kept,
        else keeps only absence/presence boxes (weak detection)

    timestamp_file: Path
        APLOSE formatted timestamp file.
        It is used to create a reshaped detection file
        with timestamps that matches the APLOSE annotations.

    user_sel: str, default "all"
        A string to filter detections of a file based on annotators.
            -'union': the common detections of all annotators and
            the unique detections of each annotator are selected;
            -'intersection': only the common detections of all annotators are selected;
            -'all': all the detections are selected.

    f_min: int
        To filter out detections inferior to the argument.

    f_max: int
        To filter out detections superior to the argument.

    score: float
        To filter out detections with score lower than the argument.

    Returns
    -------
    A DataFrame with detections corresponding to selected filters

    """
    delimiter = find_delimiter(file)

    df_loaded = (
        pd.read_csv(file, sep=delimiter, parse_dates=["start_datetime", "end_datetime"])
        .sort_values("start_datetime")
        .reset_index(drop=True)
    )

    df_loaded = df_loaded.dropna(subset=["annotation"])  # drop lines with only comments

    list_annotators = df_loaded["annotator"].drop_duplicates().to_list()
    list_labels = df_loaded["annotation"].drop_duplicates().to_list()
    max_freq = int(max(df_loaded["end_frequency"]))
    max_time = int(max(df_loaded["end_time"]))

    if datetime_begin:
        df_loaded = df_loaded[df_loaded["start_datetime"] >= datetime_begin]
        if len(df_loaded) == 0:
            msg = f"No detection found after '{datetime_begin}', upload aborted"
            raise ValueError(msg)

    if datetime_end:
        df_loaded = df_loaded[df_loaded["end_datetime"] <= datetime_end]
        if len(df_loaded) == 0:
            msg = f"No detection found before '{datetime_end}', upload aborted"
            raise ValueError(msg)

    if annotator:
        if isinstance(annotator, list):
            invalid_annotators = [a for a in annotator if a not in list_annotators]
            if invalid_annotators:
                msg = (
                    f"'{invalid_annotators}' not present in annotators, upload aborted"
                )
                raise ValueError(msg)
            df_loaded = df_loaded.loc[df_loaded["annotator"].isin(annotator)]
            list_annotators = annotator
        else:
            if annotator not in list_annotators:
                msg = f"'{annotator}' not present in annotators, upload aborted"
                raise ValueError(msg)
            df_loaded = df_loaded.loc[df_loaded["annotator"] == annotator]
            list_annotators = [annotator]

    if annotation:
        if isinstance(annotation, list):
            invalid_annotations = [a for a in annotation if a not in list_labels]
            if invalid_annotations:
                msg = f"'{invalid_annotations}' not present in labels, upload aborted"
                raise ValueError(msg)
            df_loaded = df_loaded.loc[df_loaded["annotation"].isin(annotation)]
        else:
            if annotation not in list_labels:
                msg = f"'{annotation}' not present in labels, upload aborted"
                raise ValueError(msg)
            df_loaded = df_loaded.loc[df_loaded["annotation"] == annotation]

    if f_min:
        df_loaded = df_loaded[df_loaded["start_frequency"] >= f_min]
        if len(df_loaded) == 0:
            msg = f"No detection found above {f_min}Hz, upload aborted"
            raise ValueError(msg)

    if f_max:
        df_loaded = df_loaded[df_loaded["end_frequency"] <= f_max]
        if len(df_loaded) == 0:
            msg = f"No detection found below {f_max}Hz, upload aborted"
            raise ValueError(msg)

    if score and 'score' in df_loaded.columns:
        df_loaded = df_loaded[df_loaded["score"] >= score]
        if len(df_loaded) == 0:
            msg = f"No detection found above with score above {score}, upload aborted"
            raise ValueError(msg)

    df_no_box = df_loaded.loc[
        (df_loaded["start_time"] == 0)
        & (df_loaded["end_time"] == max_time)
        & (df_loaded["end_frequency"] == max_freq)
    ]

    if not box:
        if len(df_no_box) == 0 or timebin_new is not None:
            if timestamp_file:
                timestamp = (
                    pd.read_csv(timestamp_file, parse_dates=["timestamp"])
                    .drop_duplicates()
                    .reset_index(drop=True)["timestamp"]
                )
            else:
                timestamp = None

            df_loaded = reshape_timebin(
                df=df_no_box,
                timebin_new=timebin_new,
                timestamp=timestamp,
            )
        else:
            df_loaded = df_no_box

    if len(list_annotators) > 1 and user_sel in ["union", "intersection"]:
        df_loaded = intersection_or_union(df=df_loaded, user_sel=user_sel)

    return df_loaded.sort_values("start_datetime").reset_index(drop=True)


def intersection_or_union(df: pd.DataFrame, user_sel: str) -> pd.DataFrame:
    """Compute the intersection or union of annotations from multiple annotators.

    This function identifies common and differing annotations based on the
    "start_datetime" values in the dataset. The intersection consists of
    annotations that appear in the data for all annotators, while the union
    includes all annotations regardless of overlap.

    Parameters
    ----------
    df : pd.DataFrame
        An APLOSE result DataFrame

    user_sel : str
        Specifies whether to return the "intersection" (annotations shared
        by all annotators) or the "union" (all annotations from all annotators).
        Accepted values are:
        - "intersection": Returns only annotations that appear in all annotators' data.
        - "union": Returns all annotations, including both shared and unique ones.

    Returns
    -------
    pd.DataFrame
        An APLOSE formatted DataFrame containing the selected annotations:
        - If "intersection" is chosen, the output includes only annotations
          present in all annotators' data,
          with the annotator column merged as "annotator1 ∩ annotator2".
        - If "union" is chosen, the output includes all annotations,
          with the annotator column merged as "annotator1 ∪ annotator2".

    """
    annotators = df["annotator"].drop_duplicates().to_list()
    if not len(annotators) > 1:
        msg = "Not enough annotators detected"
        raise ValueError(msg)

    if user_sel not in ["intersection", "union"]:
        msg = "'user_sel' must be either 'intersection' or 'union'"
        raise ValueError(msg)

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
        df_inter["annotator"] = [" ∩ ".join(annotators)] * len(df_inter)
        df_result = df_inter.sort_values("start_datetime").reset_index(drop=True)
    else:
        df_union = pd.concat([df_diff, df_inter]).reset_index(drop=True)
        df_union["annotator"] = [" ∪ ".join(annotators)] * len(df_union)
        df_result = df_union.sort_values("start_datetime").reset_index(drop=True)

    return df_result


def read_yaml(file: Path) -> dict:
    """Read yaml file to extract detection parameters.

    The extracted parameters are then used to import
    detections using 'sorting_detection'.

    Parameters
    ----------
    file: Path
        Yaml file.

    Returns
    -------
    dict
        Dictionary containing a set of parameters for each csv file

    """
    with file.open(encoding="utf-8") as yaml_file:
        parameters = yaml.safe_load(yaml_file)

    for filename in parameters:
        if not Path(filename).exists():
            msg = f"'{filename}' does not exist"
            raise FileNotFoundError(msg)
        parameters[filename]["file"] = Path(filename)

        if parameters[filename]["timebin_new"] and not isinstance(
            parameters[filename]["timebin_new"],
            int,
        ):
            msg = f"'{parameters[filename]['timebin_new']}' not a valid value."
            raise ValueError(msg)

        if parameters[filename]["f_min"] and not isinstance(
            parameters[filename]["f_min"],
            int,
        ):
            msg = f"'{parameters[filename]['f_min']}' not a valid value."
            raise ValueError(msg)

        if parameters[filename]["f_max"] and not isinstance(
            parameters[filename]["f_max"],
            int,
        ):
            msg = f"'{parameters[filename]['f_max']}' not a valid value."
            raise ValueError(msg)

        if parameters[filename]["datetime_begin"]:
            try:
                parameters[filename]["datetime_begin"] = pd.Timestamp(
                    parameters[filename]["datetime_begin"],
                )
            except ValueError as e:
                msg = (
                    f"'datetime_begin', invalid format: '{parameters[filename]['datetime_begin']}'",
                )
                raise ValueError(msg) from e

        if parameters[filename]["datetime_end"]:
            try:
                parameters[filename]["datetime_end"] = pd.Timestamp(
                    parameters[filename]["datetime_end"],
                )
            except ValueError as e:
                msg = (
                    f"'datetime_end', invalid format: '{parameters[filename]['datetime_end']}'",
                )
                raise ValueError(msg) from e

        if (
            all(
                [
                    parameters[filename]["datetime_begin"],
                    parameters[filename]["datetime_end"],
                ],
            )
            and parameters[filename]["datetime_begin"]
            >= parameters[filename]["datetime_end"]
        ):
            msg = f'{parameters[filename]["datetime_begin"]} >= {parameters[filename]["datetime_end"]}'
            raise ValueError(msg)

        if parameters[filename]["annotator"] and not (
            isinstance(parameters[filename]["annotator"], str)
            or (
                isinstance(parameters[filename]["annotator"], list)
                and all(
                    isinstance(item, str) for item in parameters[filename]["annotator"]
                )
            )
        ):
            msg = f"'annotator', invalid value: '{parameters[filename]['annotator']}'"
            raise ValueError(msg)

        if parameters[filename]["annotation"] and not isinstance(
            parameters[filename]["annotation"],
            str,
        ):
            msg = f"'annotation', invalid value: '{parameters[filename]['annotation']}'"
            raise ValueError(msg)

        if parameters[filename]["box"] and not isinstance(
            parameters[filename]["box"],
            bool,
        ):
            msg = f"'box', invalid value: '{parameters[filename]['box']}'"
            raise ValueError(msg)

        if parameters[filename]["user_sel"] and parameters[filename][
            "user_sel"
        ] not in ["union", "intersection", "all"]:
            msg = f"'user_sel', invalid value: '{parameters[filename]['user_sel']}'"
            raise ValueError(msg)

        if parameters[filename]["timestamp_file"]:
            if not Path(parameters[filename]["timestamp_file"]).exists():
                msg = f"'{parameters[filename]['timestamp_file']}'"
                raise FileNotFoundError(msg)
            parameters[filename]["timestamp_file"] = Path(
                parameters[filename]["timestamp_file"],
            )

    return parameters


def find_delimiter(file: str | Path) -> str:
    """Find the proper delimiter for a csv file.

    Parameters
    ----------
    file: Path
        A Path to a csv file

    Returns
    -------
    delimiter: str
        The delimiter to use to read the file

    """
    with file.open(newline="") as csv_file:
        try:
            temp_lines = csv_file.readline() + "\n" + csv_file.readline()
            dialect = csv.Sniffer().sniff(temp_lines, delimiters=",;")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        return delimiter


def t_rounder(t: pd.Timestamp, res: pd.Timedelta | int) -> pd.Timestamp:
    """Round a Timestamp to the nearest resolution.

    Parameters
    ----------
    t: pd.Timestamp
        Datetime to round
    res: pd.Timedelta | int
        Resolution as a pandas Timedelta or an integer representing seconds

    Returns
    -------
    Rounded timestamp with preserved timezone

    """
    if isinstance(res, pd.Timedelta):
        res_seconds = res.total_seconds()
    elif isinstance(res, int) and res > 0:
        res_seconds = res
    else:
        msg = "Resolution must be a positive timedelta or a positive integer"
        raise ValueError(msg)

    tz = t.tzinfo
    rounded_epoch = round(t.timestamp() / res_seconds) * res_seconds
    rounded_t = pd.Timestamp.utcfromtimestamp(rounded_epoch)

    return rounded_t.tz_convert(tz) if tz is not None else rounded_t.tz_localize(None)


def get_season(ts: pd.Timestamp, northern: bool = True) -> str:
    """Determine the meteorological season.

    In the Northern hemisphere
    Winter: Dec–Feb, Spring: Mar–May, Summer: Jun–Aug, Autumn: Sep–Nov

    In the Southern hemisphere
    Winter: Jun–Aug, Spring: Sep–Nov, Summer: Dec–Feb, Autumn: Mar–May

    Parameters
    ----------
    northern: boolean, default True.
        Specify if the seasons are northern or austral
    ts: pd.Timestamp
        Considered datetime

    Returns
    -------
    The season and year of ts

    Example:
    -------
    >>> get_season(pd.Timestamp("01/01/2023"))

    """
    if northern:
        winter = [1, 2, 12]
        spring = [3, 4, 5]
        summer = [6, 7, 8]
        autumn = [9, 10, 11]
    else:
        winter = [6, 7, 8]
        spring = [9, 10, 11]
        summer = [1, 2, 12]
        autumn = [3, 4, 5]

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
        msg = "Invalid timestamp"
        raise ValueError(msg)

    return season


def get_sun_times(
    start: pd.Timestamp,
    stop: pd.Timestamp,
    lat: float,
    lon: float,
) -> (
    list[float],
    list[float],
    list[pd.Timestamp],
    list[pd.Timestamp],
    list[pd.Timestamp],
    list[pd.Timestamp],
):
    """Fetch sunrise and sunset hours for dates between start and stop.

    Each twilight phase is defined by the solar elevation angle,
    which is the position of the Sun in relation to the horizon.
    During nautical twilight, the geometric center of the Sun's disk
    is between 6 and 12 degrees below the horizon.
    See: https://www.timeanddate.com/astronomy/nautical-twilight.html

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
    hour_sunrise: list[float]
        Sunrise decimal hours for each day between start and stop

    hour_sunset: list[float]
        Sunset decimal hours for each day between start and stop

    dt_dawn: list[pd.Timestamp]
        Dawn datetimes for each day between start and stop

    dt_day: list[pd.Timestamp]
        Day datetimes for each day between start and stop

    dt_dusk: list[pd.Timestamp]
        Dusk datetimes for each day between start and stop

    dt_night: list[pd.Timestamp]
        Night datetimes for each day between start and stop

    """
    tz = start.tz

    # localisation info
    gps = astral.LocationInfo(latitude=lat, longitude=lon, timezone=tz)

    # List of days during when the data were recorded
    h_sunrise, h_sunset, dt_dusk, dt_dawn, dt_day, dt_night = [], [], [], [], [], []

    # For each day : find time of sunset, sun rise, begin dawn and dusk
    for date in [
        ts.date() for ts in pd.date_range(start.normalize(), stop.normalize(), freq="D")
    ]:
        suntime = sun(gps.observer, date=date, dawn_dusk_depression=12, tzinfo=tz)
        dawn, day, _, dusk, night = [
            pd.Timestamp(suntime[period]).tz_convert(tz) for period in suntime
        ]

        for lst, period in zip([h_sunrise, h_sunset], [day, dusk], strict=False):
            lst.append(period.hour + period.minute / 60 + period.second / 3600)

        for lst, period in zip(
            [dt_dawn, dt_day, dt_dusk, dt_night],
            [dawn, day, dusk, night],
            strict=False,
        ):
            lst.append(period)

    return h_sunrise, h_sunset, dt_dawn, dt_day, dt_dusk, dt_night


def get_coordinates() -> tuple:
    """Ask for user input to get GPS coordinates."""
    title = "Coordinates in degree° minute'"
    msg = "latitude (N/S) and longitude (E/W)"
    field_names = ["lat decimal degree", "lon decimal degree"]
    field_values = easygui.multenterbox(msg, title, field_names)

    max_lat = 90
    max_lon = 180

    # make sure that none of the fields was left blank
    while True:
        if field_values is None:
            msg = "'get_coordinates()' was cancelled"
            raise TypeError(msg)

        lat, lon = field_values
        errmsg = ""
        try:
            lat_val = float(lat.strip())  # Convert to float for latitude
            if lat_val < -max_lat or lat_val > max_lat:
                errmsg += (
                    f"'{lat}' is not a valid latitude. It must be between -90 and 90.\n"
                )
        except ValueError:
            errmsg += f"'{lat}' is not a valid entry for latitude.\n"

        try:
            lon_val = float(lon.strip())  # Convert to float for longitude
            if lon_val < -max_lon or lon_val > max_lon:
                errmsg += (
                    f"'lon', invalid entry: '{lon}'. It must be between -180 and 180.\n"
                )
        except ValueError:
            errmsg += f"'lon', invalid entry: '{lon}'.\n"

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
) -> int | pd.DateOffset:
    """Ask user input to get time duration.

    Offset aliases are to be used,
    e.g.: '5D' => 432_000s
    '2h' => 7_200s
    '3BMS' => <3*Months>
    See https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases

    Parameters
    ----------
    title : str
        Pop-up menu title

    msg : str
        Pop-up menu message

    default : '10min' => 600s
        Displayed default value

    Returns
    -------
    The total number of seconds of the entered time alias or the time alias if not transposable to duration (<N*Months>)

    """
    value = easygui.enterbox(
        msg=f"{msg}",
        title=f"{title}",
        default=f"{default}",
        strip=True,
    )

    while True:
        if value is None:
            msg = "'get_duration()' was cancelled"
            raise TypeError(msg)

        errmsg = ""
        try:
            offset = to_offset(value)
            # Check if the offset is convertible to Timedelta
            try:
                seconds = int(pd.Timedelta(offset).total_seconds())
            except ValueError:
                # For offsets like '3MS', seconds are not defined
                return offset
        except ValueError:
            errmsg = f"'{value}' is not a valid time alias."

        if errmsg == "":
            break

        value = easygui.enterbox(msg=errmsg, title=f"{title}", strip=True)

    return seconds


def get_datetime_format(
    title: str = "Get datetime format",
    msg: str = "Enter a datetime format code",
    default: str = "%d/%m/%Y\n%H:%M",
) -> str:
    r"""Ask user input to get datetime format.

    Datetime format codes are to be used,
    See https://docs.python.org/fr/3/library/datetime.html

    Parameters
    ----------
    title: str
        Pop-up menu title

    msg: str
        Pop-up menu message

    default: str
        Displayed default value: '%d/%m/%Y\n%H:%M'

    """
    fmt = easygui.enterbox(
        msg=f"{msg}",
        title=f"{title}",
        default=f"{default}",
        strip=True,
    )

    while True:
        if fmt is None:
            msg = "'get_duration()' was cancelled"
            raise TypeError(msg)

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


def print_spectro_from_audio(
    file: Path,
    nfft: int = 1024,
    window_size: int = 1024,
    overlap: int = 20,
    *,
    ax: bool = True,
) -> tuple:
    """Compute and prints a spectrogram from an audio file.

    Parameters
    ----------
    file: Path
        Audio file path

    nfft: int
        Default: 1024

    window_size: int
        Default: 1024

    overlap: int
        Default: 20

    ax: bool
        Show axes if True, defaults to True

    Returns
    -------
    The x and y resolutions

    Examples
    --------
    audio_file = Path("path/to/file")
    print_spectro_from_audio(audio_file)

    """
    # if not is_supported_audio_format(file):
    #     msg = "Audio file format is not supported"
    #     raise TypeError(msg)

    try:
        sr, data = sf.read(file)
    except ValueError as e:
        msg = f"Failed to read file {file}: {e}"
        raise RuntimeError(msg) from e

    overlap_samples = int(overlap / 100 * window_size)  # overlap in samples

    frequencies, times, sxx = spectrogram(
        data,
        fs=sr,
        nperseg=window_size,
        noverlap=overlap_samples,
        nfft=nfft,
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
            top=1,
            bottom=0,
            right=1,
            left=0,
            hspace=0,
            wspace=0,
        )  # delete white borders
    else:
        plt.tight_layout()

    ech = len(data)
    size_x = (ech - window_size) / overlap_samples
    size_y = nfft / 2

    return size_x, size_y


def print_spectro_from_npz(file: Path, *, ax: bool = True) -> None:
    """Compute and prints a spectrogram from a npz file.

    Parameters
    ----------
    file: Path
        to the npz file

    ax: bool
        show axes based on this value

    Examples
    --------
    npz_file = Path(path/to/file')
    print_spectro_from_npz(npz_file)

    """
    if file.suffix != ".npz":
        msg = "npz file format must be provided"
        raise ValueError(msg)

    try:
        with np.load(file, allow_pickle=True) as data:
            sxx = data["Sxx"]
            freq = data["Freq"]
            time = data["Time"]
    except ValueError as e:
        msg = f"Failed to load file {file}: {e}"
        raise RuntimeError(msg) from e

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


def add_weak_detection(
    file: Path,
    datetime_format: str = TIMESTAMP_FORMAT_AUDIO_FILE,
) -> pd.DataFrame:
    """Add weak detections APLOSE formatted DataFrame with only strong detections.

    Parameters
    ----------
    file: Path
        An APLOSE formatted csv file.
    datetime_format: str
        A string corresponding to the datetime format in the `filename` column

    """
    df_weak_only = load_detections(file=file, box=True)
    annotators = df_weak_only["annotator"].drop_duplicates().tolist()
    labels = df_weak_only["annotation"].drop_duplicates().tolist()
    max_freq = int(max(df_weak_only["end_frequency"]))
    max_time = int(max(df_weak_only["end_time"]))
    dataset_id = df_weak_only["dataset"][0]
    tz = df_weak_only["start_datetime"].iloc[0].tz

    for annotator in annotators:
        for label in labels:
            filenames = (
                df_weak_only[
                    (df_weak_only["annotator"] == annotator)
                    & (df_weak_only["annotation"] == label)
                ]["filename"]
                .drop_duplicates()
                .tolist()
            )
            for f in filenames:
                test = df_weak_only[
                    (df_weak_only["filename"] == f)
                    & (df_weak_only["annotation"] == label)
                ]["is_box"]
                if test.any():
                    start_datetime = strptime_from_text(
                        text=f,
                        datetime_template=datetime_format,
                    ).tz_localize(tz)
                    end_datetime = start_datetime + pd.Timedelta(max_time, unit="s")
                    new_line = [
                        dataset_id,
                        f,
                        0,
                        max_time,
                        0,
                        max_freq,
                        label,
                        annotator,
                        start_datetime,
                        end_datetime,
                        0,
                    ]
                    df_weak_only.loc[len(df_weak_only.index)] = new_line

    return df_weak_only.sort_values("start_datetime").reset_index(drop=True)


def json2df(json_path: Path) -> pd.DataFrame:
    """Convert a metadatax json file into a DataFrame.

    Parameters
    ----------
    json_path: Path
        Json file path

    """
    with json_path.open(encoding="utf-8") as f:
        metadatax_df = pd.json_normalize(json.load(f))
        metadatax_df["deployment_date"] = pd.to_datetime(
            metadatax_df["deployment_date"],
        )
        metadatax_df["recovery_date"] = pd.to_datetime(metadatax_df["recovery_date"])

    return metadatax_df


def add_season_period(ax: Axes = None, bar_height: int = 10, northern: bool = True) -> None:
    """Add a bar at the top of the plot to seasons.

    Parameters
    ----------
    northern: boolean, default True.
        Specify if the seasons are northern or austral
    ax: Axes
        Figure plot

    bar_height: int
        Bar height in pixels

    """
    if not ax:
        ax = plt.gca()

    if not ax.has_data():
        msg = "Axes have no data"
        raise ValueError(msg)

    bins = date_range(
        start=(
            Timestamp(ax.get_xlim()[0], unit="D").normalize() - DateOffset(months=1)
        ).replace(day=1),
        end=(
            Timestamp(ax.get_xlim()[1], unit="D").normalize() + DateOffset(months=1)
        ).replace(day=1),
        freq="MS",
    )

    season_colors = {
        "winter": "#2ce5e3",
        "spring": "#4fcf50",
        "summer": "#ffcf50",
        "autumn": "#fb9a67",
    }

    bin_centers = [
        (bins[i].timestamp() + bins[i + 1].timestamp()) / 2
        for i in range(len(bins) - 1)
    ]
    bin_centers = [Timestamp(center, unit="s") for center in bin_centers]

    bin_seasons = [get_season(bc, northern).split()[0] for bc in bin_centers]
    bar_height = set_bar_height(ax, bar_height)
    bar_bottom = ax.get_ylim()[1] + (0.2 * bar_height)

    for i, season in enumerate(bin_seasons):
        ax.bar(
            bin_centers[i],
            height=bar_height,
            bottom=bar_bottom,
            width=(bins[i + 1] - bins[i]),
            color=season_colors[season],
            align="center",
            zorder=3,
            alpha=0.6,
        )

    plt.ylim(ax.dataLim.ymin, ax.dataLim.ymax)


def set_bar_height(ax: Axes = None, pixel_height: int = 10) -> float:
    """Convert pixel height to data coordinates.

    Parameters
    ----------
    ax: Axes
        Figure plot

    pixel_height: int
        In pixel

    """
    if not ax:
        ax = plt.gca()

    if not ax.has_data():
        msg = "Axes have no data"
        raise ValueError(msg)

    display_to_data = ax.transData.inverted().transform
    _, data_bottom = display_to_data((0, 0))  # Bottom of the axis
    _, data_top = display_to_data((0, pixel_height))  # Top of the bar

    return data_top - data_bottom  # Convert pixel height to data scale


def add_recording_period(
    df: pd.DataFrame,
    ax: mpl.axes.Axes = None,
    bar_height: int = 10,
) -> None:
    """Add a bar at the bottom on plot to show recording periods.

    Parameters
    ----------
    df: pd.DataFrame
        Includes the recording campaign deployment
        and recovery dates (typically extracted from metadatax)

    ax: Axes
        Figure plot

    bar_height: int
        Bar height in pixels

    """
    if not ax:
        ax = plt.gca()

    if not ax.has_data():
        msg = "Axes have no data"
        raise ValueError(msg)

    recorder_intervals = [
        (start, end - start)
        for start, end in zip(df["deployment_date"], df["recovery_date"], strict=False)
    ]

    bar_height = set_bar_height(ax=ax, pixel_height=bar_height)

    ax.broken_barh(
        recorder_intervals,
        (ax.get_ylim()[0] - (1.2 * bar_height), bar_height),
        facecolors="red",
        alpha=0.6,
    )
    plt.ylim(ax.dataLim.ymin, ax.dataLim.ymax)
