from OSmOSE.utils.timestamp_utils import strftime_osmose_format, strptime_from_text
import pandas as pd
import pytz


def clean_pamguard_false_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans PAMGuard whistle and moan detector first detection of each audio file (might be very specific to Sylence data).
    This is because the first detection on each audio file corresponds to the detection of an electronic buzz made by the recorder.

    The first detection in each audio file seem to be caused by an electronic buzz produced
    by the recorder. This function identifies and removes these false detections
    by checking if a detection occurs within the first five seconds of the corresponding audio file.

    Parameters
    ----------
    df: pd.DataFrame
        An APLOSE formatted DataFrame (presumably from PAMGuard)

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with false detections removed.
    """
    filenames = df["filename"]
    tz_data = df["start_datetime"][0].tz
    filename_datetimes = [
        strptime_from_text(fn, "%Y_%m_%d_%H_%M_%S").tz_localize(tz_data)
        for fn in filenames
    ]

    start_datetimes = df["start_datetime"]

    # compare date of filename detection and date of detection
    # and delete all lines for which the detection happens in the 5 first seconds of the file
    idx_false_detections = []
    for i in range(0, len(start_datetimes)):
        d = (start_datetimes[i] - filename_datetimes[i]).total_seconds()
        if d < 5:
            idx_false_detections.append(i)

    return df.drop(labels=idx_false_detections)


def fpod2aplose(
    df: pd.DataFrame,
    tz: pytz.BaseTzInfo,
    dataset_name: str,
    annotation: str,
    bin_size: int = 60,
) -> pd.DataFrame:
    """
    From FPOD result DataFrame to APLOSE formatted DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        FPOD result dataframe
    tz: pytz.BaseTzInfo
        Timezone object to get non-naïve datetimes
    dataset_name: str
        dataset name
    annotation: str
        annotation name
    bin_size: int
        Duration of the detections in seconds

     Returns
    -------
    pd.DataFrame
        An APLOSE formatted DataFrame
    """
    fpod_start_dt = sorted(
        [
            tz.localize(strptime_from_text(entry, "%d/%m/%Y %H:%M"))
            for entry in df["Date heure"]
        ]
    )

    fpod_end_dt = sorted(
        [entry + pd.Timedelta(seconds=bin_size) for entry in fpod_start_dt]
    )

    data = {
        "dataset": [dataset_name] * len(df),
        "filename": [""] * len(df),
        "start_time": [0] * len(df),
        "end_time": [bin_size] * len(df),
        "start_frequency": [0] * len(df),
        "end_frequency": [0] * len(df),
        "annotation": [annotation] * len(df),
        "annotator": ["FPOD"] * len(df),
        "start_datetime": [strftime_osmose_format(entry) for entry in fpod_start_dt],
        "end_datetime": [strftime_osmose_format(entry) for entry in fpod_end_dt],
    }

    return pd.DataFrame(data)

def cpod2aplose(
    df: pd.DataFrame,
    tz: pytz.BaseTzInfo,
    dataset_name: str,
    annotation: str,
    bin_size: int = 60,
) -> pd.DataFrame:
    """
    From CPOD result DataFrame to APLOSE formatted DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        CPOD result dataframe
    tz: pytz.BaseTzInfo
        Timezone object to get non-naïve datetimes
    dataset_name: str
        dataset name
    annotation: str
        annotation name
    bin_size: int
        Duration of the detections in seconds

     Returns
    -------
    pd.DataFrame
        An APLOSE formatted DataFrame
    """
    data = fpod2aplose(df,tz,dataset_name,annotation,bin_size)
    data['annotator'] = data.loc[data['annotator'] == 'FPOD'] = 'CPOD'
    return pd.DataFrame(data)