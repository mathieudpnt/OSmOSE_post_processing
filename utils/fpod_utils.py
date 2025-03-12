import pandas as pd
import pytz
from OSmOSE.utils.timestamp_utils import strftime_osmose_format, strptime_from_text


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
