# TODO: la fonction ne marche pas en l'état actuel, à modifier et à passer sur OSEkit
import pandas as pd
import pytz


def raven2aplose(
    df: pd.DataFrame,
    dataset_name: str,
    annotation: str,
    annotator: str,
    offset: int = 0,
) -> pd.DataFrame:
    """Export a Raven formatted result DataFrame to APLOSE formatted DataFrame

    Parameters
    ----------
    df: Raven formatted result DataFrame
    dataset_name: dataset name given to the APLOSE DataFrame
    annotation: annotation name given to the APLOSE DataFrame
    annotator: annotator name given to the APLOSE DataFrame
    offset: Integer, offset in minutes to specify the timezone of the datetimes

    Returns
    -------
    df_APLOSE: APLOSE formatted DataFrame

    """
    tz = pytz.FixedOffset(offset)

    # filename = df["Begin File"]
    end_time = [end - beg for end, beg in zip(df["End Time (s)"], df["Begin Time (s)"])]
    start_frequency = df["Low Freq (Hz)"]
    end_frequency = df["High Freq (Hz)"]
    start_datetime = pd.to_datetime(
        df["Begin Date Time"], format="%Y/%m/%d %H:%M:%S.0000"
    ).tolist()
    start_datetime = [s.tz_localize(tz) for s in start_datetime]
    end_datetime = [s + pd.Timedelta(seconds=end_time[0]) for s in start_datetime]

    start_datetime = [
        pd.Timestamp.strftime(s, "%Y-%m-%dT%H:%M:%S.%f%z") for s in start_datetime
    ]
    end_datetime = [
        pd.Timestamp.strftime(s, "%Y-%m-%dT%H:%M:%S.%f%z") for s in end_datetime
    ]

    df_APLOSE = pd.DataFrame(
        {
            "dataset": [dataset_name] * len(df),
            "filename": [""] * len(df),
            "start_time": [0] * len(df),
            "end_time": end_time,
            "start_frequency": start_frequency,
            "end_frequency": end_frequency,
            "annotation": [annotation] * len(df),
            "annotator": [annotator] * len(df),
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
        }
    )

    return df_APLOSE
