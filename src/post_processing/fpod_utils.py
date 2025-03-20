import logging
from pathlib import Path

import pandas as pd
import pytz
from OSmOSE.config import TIMESTAMP_FORMAT_AUDIO_FILE
from OSmOSE.utils.timestamp_utils import strftime_osmose_format, strptime_from_text


def fpod2aplose(
    df: pd.DataFrame,
    tz: pytz.BaseTzInfo,
    dataset_name: str,
    annotation: str,
    bin_size: int = 60,
) -> pd.DataFrame:
    """From FPOD result DataFrame to APLOSE formatted DataFrame.

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
        ],
    )

    fpod_end_dt = sorted(
        [entry + pd.Timedelta(seconds=bin_size) for entry in fpod_start_dt],
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
        "is_box": [0] * len(df),
    }

    return pd.DataFrame(data)


def cpod2aplose(
    df: pd.DataFrame,
    tz: pytz.BaseTzInfo,
    dataset_name: str,
    annotation: str,
    bin_size: int = 60,
    extra_columns: list | None = None,
) -> pd.DataFrame:
    """From CPOD result DataFrame to APLOSE formatted DataFrame.

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
    bin_size: int, optional
        Duration of the detections in seconds
    extra_columns: list, optional
        Additional columns added from df to data

    Returns
    -------
    pd.DataFrame
        An APLOSE formatted DataFrame

    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    df_cpod = df.rename(columns={"ChunkEnd": "Date heure"})

    # remove lines where the C-POD stopped working
    df_cpod.drop(
        df_cpod.loc[df_cpod["Date heure"] == " at minute "].index, inplace=True,
    )
    data = fpod2aplose(df_cpod, tz, dataset_name, annotation, bin_size)
    data["annotator"] = data.loc[data["annotator"] == "FPOD"] = "CPOD"
    if extra_columns:
        for col in extra_columns:
            if col in df_cpod.columns:
                data[col] = df_cpod[col].tolist()
            else:
                msg = f"The column '{col}' does not exist and will be ignored."
                logging.warning(msg)

    return pd.DataFrame(data)


def usable_data_phase(
    d_meta: pd.DataFrame,
    df: pd.DataFrame,
    dpl: str,
) -> pd.DataFrame:
    """Calculate the percentage of usable data.

    Considering the deployment dates and the collected data.

    Parameters
    ----------
    df: pd.DataFrame
        CPOD result DataFrame
    d_meta: pd.DataFrame
        Metadata DataFrame with deployments information (previously exported as json)
    dpl: str
        Deployment of interest where percentage of usable data will be calculated

    Returns
    -------
    pd.DataFrame
        Returns the percentage of usable datas in the chosen phase

    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    d_meta.loc[:, ["deployment_date", "recovery_date"]] = d_meta[
        ["deployment_date", "recovery_date"]
    ].apply(
        pd.to_datetime,
    )
    df["start_datetime"] = pd.to_datetime(df["start_datetime"])

    phase = d_meta.loc[d_meta["name"] == dpl].reset_index()
    data = df.loc[df["name"] == dpl].reset_index()
    start_date = phase.loc[0, "deployment_date"]
    end_date = phase.loc[0, "recovery_date"]

    # Calculate the percentage of collected data on the phase length of time
    if data.empty:
        percentage_data = 0
        msg = "No data for this phase"
    else:
        df_end = data.loc[data.index[-1], "start_datetime"]
        df_start = data.loc[data.index[0], "start_datetime"]
        act_length = df_end - df_start
        p_length = end_date - start_date
        percentage_data = act_length * 100 / p_length
        msg = f"Percentage of usable data : {percentage_data}%"

    logging.info(msg)
    return percentage_data


def meta_cut_aplose(
    d_meta: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """From APLOSE DataFrame with all rows to filtered DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        CPOD result dataframe
    d_meta: pd.DataFrame
        Metadata dataframe with deployments information (previously exported as json)

    Returns
    -------
    pd.DataFrame
        An APLOSE DataFrame with data from beginning to end of each deployment.
        Returns the percentage of usable datas.

    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    d_meta.loc[:, ["deployment_date", "recovery_date"]] = d_meta[
        ["deployment_date", "recovery_date"]
    ].apply(pd.to_datetime)
    df["start_datetime"] = pd.to_datetime(
        df["start_datetime"], format=TIMESTAMP_FORMAT_AUDIO_FILE,
    )

    # Add DPM column
    df["DPM"] = (df["Nfiltered"] > 0).astype(int)

    # Extract corresponding line
    campaign = df.iloc[0]["dataset"]
    phase = d_meta.loc[d_meta["name"] == campaign].reset_index()
    start_date = phase.loc[0, "deployment_date"]
    end_date = phase.loc[0, "recovery_date"]
    df = df[
        (df["start_datetime"] >= start_date) & (df["start_datetime"] <= end_date)
    ].copy()

    # Calculate the percentage of collected data on the phase length of time
    if df.empty:
        msg = "No data for this phase"
    else:
        df_end = df.loc[df.index[-1], "start_datetime"]
        df_start = df.loc[df.index[0], "start_datetime"]
        act_length = df_end - df_start
        p_length = end_date - start_date
        percentage_data = act_length * 100 / p_length
        on = int(df.loc[df.MinsOn == 1, "MinsOn"].count())
        percentage_on = percentage_data * (on / len(df))
        msg = f"Percentage of usable data : {percentage_on}%"

    logging.info(msg)
    return df


def format_calendar(path: Path) -> pd.DataFrame:
    """Format calendar.

    Parameters
    ----------
    path: Path
        Excel calendar path

    """
    df_calendar = pd.read_excel(path)
    df_calendar = df_calendar[df_calendar["Site group"] == "Data"].copy()

    return df_calendar.rename(
        columns={"Start": "start_datetime", "Stop": "end_datetime", "Site": "site.name"},
    )


def dpm_to_dph(
    df: pd.DataFrame,
    tz: pytz.BaseTzInfo,
    dataset_name: str,
    annotation: str,
    bin_size: int = 3600,
    extra_columns: list | None = None,
) -> pd.DataFrame:
    """From CPOD result DataFrame to APLOSE formatted DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        CPOD result DataFrame
    tz: pytz.BaseTzInfo
        Timezone object to get timezone-aware datetimes
    dataset_name: str
        dataset name
    annotation: str
        annotation name
    bin_size: int
        Duration of the detections in seconds
    extra_columns: list, optional
        Additional columns added from df to data

    Returns
    -------
    pd.DataFrame
        An APLOSE DataFrame

    """
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True)
    df["end_datetime"] = pd.to_datetime(df["end_datetime"], utc=True)

    # Truncate column
    df["Date heure"] = df["start_datetime"].dt.floor("h")

    # Group by hour
    dph = df.groupby(["Date heure"])["DPM"].sum().reset_index()
    dph["Date heure"] = dph["Date heure"].apply(
        lambda x: pd.Timestamp(x).strftime(format="%d/%m/%Y %H:%M:%S"),
    )

    return cpod2aplose(dph, tz, dataset_name, annotation, bin_size, extra_columns)


def assign_phase(
    meta: pd.DataFrame,
    data: pd.DataFrame,
    site: str,
) -> pd.DataFrame:
    """Add a column to an APLOSE DataFrame to specify the name of the phase.

    The name of the phase is attributed according to metadata.

    Parameters
    ----------
    meta: pd.DataFrame
        Metadata dataframe with deployments information (previously exported as json).
    data: pd.DataFrame
        Contain positive hours to detections.
    site: str
        Name of the site you wish to assign phases to.

    Returns
    -------
    pd.DataFrame
        The same dataframe with the column Phase.

    """
    data["start_datetime"] = pd.to_datetime(data["start_datetime"], utc=True)
    meta["deployment_date"] = pd.to_datetime(meta["deployment_date"], utc=True)
    meta["recovery_date"] = pd.to_datetime(meta["recovery_date"], utc=True)

    meta = meta[meta["site.name"] == site].copy()

    data["name"] = None
    for _, meta_row in meta.iterrows():
        j = 0
        while j < len(data):
            if (
                meta_row["deployment_date"]
                <= data.loc[j, "start_datetime"]
                < meta_row["recovery_date"]
            ):
                data.loc[j, "name"] = meta_row["name"]
            j += 1
    return data


def assign_phase_simple(
    meta: pd.DataFrame,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add a column to an Aplose dataframe to specify the name of the phase, according to metadata.

    Parameters
    ----------
    meta: pd.DataFrame
        Metadata dataframe with deployments information (previously exported as json).
    data: pd.DataFrame
        Dataframe containing positive hours to detections.

    Returns
    -------
    pd.DataFrame
        The same dataframe with the column Phase.

    """
    data["start_datetime"] = pd.to_datetime(data["start_datetime"], utc=True)
    meta["deployment_date"] = pd.to_datetime(meta["deployment_date"], utc=True)
    meta["recovery_date"] = pd.to_datetime(meta["recovery_date"], utc=True)
    meta["deployment_date"] = meta["deployment_date"].dt.floor("d")
    meta["recovery_date"] = meta["recovery_date"].dt.floor("d")

    data["name"] = None
    for site in data["site.name"].unique():
        site_meta = meta[meta["site.name"] == site]
        site_data = data[data["site.name"] == site]

        for i, meta_row in site_meta.iterrows():
            time_filter = (
                meta_row["deployment_date"] <= site_data["start_datetime"]
            ) & (site_data["start_datetime"] < meta_row["recovery_date"])
            data.loc[site_data.index[time_filter], "name"] = meta_row["name"]

    return data


def generate_hourly_detections(meta: pd.DataFrame, site: str) -> pd.DataFrame:
    """Create a DataFrame with one line per hour between start and end dates.

    Keep the number of detections per hour between these dates.

    Parameters
    ----------
    meta: pd.DataFrame
        Metadata dataframe with deployments information (previously exported as json)
    site: str
        A way to isolate the site you want to work on.

    Returns
    -------
    pd.DataFrame
        A full period of time with positive and negative hours to detections.

    """
    df_meta = meta[meta["site.name"] == site].copy()
    df_meta["deployment_date"] = pd.to_datetime(df_meta["deployment_date"])
    df_meta["recovery_date"] = pd.to_datetime(df_meta["recovery_date"])
    df_meta["deployment_date"] = df_meta["deployment_date"].dt.floor("h")
    df_meta["recovery_date"] = df_meta["recovery_date"].dt.floor("h")
    df_meta = df_meta.sort_values(by=["deployment_date"])

    records = []
    for _, row in df_meta.iterrows():
        name = row["name"]
        period = pd.date_range(
            start=row["deployment_date"], end=row["recovery_date"], freq="h",
        )
        for date in period:
            records.append({"name": name, "start_datetime": date})

    return pd.DataFrame(records)


def merging_tab(meta: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame with one line per hour between start and end dates.

    Keep the number of detections per hour between these dates.

    Parameters
    ----------
    meta: pd.DataFrame
        Metadata with deployments information (previously exported as json)
    data: pd.DataFrame
        Contain positive hours to detections

    Returns
    -------
    pd.DataFrame
        A full period of time with positive and negative hours to detections.

    """
    data["start_datetime"] = pd.to_datetime(data["start_datetime"], utc=True)
    meta["start_datetime"] = pd.to_datetime(meta["start_datetime"], utc=True)

    deploy_detec = data["name"].unique()
    df_filtered = meta[meta["name"].isin(deploy_detec)]

    output = df_filtered.merge(
        data[["name", "start_datetime", "DPM"]],
        on=["name", "start_datetime"],
        how="outer",
    )
    output["DPM"] = output["DPM"].fillna(0)

    output["Day"] = output["start_datetime"].dt.day
    output["Month"] = output["start_datetime"].dt.month
    output["Year"] = output["start_datetime"].dt.year
    output["hour"] = output["start_datetime"].dt.hour

    return output
