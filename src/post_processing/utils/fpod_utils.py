"""FPOD/ CPOD processing functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytz
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from osekit.utils.timestamp_utils import strftime_osmose_format, strptime_from_text
from pandas import (
    DataFrame,
    Timedelta,
    concat,
    date_range,
    notna,
    read_csv,
    to_datetime,
    to_timedelta,
)

from post_processing.utils.core_utils import get_coordinates, get_sun_times

if TYPE_CHECKING:
    import pytz

logger = logging.getLogger(__name__)
site_colors = {
    "Site A Haute": "#118B50",
    "Site B Heugh": "#5DB996",
    "Site C Chat": "#B0DB9C",
    "Site D Simone": "#E3F0AF",
    "CA4": "#80D8C3",
    "Walde": "#4DA8DA",
    "Point C": "#932F67",
    "Point D": "#D92C54",
    "Point E": "#DDDEAB",
    "Point F": "#8ABB6C",
    "Point G": "#456882",
}

season_color = {
    "spring": "green", #C5E0B4
    "summer": "darkgoldenrod", #FCF97F
    "autumn": "orange", #ED7C2F
    "winter": "blue", #B4C7E8
}

def fpod2aplose(
    df: DataFrame,
    tz: pytz.timezone,
    dataset_name: str,
    annotation: str,
    bin_size: int = 60,
) -> DataFrame:
    """Format FPOD DataFrame to match APLOSE format.

    Parameters
    ----------
    df: DataFrame
        FPOD result dataframe
    tz: pytz.timezone
        Timezone object to get non-naïve datetimes
    dataset_name: str
        dataset name
    annotation: str
        annotation name
    bin_size: int
        Duration of the detections in seconds

    Returns
    -------
    DataFrame
        An APLOSE formatted DataFrame

    """
    fpod_start_dt = sorted(
        [
            tz.localize(strptime_from_text(entry, "%d/%m/%Y %H:%M"))
            for entry in df["ChunkEnd"]
        ],
    )

    fpod_end_dt = sorted(
        [entry + Timedelta(seconds=bin_size) for entry in fpod_start_dt],
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
    if "deploy.name" in df.columns:
        data["deploy.name"] = df["deploy.name"]

    return DataFrame(data)


def cpod2aplose(
    df: DataFrame,
    tz: pytz.BaseTzInfo,
    dataset_name: str,
    annotation: str,
    bin_size: int = 60,
    extra_columns: list | None = None,
) -> DataFrame:
    """Format CPOD DataFrame to match APLOSE format.

    Parameters
    ----------
    df: DataFrame
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
    DataFrame
        An APLOSE formatted DataFrame

    """
    results = []

    for deploy_name in df["deploy.name"].unique():
        df_deploy = df[df["deploy.name"] == deploy_name].copy()

        result = fpod2aplose(df_deploy, tz, dataset_name, annotation, bin_size)
        result["annotator"] = result.loc[result["annotator"] == "FPOD"] = "CPOD"

        if extra_columns:
            for col in extra_columns:
                if col in df_deploy.columns:
                    result[col] = df_deploy[col].tolist()

        results.append(result)

    return concat(results, ignore_index=True)


def pod2aplose(
    df: DataFrame,
    tz: pytz.timezone,
    dataset_name: str,
    annotation: str,
    annotator: str,
    bin_size: int = 60,
) -> DataFrame:
    """Format PODs DataFrame to match APLOSE format.

    Parameters
    ----------
    df: DataFrame
        FPOD result dataframe
    tz: pytz.timezone
        Timezone object to get non-naïve datetimes
    dataset_name: str
        dataset name
    annotation: str
        annotation name
    annotator: str
        annotator name
    bin_size: int
        Duration of the detections in seconds

    Returns
    -------
    DataFrame
        An APLOSE formatted DataFrame

    """
    df = df.copy()
    df["_temp_dt"] = [
        tz.localize(strptime_from_text(entry, "%d/%m/%Y %H:%M"))
        for entry in df["ChunkEnd"]
    ]

    # Trier le DataFrame selon ces datetime
    df = df.sort_values("_temp_dt").reset_index(drop=True)

    # Maintenant extraire les colonnes triées
    fpod_start_dt = df["_temp_dt"].tolist()
    fpod_end_dt = [entry + Timedelta(seconds=bin_size) for entry in fpod_start_dt]

    data = {
        "dataset": [dataset_name] * len(df),
        "filename": df["deploy.name"].tolist(),
        "start_time": [0] * len(df),
        "end_time": [bin_size] * len(df),
        "start_frequency": [0] * len(df),
        "end_frequency": [0] * len(df),
        "annotation": [annotation] * len(df),
        "annotator": [annotator] * len(df),
        "start_datetime": [strftime_osmose_format(entry) for entry in fpod_start_dt],
        "end_datetime": [strftime_osmose_format(entry) for entry in fpod_end_dt],
        "is_box": [0] * len(df),
    }

    return DataFrame(data)


def csv_folder(
    folder_path: Path,
    sep: str = ";",
    encoding: str = "latin-1",
) -> DataFrame:
    """Process all CSV files from a folder.

    Parameters
    ----------
    folder_path: Path
        Folder's place.
    sep: str, default=";"
        Column separator.
    encoding: str, default="latin-1"
        File encoding.

    Returns
    -------
    DataFrame
        Concatenated data with optional filename column.

    Raises
    ------
    ValueError
        If no CSV files found.

    """
    all_files = list(folder_path.rglob("*.csv"))

    if not all_files:
        msg = f"No .csv files found in {folder_path}"
        raise ValueError(msg)

    all_data = []
    for file in all_files:
        df = read_csv(file, sep=sep, encoding=encoding)
        df["deploy.name"] = file.stem
        all_data.append(df)

    return concat(all_data, ignore_index=True)


def txt_folder(
    folder_path: Path,
    sep: str = "\t",
) -> DataFrame:
    r"""Process all TXT files from a folder.

    Parameters
    ----------
    folder_path: Path
        Folder's place.
    sep: str, default="\t"
        Column separator.

    Returns
    -------
    DataFrame
       Concatenated data from all TXT files.

    """
    all_files = list(Path(folder_path).rglob("*.txt"))

    if not all_files:
        msg = f"No .txt files found in {folder_path}"
        raise ValueError(msg)

    all_data = []
    for file in all_files:
        file_path = folder_path / file
        df = read_csv(file_path, sep=sep)
        all_data.append(df)

    return concat(all_data, ignore_index=True)


def parse_timestamps(
    df: DataFrame,
    col_timestamp: str,
    date_formats: list[str] | None = None,
) -> DataFrame:
    """Parse timestamp column with multiple possible formats.

    Parameters
    ----------
    df: DataFrame
        Input dataframe.
    col_timestamp: str
        Name of the timestamp column to parse.
    date_formats: list[str], optional
        List of strptime formats to try. If None, uses common formats.

    Returns
    -------
    DataFrame
        Copy of df with parsed timestamps.

    Raises
    ------
    ValueError
        If timestamps cannot be parsed with any format.

    """
    if date_formats is None:
        date_formats = [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S:%Z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%d/%m/%Y %H:%M",
        ]

    df = df.copy()
    df[col_timestamp] = df[col_timestamp].apply(
        lambda x: strptime_from_text(x, date_formats))
    return df


def required_columns(
    df: DataFrame,
    columns: list[str],
) -> None:
    """Validate that required columns exist in dataframe.

    Parameters
    ----------
    df : DataFrame
        Dataframe to validate.
    columns : list[str]
        List of required column names.

    Raises
    ------
    ValueError
        If any required column is missing.

    """
    for col in columns:
        if col not in df.columns:
            msg = f"'{col}' not found in {df}"
            raise ValueError(msg)


def create_mask(
    df: DataFrame,
    col_timestamp: str,
    col_start: str,
    col_end: str,
) -> DataFrame:
    """Filter rows to keep only those within deployment period.

    Parameters
    ----------
    df : DataFrame
        Dataframe with timestamp and deployment period columns.
    col_timestamp : str
        Name of timestamp column.
    col_start : str
        Name of deployment start date column.
    col_end : str
        Name of deployment end date column.

    Returns
    -------
    DataFrame
        Filtered dataframe with rows in deployment periods.

    """
    mask = (
        (df[col_timestamp] >= df[col_start])
    & (df[col_timestamp] <= df[col_end])
        & df[col_timestamp].notna()
        & df[col_start].notna()
        & df[col_end].notna()
    )
    return df[mask].copy()


def meta_cut_aplose(
    raw_data: DataFrame,
    metadata: DataFrame,
) -> DataFrame:
    """Filter data to keep only rows within deployment periods.

    Parameters
    ----------
    raw_data : DataFrame
        Dataframe containing deployment name and timestamps.
    metadata : DataFrame
        Metadata with deployment periods (start/end dates).

    Returns
    -------
    DataFrame
        Filtered data with only rows within deployment periods.

    """
    required_columns(
        raw_data,["deploy.name", "start_datetime"])
    required_columns(
        metadata,["deploy.name", "deployment_date","recovery_date"])

    raw = parse_timestamps(raw_data, "start_datetime")
    raw = raw.sort_values(["start_datetime"])

    dfm = raw.merge(
        metadata[["deploy.name", "deployment_date","recovery_date"]],
        on="deploy.name",
        how="left",
    )

    out = create_mask(dfm, "start_datetime", "deployment_date", "recovery_date")

    columns_to_drop = [
        col for col in ["deployment_date","recovery_date"] if col not in raw_data.
        columns]
    if columns_to_drop:
        out = out.drop(columns=columns_to_drop)

    return out.sort_values(["start_datetime"]).reset_index(drop=True)


def add_utc(
    df: DataFrame,
    cols: list,
    fr:str="h",
) -> DataFrame:
    """Create a DataFrame with one line per hour between start and end dates.

    Keep the number of detections per hour between these dates.

    Parameters
    ----------
    df: pd.DataFrame
        Metadata dataframe with deployments information (previously exported as json).
    cols:list
        Timestamp column names.
    fr:str
        Frequency of the range of detections.

    Returns
    -------
    pd.DataFrame
        A full period of time with positive and negative hours to detections.

    """
    for col in df[cols]:
        df[col] = to_datetime(df[col], utc=True)
        df[col] = df[col].dt.floor(fr)
    return df


def build_range(
    df: DataFrame,
    fr:str="h",
) -> DataFrame:
    """Create a DataFrame with one line per hour between start and end dates.

    Keep the number of detections per hour between these dates.

    Parameters
    ----------
    df: pd.DataFrame
        Metadata dataframe with deployments information (previously exported as json)
    fr:str
        Frequency of the range of detections.

    Returns
    -------
    pd.DataFrame
        A full period of time with positive and negative hours to detections.

    """
    add_utc(df, ["Deb","Fin"], fr)

    all_ranges = []
    for _, row in df.iterrows():
        hours = date_range(row["Deb"], row["Fin"], freq=fr)
        tmp = DataFrame(
            {
                "deploy.name": row["deploy.name"],
                "start_datetime": hours,
            },
        )
        all_ranges.append(tmp)

    return concat(all_ranges, ignore_index=True)


def feeding_buzz(
    df: DataFrame,
    species: str,
) -> DataFrame:
    """Process a CPOD/FPOD feeding buzz detection file.

    Gives the feeding buzz duration, depending on the studied species.

    Parameters
    ----------
    df: DataFrame
        Path to cpod.exe feeding buzz file
    species: str
        Select the species to use between porpoise and Commerson's dolphin

    Returns
    -------
    DataFrame
        Containing all ICIs for every positive minutes to clicks

    """
    df["microsec"] = df["microsec"] / 1e6
    df["ICI"] = df["microsec"].diff()

    if species == "Dauphin":  # Herzing et al., 2014
        df["Buzz"] = (df["ICI"].between(0, 0.02)).astype(int)
    elif species == "Marsouin":  # Nuuttila et al., 2013
        df["Buzz"] = (df["ICI"].between(0, 0.01)).astype(int)
    elif species == "Commerson":  # Reyes Reyes et al., 2015
        df["Buzz"] = (df["ICI"].between(0, 0.005)).astype(int)
    else:
        msg = "This species is not supported"
        raise ValueError(msg)

    try:
        df["Minute"].astype(int)
        df["datetime"] = (to_datetime("1900-01-01") +
                            to_timedelta(df["Minute"], unit="min") +
                            to_timedelta(df["microsec"], unit="us") -
                            to_timedelta(2, unit="D"))
        df["start_datetime"] = df["datetime"].dt.floor("min")
    except (ValueError, TypeError):
        df["start_datetime"] = to_datetime(df["Minute"], dayfirst=True)

    f = df.groupby(["start_datetime"])["Buzz"].sum().reset_index()

    f["Foraging"] = (f["Buzz"] != 0).astype(int)

    return f


def assign_daytime(
    df: DataFrame,
) -> DataFrame:
    """Assign datetime categories to temporal events.

    Categorize daytime of the detection (among 4 categories).

    Parameters
    ----------
    df: DataFrame
        Contains positive hours to detections.

    Returns
    -------
    DataFrame
        The same dataframe with the column daytime.

    """
    df["start_datetime"] = to_datetime(df["start_datetime"], utc=True)
    start = df["start_datetime"].min()
    stop = df["start_datetime"].max()
    lat, lon = get_coordinates()
    sunrise, sunset = get_sun_times(start, stop, lat, lon)

    sun_times = DataFrame(
        {   "date": date_range(start, stop, freq="D"),
            "sunrise": [Timedelta(h, "hours") for h in sunrise],
            "sunset": [Timedelta(h, "hours") for h in sunset],
        })

    sun_times["sunrise"] = sun_times["date"].dt.floor("D") + sun_times["sunrise"]
    sun_times["sunset"] = sun_times["date"].dt.floor("D") + sun_times["sunset"]

    for i, row in df.iterrows():
        dpm_i = row["start_datetime"]
        if notna(dpm_i):  # Check if time is not NaN
            jour_i = sun_times[
                (sun_times["sunrise"].dt.year == dpm_i.year)
                & (sun_times["sunrise"].dt.month == dpm_i.month)
                & (sun_times["sunrise"].dt.day == dpm_i.day)
                ]
            if not jour_i.empty:  # Ensure there's a matching row
                jour_i = jour_i.iloc[0]  # Extract first match
                if (dpm_i <= jour_i["sunrise"]) | (dpm_i > jour_i["sunset"]):
                    df.loc[i, "REGIME"] = 1
                else:
                    df.loc[i, "REGIME"] = 2

    return df


def is_dpm_col(
    df: DataFrame,
) -> DataFrame:
    """Ensure DPM column exists with default value of 1.

    Parameters
    ----------
    df: DataFrame
        Input dataframe.

    Returns
    -------
    DataFrame
        Copy of df with DPM column.

    """
    df = df.copy()
    if "DPM" not in df.columns:
        df["DPM"] = 1
    return df


def pf_datetime(
    df: DataFrame,
    col_datetime: str,
    frequency: str,
) -> DataFrame:
    """Parse datetime column and floor to specified frequency.

    Parameters
    ----------
    df: DataFrame
        Input dataframe.
    col_datetime: str
        Name of datetime column.
    frequency: str
        Pandas frequency string (e.g., "D", "h", "10min").

    Returns
    -------
    DataFrame
        Copy of df with parsed and floored datetime.

    """
    df = df.copy()
    df[col_datetime] = to_datetime(df[col_datetime], utc=True)
    df[col_datetime] = df[col_datetime].dt.floor(frequency)
    return df


def build_aggregation_dict(
    df: DataFrame,
    base_agg: dict[str, str],
    extra_columns: list[str] | None = None,
) -> dict[str, str]:
    """Build aggregation dictionary with validation.

    Parameters
    ----------
    df: DataFrame
        Input dataframe to check column existence.
    base_agg: dict[str, str]
        Base aggregation dictionary (e.g., {"DPM": "sum"}).
    extra_columns: list[str], optional
        Additional columns to aggregate with "first" strategy.

    Returns
    -------
    dict[str, str]
        Complete aggregation dictionary.

    """
    agg_dict = base_agg.copy()

    if extra_columns:
        for col in extra_columns:
            if col in df.columns:
                agg_dict[col] = "first"
            else:
                logger.warning("Column '%s' does not exist and will be ignored.", col)

    return agg_dict


def resample_dpm(
    df: DataFrame,
    frq: str,
    group_by: list[str] | None = None,
    extra_columns: list[str] | None = None,
) -> DataFrame:
    """Resample DPM data to specified time frequency.

    Aggregates Detection Positive Minutes (DPM) by time period,
    optionally preserving grouping columns like deployment name.

    Parameters
    ----------
    df: DataFrame
        CPOD result DataFrame with DPM data.
    frq: str
        Pandas frequency string: "D" (day), "h" (hour), "10min", etc.
    group_by: list[str], optional
        Columns to group by (e.g., ["deploy.name", "start_datetime"]).
        If None, groups only by start_datetime.
    extra_columns: list[str], optional
        Additional columns to preserve (uses "first" aggregation).

    Returns
    -------
    DataFrame
        Resampled DataFrame with aggregated DPM values.

    Examples
    --------
    >>> # Daily aggregation per deployment
    >>> resample_dpm(df, "D", group_by=["deploy.name"])

    >>> # Hourly aggregation with site info preserved
    >>> resample_dpm(df, "h", extra_columns=["site.name"])

    """
    df = is_dpm_col(df)
    df = add_utc(df, ["start_datetime"], frq)

    # Determine grouping columns
    if group_by is None:
        group_by = ["start_datetime"]

    # Build aggregation dictionary
    agg_dict = build_aggregation_dict(
        df,
        base_agg={"DPM": "sum"},
        extra_columns=extra_columns,
    )

    return df.groupby(group_by).agg(agg_dict).reset_index()


def deploy_period(
    df: DataFrame,
    col_timestamp: str = "start_datetime",
    col_deployment: str = "deploy.name",
) -> DataFrame:
    """Extract start and end timestamps for each deployment.

    Parameters
    ----------
    df: DataFrame
        Input dataframe with parsed timestamps.
    col_timestamp: str, default="start_datetime"
        Name of the timestamp column.
    col_deployment: str, default="deploy.name"
        Name of the deployment identifier column.

    Returns
    -------
    DataFrame
        DataFrame with columns: [col_deployment, 'Deb', 'Fin'].

    """
    return (
        df.groupby([col_deployment])
        .agg(Deb=(col_timestamp, "first"), Fin=(col_timestamp, "last"))
        .reset_index()
    )


def first_last(
    df: DataFrame,
    col_timestamp: str = "start_datetime",
    col_deployment: str = "deploy.name",
    date_formats: list[str] | None = None,
) -> DataFrame:
    """Isolate beginning and end of every deployment.

    Parameters
    ----------
    df: DataFrame
        CPOD result DataFrame.
    col_timestamp: str, default="start_datetime"
        Name of the timestamps column.
    col_deployment: str, default="deploy.name"
        Name of the deployment identifier column.
    date_formats: list[str], optional
        List of date formats to try for parsing.

    Returns
    -------
    DataFrame
        DataFrame with deployment periods (Deb, Fin).

    """
    df_parsed = parse_timestamps(df, col_timestamp, date_formats)
    return deploy_period(df_parsed, col_timestamp, col_deployment)

def actual_data(
    df: DataFrame,
    meta: DataFrame,
) -> DataFrame:
    """Create a table with beginning and end of every deployment using metadata.

    Parameters
    ----------
    df: DataFrame
        Contains beginning and end for every deployment.
    meta: DataFrame
        Contains metadata for every deployment.

    Returns
    -------
    DataFrame
        DataFrame with corrected deployment periods (Deb, Fin).

    """
    required_columns(
        df,["deploy.name","ChunkEnd"])
    required_columns(
        meta,["deploy.name", "deployment_date","recovery_date"])

    beg_end = first_last(df, "ChunkEnd")

    beg_end = add_utc(beg_end, ["Deb", "Fin"])

    final = beg_end.merge(meta[["deployment_date","recovery_date","deploy.name"]],
                          on = "deploy.name", how="left")
    final.loc[final["Deb"] < final["deployment_date"], "Deb"] = final["deployment_date"]
    final.loc[final["Fin"] > final["recovery_date"], "Fin"] = final["recovery_date"]
    final.loc[final["Deb"] > final["Fin"], ["Deb", "Fin"]] = None
    final = final.sort_values(by=["Deb"])
    return final.drop(["deployment_date", "recovery_date"], axis=1)

def create_matrix(
    df: DataFrame,
    group_cols: list,
    agg_cols: list,
)-> DataFrame:
    """Create a stats matrix (mean & std).

    Parameters
    ----------
    df : DataFrame
        Extended frame with raw data to calculate stats for
    group_cols : list
        Additional columns to group by
    agg_cols : list
        Columns to aggregate

    Returns
    -------
    Give a matrix of the data in [agg_cols] grouped by [group_cols].

    """
    matrix = df.groupby(group_cols).agg({
        col: ["mean", "std"] for col in agg_cols
    })
    matrix = matrix.reset_index()

    matrix.columns = group_cols + [f"{col}_{stat}"
                                    for col in agg_cols
                                    for stat in ["mean", "std"]]
    return matrix


def extract_site(
    df: DataFrame,
) -> DataFrame:
    """Create new columns: site.name and campaign.name, in order to match the metadata.

    Parameters
    ----------
    df: DataFrame
        All values concatenated

    Returns
    -------
    DataFrame
        The same dataframe with two additional columns.

    """
    required_columns(df, ["deploy.name"])
    df[["site.name", "campaign.name"]] = df["deploy.name"].str.split("_", expand=True)
    return df


def percent_calc(
    data: DataFrame,
    time_unit: str | None = None,
) -> DataFrame:
    """Calculate percentage of clicks, feeding buzzes and positive hours to detection.

    Computed on the entire effort and for every site.

    Parameters
    ----------
    data: DataFrame
        All values concatenated

    time_unit: str
        Time unit you want to group your data in

    Returns
    -------
    DataFrame

    """
    group_cols = ["site.name"]
    if time_unit is not None:
        group_cols.insert(0, time_unit)

    # Aggregate and compute metrics
    df = (
        data.groupby(group_cols)
        .agg(
            {
                "DPH": "sum",
                "DPM": "sum",
                "Day": "size",
                "Foraging": "sum",
            },
        )
        .reset_index()
    )

    df["%click"] = df["DPM"] * 100 / (df["Day"] * 60)
    df["%DPH"] = df["DPH"] * 100 / df["Day"]
    df["FBR"] = df["Foraging"] * 100 / df["DPM"]
    df["%buzzes"] = df["Foraging"] * 100 / (df["Day"] * 60)
    return df


def site_percent(df: DataFrame, metric: str) -> None:
    """Plot a graph with percentage of minutes positive to detection for every site.

    Parameters
    ----------
    df: DataFrame
        All percentages grouped by site
    metric: str
        Type of percentage you want to show on the graph

    """
    ax = sns.barplot(
        data=df,
        x="site.name",
        y=metric,
        hue="site.name",
        dodge=False,
        palette=site_colors,
    )
    ax.set_title(f"{metric} per site")
    ax.set_ylabel(f"{metric}")
    if metric in ("%buzzes", "FBR"):
        for _, bar in enumerate(ax.patches):
            bar.set_hatch("/")
    plt.show()


def year_percent(df: DataFrame, metric: str) -> None:
    """Plot a graph with the percentage of minutes positive to detection per site/year.

    Parameters
    ----------
    df: DataFrame
        All percentages grouped by site and year
    metric: str
        Type of percentage you want to show on the graph

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(14, 2.5 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]
    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site]
        ax = axs[i]
        ax.bar(
            site_data["Year"],
            site_data[metric],
            label=f"Site {site}",
            color=site_colors.get(site, "gray"),
        )
        ax.set_title(f"{site}")
        ax.set_ylim(0, max(df[metric]) + 0.2)
        ax.set_ylabel(metric)
        if i != 3:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Year")
        if metric in ("%buzzes", "FBR"):
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")
    fig.suptitle(f"{metric} per year", fontsize=16)
    plt.show()


def ym_percent(df: DataFrame, metric: str) -> None:
    """Plot a graph with the percentage of DPM per site/month-year.

    Parameters
    ----------
    df: DataFrame
        All percentages grouped by site and month per year
    metric: str
        Type of percentage you want to show on the graph

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(14, 2.5 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]
    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site]
        ax = axs[i]
        bar_colors = site_data["Season"].map(season_color).fillna("gray")
        ax.bar(
            site_data["YM"],
            site_data[metric],
            label=f"Site {site}",
            color=bar_colors,
            width=25,
        )
        ax.set_title(f"{site} - Percentage of minutes postitive to detection per month")
        ax.set_ylim(0, max(df[metric]) + 0.2)
        ax.set_ylabel(metric)
        if i != 3:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Months")
        if metric in ("%buzzes", "FBR"):
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")
    legend_elements = [
        Patch(facecolor=col, edgecolor="black", label=season.capitalize())
        for season, col in season_color.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        title="Seasons",
        bbox_to_anchor=(0.95, 0.95),
    )
    fig.suptitle(f"{metric} per month", fontsize=16)
    plt.show()


def month_percent(df: DataFrame, metric: str) -> None:
    """Plot a graph with the percentage of minutes positive to detection per site/month.

    Parameters
    ----------
    df: DataFrame
        All percentages grouped by site and month
    metric: str
        Type of percentage you want to show on the graph

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(14, 2.5 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]
    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site]
        ax = axs[i]
        ax.bar(
            site_data["Month"],
            site_data[metric],
            label=f"Site {site}",
            color=site_colors.get(site, "gray"),
        )
        ax.set_title(f"{site} - Percentage of minutes postitive to detection per month")
        ax.set_ylim(0, max(df[metric]) + 0.2)
        ax.set_ylabel(metric)
        ax.set_xticks(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Agu",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
        )
        if i != 3:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Months")
        if metric in ("%buzzes", "FBR"):
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")
    fig.suptitle(f"{metric} per month", fontsize=16)
    plt.show()


def hour_percent(df: DataFrame, metric: str) -> None:
    """Plot a graph with the percentage of minutes positive to detection per site/hour.

    Parameters
    ----------
    df: DataFrame
        All percentages grouped by site and hour
    metric: str
        Type of percentage you want to show on the graph

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(14, 2.5 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]
    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site]
        ax = axs[i]
        ax.bar(
            site_data["Hour"],
            site_data[metric],
            label=f"Site {site}",
            color=site_colors.get(site, "gray"),
        )
        ax.set_title(
            f"Site {site} - Percentage of minutes positive to detection per hour",
        )
        ax.set_ylim(0, max(df[metric]) + 0.2)
        ax.set_ylabel(metric)
        if i != 3:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Hour")
        if metric in ("%buzzes", "FBR"):
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")
    fig.suptitle(f"{metric} per hour", fontsize=16)
    plt.show()


def calendar(
    meta: DataFrame,
    data: DataFrame,
) -> None:
    """Produce the calendar of the given data.

    Parameters
    ----------
    meta: DataFrame
        metadatax file
    data: DataFrame
        cpod file from all sites and phases

    Returns
    -------
    Return a plot of all deployments and associated data.

    """
    # format the dataframe
    meta["deployment_date"] = to_datetime(meta["deployment_date"])
    meta["recovery_date"] = to_datetime(meta["recovery_date"])
    meta = meta.sort_values(["deploy.name", "deployment_date"]).reset_index(drop=True)
    data = data.sort_values(["deploy.name", "Deb"]).reset_index(drop=True)
    df_fusion = data.merge(
        meta[["deploy.name", "deployment_date", "recovery_date"]],
        on=["deploy.name"],
        how="outer",
    )

    df_fusion["Deb"] = df_fusion["Deb"].fillna(df_fusion["deployment_date"])
    df_fusion["Fin"] = df_fusion["Fin"].fillna(df_fusion["deployment_date"])

    df_fusion[["Site", "Phase"]] = df_fusion["deploy.name"].str.split("_", expand=True)
    df_fusion["color"] = df_fusion["Site"].map(site_colors)

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 4))

    sites = sorted(df_fusion["Site"].unique(), reverse=True)
    site_mapping = {site: idx for idx, site in enumerate(sites)}

    for _, row in df_fusion.iterrows():
        y_pos = site_mapping[row["Site"]]
        ax.broken_barh(
            [(row["deployment_date"], row["recovery_date"] - row["deployment_date"])],
            (y_pos - 0.3, 0.6),
            facecolors="#F5F5F5",
            edgecolors="black",
            linewidth=0.8,
        )

        if notna(row["Deb"]) and notna(row["Fin"]) and row["Fin"] > row["Deb"]:
            ax.broken_barh(
                [(row["Deb"], row["Fin"] - row["Deb"])],
                (y_pos - 0.15, 0.3),
                facecolors=row["color"],
                edgecolors="black",
                linewidth=0.8,
            )


    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(sites, fontsize=12)

    legend_elements = [
        Patch(facecolor="#F5F5F5", edgecolor="black", label="Deployment"),
    ]
    for site, color in site_colors.items():
        if site in sites:
            legend_elements.append(
                Patch(facecolor=color, edgecolor="black", label=f"{site}"),
            )

    ax.legend(handles=legend_elements, loc="upper left", fontsize=11, frameon=True)
    # Layout final
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def hist_mean_m(
    df: DataFrame,
    metric_mean: str,
    metric_std: str,
    ylabel: str | None = None,
    title_suffix: str | None = None,
) -> None:
    """Produce a histogram of the given data.

    It shows mean and standard deviation of the metric.

    Parameters
    ----------
    df: DataFrame
        All data grouped by site and month
    metric_mean: str
        Column name for the mean values (e.g., "%click_mean")
    metric_std: str
        Column name for the standard deviation values (e.g., "%click_std")
    ylabel: str, optional
        Label for y-axis. If None, uses metric_mean
    title_suffix: str, optional
        Suffix for the main title. If None, uses metric_mean

    Returns
    -------
    Return a plot of all deployments and associated data.

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(14, 3 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]

    # Calculate max for y-axis scaling
    max_value = max(df[metric_mean] + df[metric_std])

    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site]
        ax = axs[i]

        ax.bar(
            x=site_data["Month"],
            height=site_data[metric_mean],
            yerr=site_data[metric_std],
            capsize=4,
            color=site_colors.get(site, "gray"),
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label=f"Site {site}")

        ax.set_title(f"{site}", fontsize=12)
        ax.set_ylim(0, max_value * 1.1)
        ax.set_ylabel(ylabel if ylabel else metric_mean, fontsize=10)

        # Only set x-label on last subplot
        if i == n_sites - 1:
            ax.set_xlabel("Mois", fontsize=10)
            ax.set_xticks(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                [
                    "Jan",
                    "Fev",
                    "Mar",
                    "Avr",
                    "Mai",
                    "Jun",
                    "Jul",
                    "Aou",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ],
            )
        if metric_mean in ("%buzzes_mean", "FBR_mean"):
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")

    fig.suptitle(
        f"{title_suffix if title_suffix else metric_mean} per month",
        fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def hist_mean_h(
    df: DataFrame,
    metric_mean: str,
    metric_std: str,
    ylabel: str | None = None,
    title_suffix: str | None = None,
) -> None:
    """Produce a histogram of the given data.

    It shows mean and standard deviation of the metric.

    Parameters
    ----------
    df: DataFrame
        All data grouped by site and month
    metric_mean: str
        Column name for the mean values (e.g., "%click_mean")
    metric_std: str
        Column name for the standard deviation values (e.g., "%click_std")
    ylabel: str, optional
        Label for y-axis. If None, uses metric_mean
    title_suffix: str, optional
        Suffix for the main title. If None, uses metric_mean

    Returns
    -------
    Return a plot of all deployments and associated data.

    """
    sites = df["site.name"].unique()
    n_sites = len(sites)
    fig, axs = plt.subplots(n_sites, 1, figsize=(14, 5 * n_sites), sharex=True)
    if n_sites == 1:
        axs = [axs]

    # Calculate max for y-axis scaling
    max_value = max(df[metric_mean] + df[metric_std])

    for i, site in enumerate(sorted(sites)):
        site_data = df[df["site.name"] == site]
        ax = axs[i]

        ax.bar(
            x=site_data["Hour"],
            height=site_data[metric_mean],
            yerr=site_data[metric_std],
            capsize=4,
            color=site_colors.get(site, "gray"),
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            label=f"Site {site}",
        )

        ax.set_title(f"{site}", fontsize=12)
        ax.set_ylim(0, max_value * 1.1)
        ax.set_ylabel(ylabel if ylabel else metric_mean, fontsize=10)
        ax.set_xticks(range(24))

        # Only set x-label on last subplot
        if i == n_sites - 1:
            ax.set_xlabel("Heure", fontsize=10)
        if metric_mean in ("%buzzes_mean", "FBR_mean"):
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")

    fig.suptitle(
        f"{title_suffix if title_suffix else metric_mean} per hour", fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def hist_mean_s(
    df: DataFrame,
    metric_mean: str,
    metric_std: str,
    ylabel: str | None = None,
    title_suffix: str | None = None,
) -> None:
    """Plot bar chart with mean values and error bars (std) per site.

    Parameters
    ----------
    df: DataFrame
        All data grouped by site
    metric_mean: str
        Column name for the mean values (e.g., "FBR_mean")
    metric_std: str
        Column name for the standard deviation values (e.g., "FBR_std")
    ylabel: str, optional
        Label for y-axis. If None, uses metric_mean
    title_suffix: str, optional
        Suffix for the title. If None, uses metric_mean
    add_hatch: bool, optional
        Add hatching pattern to bars (useful for FBR, %buzzes). Default False

    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by site and calculate means if needed
    plot_data = df.groupby("site.name")[[metric_mean, metric_std]].mean().reset_index()

    x_pos = range(len(plot_data))

    # Create bars
    bars = ax.bar(
        x=x_pos,
        height=plot_data[metric_mean],
        color=[site_colors.get(site, "gray") for site in plot_data["site.name"]],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5)

    # Add hatching if requested
    if metric_mean in ("%buzzes_mean", "FBR_mean"):
        for _, bar in enumerate(ax.patches):
            bar.set_hatch("/")

    # Add error bars
    for i, (_, row) in enumerate(plot_data.iterrows()):
        # Ensure error bar doesn't go below zero
        yerr_lower = min(row[metric_mean], row[metric_std])
        yerr_upper = row[metric_std]
        ax.errorbar(
            i,
            row[metric_mean],
            yerr=[[yerr_lower], [yerr_upper]],
            fmt="none",
            color="black",
            capsize=5,
            linewidth=2,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_data["site.name"])
    ax.set_title(f"{title_suffix if title_suffix else metric_mean} per site",
                 fontsize=12)
    ax.set_ylabel(ylabel if ylabel else metric_mean, fontsize=10)
    ax.set_xlabel("Site", fontsize=10)

    plt.tight_layout()
    plt.show()