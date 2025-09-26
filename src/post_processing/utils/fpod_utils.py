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
    Series,
    Timedelta,
    concat,
    date_range,
    notna,
    read_csv,
    read_excel,
    to_datetime,
)

from post_processing.utils.core_utils import get_coordinates, get_sun_times

if TYPE_CHECKING:
    import pytz


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
            for entry in df["Date heure"]
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

        if extra_columns:
            for col in extra_columns:
                if col in df_deploy.columns:
                    result[col] = df_deploy[col].tolist()

        results.append(result)

    return concat(results, ignore_index=True)


def usable_data_phase(
    d_meta: DataFrame,
    df: DataFrame,
    dpl: str,
) -> DataFrame:
    """Calculate the percentage of usable data.

    Considering the deployment dates and the collected data.

    Parameters
    ----------
    df: DataFrame
        CPOD result DataFrame
    d_meta: DataFrame
        Metadata DataFrame with deployments information (previously exported as json)
    dpl: str
        Deployment of interest where percentage of usable data will be calculated

    Returns
    -------
    DataFrame
        Returns the percentage of usable datas in the chosen phase

    """
    d_meta.loc[:, ["deployment_date", "recovery_date"]] = d_meta[
        ["deployment_date", "recovery_date"]
    ].apply(
        to_datetime,
    )
    df["start_datetime"] = to_datetime(df["start_datetime"])

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
    raw_data: DataFrame,
    metadata: DataFrame,
    col_deploy_name: str = "deploy.name",
    col_timestamp: str = "start_datetime",
    col_debut: str = "deployment_date",
    col_fin: str = "recovery_date",
) -> DataFrame:
    """Filter data to keep only the ones corresponding to a deployment.

    Parameters
    ----------
    raw_data : DataFrame
        Dataframe containing deploy.name et timestamp
    metadata : DataFrame
        Metadata containing deploy.name, deployment_date, recovery_date
    col_deploy_name : str
        Name of the deployment name column (default: 'deploy.name')
    col_timestamp : str
        Name of the timestamps column in raw_data (default: 'start_datetime')
    col_debut : str
        Name of the deployment column in metadata (default: 'deployment_date')
    col_fin : str
        Name of the recovery column in metadata (default: 'recovery_date')

    Returns
    -------
    DataFrame
        Filtered data containing only rows in deployment periods

    """
    required_raw = [col_deploy_name, col_timestamp]
    required_meta = [col_deploy_name, col_debut, col_fin]
    for col in required_raw:
        if col not in raw_data.columns:
            msg = f"'{col}' not found in raw_data"
            raise ValueError(msg)
    for col in required_meta:
        if col not in metadata.columns:
            msg = f"'{col}' not found in metadata"
            raise ValueError(msg)

    # Convert to datetime
    raw = raw_data.copy()
    meta = metadata.copy()
    raw[col_timestamp] = to_datetime(raw[col_timestamp], errors="coerce")
    meta[col_debut] = to_datetime(meta[col_debut], errors="coerce")
    meta[col_fin] = to_datetime(meta[col_fin], errors="coerce")

    dfm = raw.merge(
        meta[[col_deploy_name, col_debut, col_fin]],
        on=col_deploy_name,
        how="left",
    )

    out = dfm[
        (dfm[col_timestamp] >= dfm[col_debut])
        & (dfm[col_timestamp] <= dfm[col_fin])
        & dfm[col_timestamp].notna()
        & dfm[col_debut].notna()
        & dfm[col_fin].notna()
    ].copy()

    columns_to_drop = [
        col for col in [col_debut, col_fin] if col not in raw_data.columns
    ]
    if columns_to_drop:
        out = out.drop(columns=columns_to_drop)

    return out.sort_values([col_deploy_name, col_timestamp]).reset_index(drop=True)


def format_calendar(path: Path) -> DataFrame:
    """Format calendar.

    Parameters
    ----------
    path: Path
        Excel calendar path

    """
    df_calendar = read_excel(path)
    df_calendar = df_calendar[df_calendar["Site group"] == "Data"].copy()

    return df_calendar.rename(
        columns={
            "Start": "start_datetime",
            "Stop": "end_datetime",
            "Site": "site.name",
        },
    )


def assign_phase(
    meta: DataFrame,
    data: DataFrame,
    site: str,
) -> DataFrame:
    """Add a column to an APLOSE DataFrame to specify the name of the phase.

    The name of the phase is attributed according to metadata.

    Parameters
    ----------
    meta: DataFrame
        Metadata dataframe with deployments information (previously exported as json).
    data: DataFrame
        Contain positive hours to detections.
    site: str
        Name of the site you wish to assign phases to.

    Returns
    -------
    DataFrame
        The same dataframe with the column Phase.

    """
    data["start_datetime"] = to_datetime(data["start_datetime"], utc=True)
    meta["deployment_date"] = to_datetime(meta["deployment_date"], utc=True)
    meta["recovery_date"] = to_datetime(meta["recovery_date"], utc=True)

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
                data.loc[j, "name"] = (
                    f"{meta_row['site.name']}_{meta_row['campaign.name']}"
                )
            j += 1
    return data


def assign_phase_simple(
    meta: DataFrame,
    data: DataFrame,
) -> DataFrame:
    """Add column to an Aplose DataFrame to specify the phase, according to metadata.

    Parameters
    ----------
    meta: DataFrame
        Metadata dataframe with deployments information (previously exported as json).
    data: DataFrame
        Contain positive hours to detections.

    Returns
    -------
    DataFrame
        The same dataframe with the column Phase.

    """
    data["start_datetime"] = to_datetime(data["start_datetime"], utc=True)
    data["end_datetime"] = to_datetime(data["end_datetime"], dayfirst=True, utc=True)
    meta["deployment_date"] = to_datetime(meta["deployment_date"], utc=True)
    meta["recovery_date"] = to_datetime(meta["recovery_date"], utc=True)
    meta["deployment_date"] = meta["deployment_date"].dt.floor("d")
    meta["recovery_date"] = meta["recovery_date"].dt.floor("d")

    data["name"] = None
    for site in data["deploy.name"].unique():
        site_meta = meta[meta["deploy.name"] == site]
        site_data = data[data["deploy.name"] == site]

        for _, meta_row in site_meta.iterrows():
            time_filter = (
                meta_row["deployment_date"] <= site_data["start_datetime"]
            ) & (site_data["start_datetime"] < meta_row["recovery_date"])
            data.loc[site_data.index[time_filter], "name"] = meta_row["name"]

    return data


def generate_hourly_detections(meta: DataFrame, site: str) -> DataFrame:
    """Create a DataFrame with one line per hour between start and end dates.

    Keep the number of detections per hour between these dates.

    Parameters
    ----------
    meta: DataFrame
        Metadata dataframe with deployments information (previously exported as json)
    site: str
        A way to isolate the site you want to work on.

    Returns
    -------
    DataFrame
        A full period of time with positive and negative hours to detections.

    """
    df_meta = meta[meta["site.name"] == site].copy()
    df_meta["deployment_date"] = to_datetime(df_meta["deployment_date"])
    df_meta["recovery_date"] = to_datetime(df_meta["recovery_date"])
    df_meta["deployment_date"] = df_meta["deployment_date"].dt.floor("h")
    df_meta["recovery_date"] = df_meta["recovery_date"].dt.floor("h")
    df_meta = df_meta.sort_values(by=["deployment_date"])

    records = [
        {"name": row["name"], "start_datetime": date}
        for _, row in df_meta.iterrows()
        for date in date_range(
            start=row["deployment_date"],
            end=row["recovery_date"],
            freq="h",
        )
    ]

    return DataFrame(records)


def build_hour_range(dph: DataFrame) -> DataFrame:
    """Create a DataFrame with one line per hour between start and end dates.

    Keep the number of detections per hour between these dates.

    Parameters
    ----------
    dph: pd.DataFrame
        Metadata dataframe with deployments information (previously exported as json)

    Returns
    -------
    pd.DataFrame
        A full period of time with positive and negative hours to detections.

    """
    dph["Date heure"] = to_datetime(dph["Date heure"], dayfirst=True)

    deploy_ranges = (
        dph.groupby("deploy.name")["Date heure"]
        .agg(start="min", end="max")
        .reset_index()
    )

    all_ranges = []
    for _, row in deploy_ranges.iterrows():
        hours = date_range(row["start"], row["end"], freq="h")
        tmp = DataFrame(
            {
                "deploy.name": row["deploy.name"],
                "Date heure": hours,
            },
        )
        all_ranges.append(tmp)

    return concat(all_ranges, ignore_index=True)


def merging_tab(meta: DataFrame, data: DataFrame) -> DataFrame:
    """Create a DataFrame with one line per hour between start and end dates.

    Keep the number of detections per hour between these dates.

    Parameters
    ----------
    meta: DataFrame
        Metadata with deployments information (previously exported as json)
    data: DataFrame
        Contain positive hours to detections

    Returns
    -------
    DataFrame
        A full period of time with positive and negative hours to detections.

    """
    data["start_datetime"] = to_datetime(data["start_datetime"], utc=True)
    meta["start_datetime"] = to_datetime(meta["start_datetime"], utc=True)

    deploy_detec = data["deploy.name"].unique()
    df_filtered = meta[meta["deploy.name"].isin(deploy_detec)]

    output = df_filtered.merge(
        data[["deploy.name", "start_datetime", "DPM"]],
        on=["deploy.name", "start_datetime"],
        how="outer",
    )
    output["DPM"] = output["DPM"].fillna(0)

    output["Day"] = output["start_datetime"].dt.day
    output["Month"] = output["start_datetime"].dt.month
    output["Year"] = output["start_datetime"].dt.year
    output["hour"] = output["start_datetime"].dt.hour

    return output


def feeding_buzz(df: DataFrame, species: str) -> DataFrame:
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
    df["microsec_formatted"] = df["microsec"].apply(lambda x: f"{x:.6f}")

    df["Time"] = df["Minute"].astype(str) + ":" + df["microsec_formatted"].astype(str)

    df["Time"] = to_datetime(df["Time"], dayfirst=True)

    df = df.sort_values(by="Time").reset_index(drop=True)
    df["ICI"] = df["Time"].diff().dt.total_seconds()

    df["Buzz"] = 0
    if species == "Marsouin":
        feeding_idx = df.index[df["ICI"] < 0.01]
    elif species == "Commerson":
        feeding_idx = df.index[df["ICI"] <= 0.005]
    else:
        msg = "This species is not supported"
        raise ValueError(msg)

    df.loc[feeding_idx, "Buzz"] = 1
    df.loc[feeding_idx - 1, "Buzz"] = 1
    df.loc[df.index < 0, "Buzz"] = 0

    df["start_datetime"] = df["Time"].dt.floor("min")
    df["start_datetime"] = to_datetime(df["start_datetime"], dayfirst=False, utc=True)
    f = df.groupby(["start_datetime"])["Buzz"].sum().reset_index()

    f["Foraging"] = (f["Buzz"] != 0).astype(int)

    return f


def assign_daytime(
    df: DataFrame,
) -> DataFrame:
    """Assign datetime categories to events.

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
    start = df.iloc[0]["Time"]
    stop = df.iloc[-1]["Time"]
    lat, lon = get_coordinates()
    _, _, dawn, day, dusk, night = get_sun_times(start, stop, lat, lon)
    dawn = Series(dawn, name="dawn")
    day = Series(day, name="day")
    dusk = Series(dusk, name="dusk")
    night = Series(night, name="night")
    jour = concat([day, night, dawn, dusk], axis=1)

    for i, row in df.iterrows():
        dpm_i = row["Time"]
        if notna(dpm_i):  # Check if time is not NaN
            jour_i = jour[
                (jour["dusk"].dt.year == dpm_i.year)
                & (jour["dusk"].dt.month == dpm_i.month)
                & (jour["dusk"].dt.day == dpm_i.day)
            ]
            if not jour_i.empty:  # Ensure there"s a matching row
                jour_i = jour_i.iloc[0]  # Extract first match
                if dpm_i <= jour_i["day"]:
                    df.loc[i, "REGIME"] = 1
                elif dpm_i < jour_i["dawn"]:
                    df.loc[i, "REGIME"] = 2
                elif dpm_i < jour_i["dusk"]:
                    df.loc[i, "REGIME"] = 3
                elif dpm_i > jour_i["night"]:
                    df.loc[i, "REGIME"] = 1
                elif dpm_i > jour_i["dusk"]:
                    df.loc[i, "REGIME"] = 4
                else:
                    df.loc[i, "REGIME"] = 1

    return df


def fb_folder(folder_path: Path, species: str) -> DataFrame:
    """Process a folder containing all CPOD/FPOD feeding buzz detection files.

    Apply the feeding buzz function to these files.

    Parameters
    ----------
    folder_path: Path
        Path to the folder.
    species: str
        Select the species to use between porpoise and Commerson's dolphin

    Returns
    -------
    DataFrame
       Compiled feeding buzz detection positive minutes.

    """
    all_files = list(Path(folder_path).rglob("*.txt"))
    all_data = []

    for file in all_files:
        file_path = folder_path / file
        df = read_csv(file_path, sep="\t")
        processed_df = feeding_buzz(df, species)
        all_data.append(processed_df)

    return concat(all_data, ignore_index=True)


colors = {
    "Site A Haute": "#118B50",
    "Site B Heugh": "#5DB996",
    "Site C Chat": "#B0DB9C",
    "Site D Simone": "#E3F0AF",
    "CA4": "#FF0066",
    "Walde": "#934790",
}

season_color = {
    "spring": "#C5E0B4",
    "summer": "#FCF97F",
    "autumn": "#ED7C2F",
    "winter": "#B4C7E8",
}


def extract_site(df: DataFrame) -> DataFrame:
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
    df[["site.name", "campaign.name"]] = df["deploy.name"].str.split("_", expand=True)
    return df


def percent_calc(data: DataFrame, time_unit: str | None = None) -> DataFrame:
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
        palette=colors,
    )
    ax.set_title(f"{metric} per site")
    ax.set_ylabel(f"{metric}")
    if metric == "%buzzes":
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
            color=colors.get(site, "gray"),
        )
        ax.set_title(f"{site}")
        ax.set_ylim(0, max(df[metric]) + 0.2)
        ax.set_ylabel(metric)
        if i != 3:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Year")
        if metric == "%buzzes":
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
        if metric == "%buzzes":
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")
    legend_elements = [
        Patch(facecolor=season_color, edgecolor="black", label=season.capitalize())
        for season, season_color in season_color.items()
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
            color=colors.get(site, "gray"),
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
        if metric == "%buzzes":
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
            color=colors.get(site, "gray"),
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
        if metric == "%buzzes":
            for _, bar in enumerate(ax.patches):
                bar.set_hatch("/")
    fig.suptitle(f"{metric} per hour", fontsize=16)
    plt.show()


def csv_folder(folder_path: str | Path, **kwargs) -> DataFrame:
    """Process a folder containing data files and concatenate them.

    Parameters
    ----------
    folder_path: Union[str, Path]
        Path to the folder containing files.
    **kwargs: dict
        Additional parameters for pd.read_csv (sep, skiprows, etc.)

    Returns
    -------
    DataFrame
        Concatenated dataframe with all files data and file column.

    Raises
    ------
    ValueError
        If file_format is not supported or no files found.
    FileNotFoundError
        If folder_path doesn't exist.

    """
    folder_path = Path(folder_path)

    # Folder validation
    if not folder_path.exists():
        raise FileNotFoundError

    if not folder_path.is_dir():
        message = f"{folder_path} is not a directory."
        raise ValueError(message)

    # Configuration
    default_params = {"sep": ";", "encoding": "latin-1"}

    # Parameters fusion
    read_params = {**default_params, **kwargs}

    # File research
    files = list(folder_path.rglob("*csv"))

    if not files:
        msg = f"No CSV file found in {folder_path}"
        raise ValueError(msg)

    all_data = []

    for file in files:
        df = read_csv(file, **read_params)
        df["deploy.name"] = file.stem
        all_data.append(df)

    if not all_data:
        msg = f"No valid CSV file found in {folder_path}"
        raise ValueError(msg)

    return concat(all_data, ignore_index=True)


def dpm_to_dp10m(
    df: DataFrame,
    extra_columns: list | None = None,
) -> DataFrame:
    """From CPOD result with a line per minute (DPM) to one line per 10 minutes (DP10M).

    Parameters
    ----------
    df: DataFrame
        CPOD result DataFrame, DPM.
    extra_columns: list
        Additional columns added from df to data.

    Returns
    -------
    DataFrame
        DPM10M Dataframe.

    """
    df = df.copy()
    df["ChunkEnd"] = to_datetime(df["ChunkEnd"], dayfirst=True)

    df["Date heure"] = df["ChunkEnd"].dt.floor("10min")

    agg_dict = {"DPM": "sum"}

    if extra_columns:
        for col in extra_columns:
            if col in df.columns:
                agg_dict[col] = "first"
            else:
                logging.warning(f"Column '{col}' does not exist and will be ignored.")

    return df.groupby("Date heure").agg(agg_dict).reset_index()


def dpm_to_dph(
    df: DataFrame,
    extra_columns: list | None = None,
) -> DataFrame:
    """From CPOD result with a line per minute (DPM) to one line per hour (DPH).

    Parameters
    ----------
    df: pd.DataFrame
        CPOD result DataFrame
    extra_columns: list
        Additional columns added from df to data

    Returns
    -------
    pd.DataFrame
        DPH Dataframe.

    """
    df = df.copy()
    df["ChunkEnd"] = to_datetime(df["ChunkEnd"], dayfirst=True)

    # Truncate column
    df["Date heure"] = df["ChunkEnd"].dt.floor("h")

    agg_dict = {"DPM": "sum"}

    if extra_columns:
        for col in extra_columns:
            if col in df.columns:
                agg_dict[col] = "first"
            else:
                logging.warning(f"Column '{col}' does not exist and will be ignored.")

    return df.groupby("Date heure").agg(agg_dict).reset_index()


def dpm_to_dpd(
    df: DataFrame,
    extra_columns: list | None = None,
) -> DataFrame:
    """From CPOD result with a line per minute (DPM) to one line per day (DPD).

    Parameters
    ----------
    df: pd.DataFrame
        CPOD result DataFrame
    extra_columns: list
        Additional columns added from df to data

    Returns
    -------
    pd.DataFrame
        DPD Dataframe.

    """
    df = df.copy()
    df["ChunkEnd"] = to_datetime(df["ChunkEnd"], dayfirst=True)

    # Truncate column
    df["Date heure"] = df["ChunkEnd"].dt.floor("D")

    agg_dict = {"DPM": "sum"}

    if extra_columns:
        for col in extra_columns:
            if col in df.columns:
                agg_dict[col] = "first"
            else:
                logging.warning(f"Column '{col}' does not exist and will be ignored.")

    return df.groupby("Date heure").agg(agg_dict).reset_index()


def date_format(
    df: DataFrame,
) -> DataFrame:
    """Change the date time format of a DataFrame to "%d/%m/%Y %H:%M:%S".

    Parameters
    ----------
    df: pd.DataFrame
        CPOD result DataFrame

    Returns
    -------
    Return the same dataframe with a different time format.

    """
    df["Date heure"] = to_datetime(df["Date heure"], format="%Y-%m-%d %H:%M:%S")
    df["Date heure"] = df["Date heure"].dt.strftime("%d/%m/%Y %H:%M:%S")

    return df


def actual_data(
    df: DataFrame,
    col_timestamp: str = "start_datetime",
) -> DataFrame:
    """Create a table with beginning and end of every deployment.

    Parameters
    ----------
    col_timestamp
    df: pd.DataFrame
        CPOD result DataFrame
    col_timestamp: str
        Name of the timestamps column in raw_data (default: 'start_datetime')

    Returns
    -------
    pd.DataFrame
        Simple Dataframe with beginning and end columns.

    """
    df[col_timestamp] = df[col_timestamp].apply(
        lambda x: strptime_from_text(
            x, ["%Y-%m-%dT%H:%M:%S:%Z", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y %H:%M"],
        ),
    )
    return (
        df.groupby(["deploy.name"])
        .agg(Début=(col_timestamp, "first"), Fin=(col_timestamp, "last"))
        .reset_index()
    )


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
    data = data.sort_values(["deploy.name", "Début"]).reset_index(drop=True)
    df_fusion = data.merge(
        meta[["deploy.name", "deployment_date", "recovery_date"]],
        on=["deploy.name"],
        how="outer",
    )

    df_fusion["Début"] = df_fusion["Début"].fillna(df_fusion["deployment_date"])
    df_fusion["Fin"] = df_fusion["Fin"].fillna(df_fusion["deployment_date"])

    df_fusion[["Site", "Phase"]] = df_fusion["deploy.name"].str.split("_", expand=True)
    df_fusion["color"] = df_fusion["Site"].map(colors)

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

        if row["Début"] != row["deployment_date"]:
            ax.broken_barh(
                [(row["Début"], row["Fin"] - row["Début"])],
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
    for site, color in colors.items():
        if site in sites:
            legend_elements.append(
                Patch(facecolor=color, edgecolor="black", label=f"{site}"),
            )

    ax.legend(handles=legend_elements, loc="upper left", fontsize=11, frameon=True)
    # Layout final
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def f_b2(df: DataFrame, species: str) -> DataFrame:
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

    if species == "Marsouin":  # Nuuttila et al., 2013
        df["Buzz"] = (df["ICI"].between(0, 0.01)).astype(int)
    elif species == "Commerson":
        df["Buzz"] = (df["ICI"].between(0, 0.005)).astype(int)
    else:
        msg = "This species is not supported"
        raise ValueError(msg)

    df["Minute"] = to_datetime(df["Minute"], dayfirst=False, utc=True)
    f = df.groupby(["Minute"])["Buzz"].sum().reset_index()

    # df['datetime'] = to_datetime('1900-01-01') + to_timedelta(df['Minute'], unit='min')
    # + to_timedelta(df['microsec'], unit='us') - to_timedelta(2, unit='D')

    f["Foraging"] = (f["Buzz"] != 0).astype(int)

    return f
