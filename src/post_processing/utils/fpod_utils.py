"""FPOD/ CPOD processing functions."""

from __future__ import annotations

import logging
from itertools import cycle
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from numpy import (
    arange,
    argsort,
    exp,
    linspace,
    log,
    nan,
    ndarray,
    sort,
    sqrt,
    zeros,
)
from osekit.utils.timestamp import strftime_osmose_format, strptime_from_text
from pandas import (
    DataFrame,
    Series,
    Timedelta,
    Timestamp,
    concat,
    notna,
    read_csv,
    to_datetime,
    to_numeric,
)
from scipy import stats
from sklearn import mixture

from post_processing.utils.filtering_utils import find_delimiter
from post_processing.utils.plot_utils import set_dynamic_ylim

if TYPE_CHECKING:
    from pathlib import Path

    import pytz
    from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


def pod2aplose(
    df: DataFrame,
    tz: pytz.timezone,
    dataset_name: str,
    label: str,
    annotator: str,
    bin_size: Timedelta,
) -> DataFrame:
    """Format PODs DataFrame to match an APLOSE format.

    Parameters
    ----------
    df: DataFrame
        FPOD result dataframe
    tz: pytz.timezone
        Timezone object to get non-naïve datetime.
    dataset_name: str
        dataset name.
    label: str
        label name.
    annotator: str
        annotator name.
    bin_size: Timedelta
        Duration of the detections in seconds.

    Returns
    -------
    DataFrame
        An APLOSE formatted DataFrame.

    """
    fpod_start_dt = [tz.localize(entry) for entry in df["Datetime"]]

    data = {
        "dataset": [dataset_name] * len(df),
        "filename": list(fpod_start_dt),
        "start_time": [0] * len(df),
        "end_time": [bin_size.total_seconds()] * len(df),
        "min_frequency": [0] * len(df),
        "max_frequency": [0] * len(df),
        "label": [label] * len(df),
        "annotator": [annotator] * len(df),
        "start_datetime": [
            strftime_osmose_format(entry.floor(bin_size)) for entry in fpod_start_dt
        ],
        "end_datetime": [
            strftime_osmose_format(entry.floor(bin_size) + bin_size)
            for entry in fpod_start_dt
        ],
        "type": ["WEAK"] * len(df),
    }

    return DataFrame(data)


def load_pod_folder(
    folder: Path,
    ext: str,
) -> DataFrame:
    """Read POD's result files from a folder.

    Parameters
    ----------
    folder: Path
        Folder's place.
    ext: str
        File extension of result files.

    Returns
    -------
    DataFrame
        Concatenated data.

    Raises
    ------
    ValueError
        If no result files are found.

    """
    if ext not in {"csv", "txt"}:
        msg = f"Invalid file extension: {ext}"
        raise ValueError(msg)

    all_files = sorted(folder.rglob(f"*.{ext}"))

    if not all_files:
        msg = f"No .{ext} files found in {folder}"
        raise ValueError(msg)

    all_data = []
    for file in all_files:
        sep = find_delimiter(file)
        df = read_csv(
            file,
            sep=sep,
            dtype={"microsec": "Int32"},
            usecols=lambda col: col not in {"SmoothedICI", "ICIslope"},
        ).dropna()

        df["dataset"] = file.stem.strip().lower().replace(" ", "_")
        all_data.append(df)

    data = concat(all_data, ignore_index=True)

    if ext == "csv":
        return _process_csv_data(data)
    if ext == "txt":
        return _process_txt_data(data)

    msg = f"Could not load {ext} result folder"
    raise ValueError(msg)


def _process_csv_data(data: DataFrame) -> DataFrame:
    """Process CSV data with filtering and datetime conversion."""
    data_filtered = _filter_csv_data(data)
    data_filtered["Datetime"] = [
        strptime_from_text(dt, "%d/%m/%Y %H:%M") for dt in data_filtered["ChunkEnd"]
    ]
    return data_filtered.sort_values(by=["Datetime"]).reset_index(drop=True)


def _filter_csv_data(data: DataFrame) -> DataFrame:
    """Filter CSV data based on available columns."""
    if "%TimeLost" in data.columns:
        data_filtered = data[data["File"].notna()].copy()
        data_filtered = data_filtered[data_filtered["Nall/m"].notna()]
    else:
        data_filtered = data[data["DPM"] > 0].copy()
        data_filtered = data_filtered[data_filtered["MinsOn"].notna()]

    return data_filtered


def _process_txt_data(data: DataFrame) -> DataFrame:
    """Process TXT data with datetime conversion."""
    data["Datetime"] = data.apply(get_feeding_buzz_datetime, axis=1)
    return data.drop_duplicates().sort_values(by=["Datetime"]).reset_index(drop=True)


def get_feeding_buzz_datetime(row: Series) -> Timestamp:
    """Convert feeding buzz timestamp into a standard Timestamp.

    The conversion method differs based on the POD type.
    """
    exceptions = []
    try:
        return (
            Timestamp("1899-12-30")
            + Timedelta(minutes=row["Minute"])
            + Timedelta(microseconds=row["microsec"])
        )
    except (KeyError, TypeError, ValueError) as e:
        exceptions.append(e)

    try:
        return strptime_from_text(row["Minute"], "%-d/%-m/%Y %H:%M") + Timedelta(
            microseconds=row["microsec"]
        )
    except (KeyError, TypeError, ValueError) as e:
        exceptions.append(e)

    msg = "Could not convert feeding buzz timestamp."
    raise ExceptionGroup(msg, exceptions)


def process_feeding_buzz(
    df: DataFrame,
    species: str,
) -> DataFrame:
    """Process a POD feeding buzz detection DataFrame.

    Give the feeding buzz duration, depending on the studied species
    (`delphinid`, `porpoise` or `commerson`).

    Parameters
    ----------
    df: DataFrame
        Path to cpod.exe feeding buzz file
    species: str
        Select the species to use between porpoise and Commerson's dolphin

    Returns
    -------
    DataFrame
        Containing all ICIs for every positive minute to click

    """
    df["ICI"] = df["Datetime"].diff()
    df["Datetime"] = df["Datetime"].dt.floor("min")

    if species.lower() == "delphinid":  # Herzing et al., 2014
        df["Buzz"] = (
            df["ICI"]
            .between(
                Timedelta(0),
                Timedelta(seconds=0.02),
            )
            .astype(int)
        )
    elif species.lower() == "porpoise":  # Nuuttila et al., 2013
        df["Buzz"] = (
            df["ICI"]
            .between(
                Timedelta(0),
                Timedelta(seconds=0.01),
            )
            .astype(int)
        )
    elif species.lower() == "commerson":  # Reyes Reyes et al., 2015
        df["Buzz"] = (
            df["ICI"]
            .between(
                Timedelta(0),
                Timedelta(seconds=0.005),
            )
            .astype(int)
        )
    else:
        msg = "This species is not supported"
        raise ValueError(msg)

    df_buzz = df.groupby(["Datetime"])["Buzz"].sum().reset_index()
    df_buzz["fbm_count"] = to_numeric(
        df_buzz["Buzz"] != 0,
        downcast="integer",
    ).astype(int)

    return df_buzz


def compute_ici(df: DataFrame) -> DataFrame:
    """Calculate Inter-Click Intervals (in minutes) from feeding buzz timestamps."""
    df = df.copy()
    df["ICI_minutes"] = df["Datetime"].diff().dt.total_seconds() / 60
    return df[df["ICI_minutes"] > 0].dropna(subset=["ICI_minutes"])


def fit_gmm(df: DataFrame, comp: int) -> tuple[DataFrame, ndarray, GaussianMixture]:
    """Fit a GMM on log-transformed ICIs and label clusters by ascending mean.

    Parameters
    ----------
    df: DataFrame
        POD loaded dataframe
    comp: int
        Number of components to apply to the GMM.

    Returns
    -------
    tuple
    Returns the enriched DataFrame, the log-ICI array, and the fitted GMM.

    """
    df = compute_ici(df)
    ici_log = log(df["ICI_minutes"].to_numpy()).reshape(-1, 1)

    gmm = mixture.GaussianMixture(
        n_components=comp,
        covariance_type="full",
        random_state=42,
        n_init=20,
    )
    labels = gmm.fit_predict(ici_log)

    rank = argsort(argsort(gmm.means_.flatten()))
    df["cluster"] = rank[labels]

    return df, ici_log, gmm


def cluster_info(gmm: GaussianMixture) -> list[dict]:
    """Extract per-component statistics from a fitted GMM, sorted by ascending mean."""
    component_names = ["Buzz ICIs", "Regular ICIs", "Long ICIs"]
    sorted_means = sort(gmm.means_, axis=0)

    return [
        {
            "name": component_names[i],
            "id": i,
            "mean_log": sorted_means[i][0],
            "std_log": sqrt(gmm.covariances_[i][0][0]),
            "mean_minutes": exp(sorted_means[i][0]),
            "mean_ms": exp(sorted_means[i][0]) * 60 * 1000,
        }
        for i in range(gmm.n_components)
    ]


def _mixture_density(gmm: GaussianMixture, x_range: ndarray) -> ndarray:
    """Compute the total GMM mixture density over x_range."""
    density = zeros(len(x_range))
    for idx in range(gmm.n_components):
        mean = gmm.means_[idx][0]
        std = sqrt(gmm.covariances_[idx][0][0])
        density += gmm.weights_[idx] * stats.norm.pdf(x_range, mean, std)
    return density


def gmm_feeding_buzz(df: DataFrame, comp: int) -> DataFrame:
    """Categorize ICIs with a GMM and aggregate foraging activity per minute.

    Parameters
    ----------
    df: DataFrame
        POD loaded dataframe
    comp: int
        Number of components to apply to the GMM.

    Returns
    -------
    DataFrame
        A DataFrame of two columns : minute positive to feeding buzz or not and number
        of buzzes.

    """
    df, _, _ = fit_gmm(df, comp)

    df["Buzz"] = nan
    df.loc[df["cluster"] == 0, "Buzz"] = 1
    df["start_datetime"] = df["Datetime"].dt.floor("min")

    df_buzz = df.groupby("start_datetime")["Buzz"].sum().reset_index()
    df_buzz["fbm_count"] = to_numeric(df_buzz["Buzz"] != 0, downcast="integer").astype(
        int
    )
    return df_buzz


def plot_gmm_ici(df: DataFrame, comp: int) -> tuple[plt.Figure, plt.Axes]:
    """Plot a histogram of log ICIs overlaid with GMM components and total mixture."""
    df, ici_log, gmm = fit_gmm(df, comp)

    x_flat = sort(ici_log.flatten())
    x_range = linspace(ici_log.min(), ici_log.max(), 2000)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(
        ici_log,
        bins=200,
        histtype="bar",
        density=True,
        alpha=0.6,
        color="lightgray",
        edgecolor="black",
        linewidth=0.5,
    )

    lines = []
    for idx in range(comp):
        mean, std, weight = (
            gmm.means_[idx, 0],
            sqrt(gmm.covariances_[idx, 0, 0]),
            gmm.weights_[idx],
        )
        (line,) = ax.plot(
            x_flat,
            weight * stats.norm.pdf(x_flat, mean, std),
            label=f"(μ={mean:.2f}, σ={std:.2f})",
        )
        lines.append(line)

    (mix_line,) = ax.plot(
        x_range,
        _mixture_density(gmm, x_range),
        linewidth=2,
        color="black",
        linestyle="--",
        label="Total mixture",
        alpha=0.7,
    )
    lines.append(mix_line)

    ax.set(
        xlabel="Log ICI (log minutes)",
        ylabel="Density",
        title="GMM clustering of Inter-Click Intervals",
    )
    ax.legend(handles=lines)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()
    return fig, ax


def process_timelost(df: DataFrame, threshold: int = 0) -> Series[Any]:
    """Process TimeLost DataFrame.

    Returns relevant columns and reshape into hourly data.

    Parameters
    ----------
    df: DataFrame
        All your Environmental data files.
    threshold: float
        TimeLost threshold.

    Returns
    -------
    %TimeLost DataFrame.

    """
    if threshold not in range(101):
        msg = "Threshold must integer between 0 and 100."
        raise ValueError(msg)

    df["Datetime"] = df["Datetime"].dt.floor("h")
    cols_to_drop = [
        col
        for col in df.columns
        if col
        not in {
            "File",
            "Datetime",
            "Temp",
            "Angle",
            "%TimeLost",
            "Deploy",
        }
    ]
    return (
        df[df["%TimeLost"] <= threshold]
        .drop(
            columns=cols_to_drop,
        )
        .sort_values(["Datetime"])
        .reset_index(drop=True)
    )


def create_matrix(
    df: DataFrame,
    group_cols: list,
    agg_cols: list,
) -> DataFrame:
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
    matrix = df.groupby(group_cols).agg({col: ["mean", "std"] for col in agg_cols})
    matrix = matrix.reset_index()

    matrix.columns = group_cols + [
        f"{col}_{stat}" for col in agg_cols for stat in ["mean", "std"]
    ]
    return matrix


def percent_calc(
    data: DataFrame,
    time_unit: str | None = None,
) -> DataFrame:
    """Calculate % of clicks, feeding buzzes and positive hours to detection.

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
    df = (
        data.groupby(time_unit)
        .agg(
            DP_unit=("DPh", "sum"),
            FB_unit=("FBh", "sum"),
            dpm_count=("dpm_count", "sum"),
            tot_samp=("Day", "size"),
            fbm_count=("fbm_count", "sum"),
        )
        .reset_index()
    )

    df["%click"] = df["dpm_count"] * 100 / (df["tot_samp"] * 60)
    df["%buzzes"] = df["fbm_count"] * 100 / (df["tot_samp"] * 60)
    df["%DPh"] = df["DP_unit"] * 100 / df["tot_samp"]
    df["%FBh"] = df["FB_unit"] * 100 / df["tot_samp"]
    df["FBR"] = df.apply(
        lambda row: (row["fbm_count"] * 100 / row["dpm_count"])
        if row["dpm_count"] > 0
        else 0,
        axis=1,
    )
    return df


def calendar(
    data: DataFrame,
) -> None:
    """Produce the calendar of the given data. Deployments and actual collection of data.

    Parameters
    ----------
    data: DataFrame
        Custom file containing all beginning and end of deployment and recordings.

    """
    for i in data["Site"].unique():
        mask = data["Site"] == i
        data["start_recording"] = to_datetime(data["start_recording"])
        data["end_recording"] = to_datetime(data["end_recording"])
        data["start_deployment"] = to_datetime(data["start_deployment"])
        data["end_deployment"] = to_datetime(data["end_deployment"])

        data.loc[
            mask & (data["start_recording"] < data["start_deployment"]),
            "start_recording",
        ] = data.loc[
            mask & (data["start_recording"] < data["start_deployment"]),
            "start_deployment",
        ]

        data.loc[
            mask & (data["end_recording"] > data["end_deployment"]), "end_recording"
        ] = data.loc[
            mask & (data["end_recording"] > data["end_deployment"]), "end_deployment"
        ]

        data.loc[
            mask & (data["start_recording"] > data["end_recording"]),
            ["start_recording", "end_recording"],
        ] = None
        data = data.sort_values(["Phase", "start_deployment"]).reset_index(drop=True)

    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    site_colors = {site: next(color_cycle) for site in data["Site"].unique()}

    data["color"] = data["Site"].map(site_colors)

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 4))

    sites = sorted(data["Site"].unique(), reverse=True)
    site_mapping = {site: idx for idx, site in enumerate(sites)}

    for _, row in data.iterrows():
        y_pos = site_mapping[row["Site"]]
        ax.broken_barh(
            [
                (
                    row["start_deployment"],
                    row["end_deployment"] - row["start_deployment"],
                ),
            ],
            (y_pos - 0.3, 0.6),
            facecolors="#F5F5F5",
            edgecolors="black",
            linewidth=0.8,
        )

        if (
            notna(row["start_recording"])
            and notna(row["end_recording"])
            and row["end_recording"] > row["start_recording"]
        ):
            ax.broken_barh(
                [
                    (
                        row["start_recording"],
                        row["end_recording"] - row["start_recording"],
                    ),
                ],
                (y_pos - 0.15, 0.3),
                facecolors=row["color"],
                edgecolors="black",
                linewidth=0.8,
            )

    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(sites, fontsize=15)

    plt.xticks(fontsize=15)
    plt.tight_layout()
    plt.show()


default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def get_group_colors(df: DataFrame, group_col: str) -> dict[str, str]:
    """Map each unique value of `group_col` to a stable color.

    Mirrors `get_colors`, but keys colors by group label instead of
    by column position (needed since bars share colors within a group).
    """
    groups = df[group_col].unique()
    color_cycle = cycle(default_colors)
    return {group: next(color_cycle) for group in groups}


def _shade_missing_bars(
    ax: plt.Axes,
    missing_mask: ndarray,
    positions: ndarray | None = None,
    *,
    color: str = "grey",
    alpha: float = 0.3,
) -> None:
    """Shade contiguous runs of missing/NaN bars."""
    pos = positions if positions is not None else arange(len(missing_mask))
    start = None
    for i, is_missing in enumerate(missing_mask):
        if is_missing and start is None:
            start = i
        elif not is_missing and start is not None:
            ax.axvspan(pos[start] - 0.5, pos[i - 1] + 0.5, color=color, alpha=alpha, zorder=0)
            start = None
    if start is not None:
        ax.axvspan(pos[start] - 0.5, pos[-1] + 0.5, color=color, alpha=alpha, zorder=0)


def percent_barplot(
    df: DataFrame,
    x: str,
    metric: str,
    ax: plt.Axes | None = None,
    **kwargs: bool | str | list[str] | None,
) -> plt.Axes:
    """Plot a bar chart of a percentage/rate metric, grouped by category and colored by site.

    Parameters
    ----------
    df: DataFrame
        Data containing at least `x` and the grouping column, plus either:
        - a plain `metric` column, or
        - two `f"{metric}_mean"` and `f"{metric}_std"` columns used if `show_std` is selected.
    x: str
        Column used for the x-axis categories (e.g. a time unit).
    metric: str
        Name of the value plotted on the y-axis (e.g. "%buzzes", "FBR", "%FBh").
        Resolved to `mean_col` (see below) to find the actual data column.
    ax: matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if not provided,
        matching the `ax`-as-parameter pattern used by `histo`/`timeline`.
    **kwargs: Additional keyword arguments.
        - group_col: str
            Column used to color bars by group (default "Site").
        - colors: dict[str, str]
            Explicit {group: color} mapping. Overrides auto-assigned colors.
        - hatch_metrics: set[str]
            Metric names that should be drawn with a hatch pattern
            (default {"%buzzes", "FBR", "%FBh"}).
        - legend: bool
            Whether to show a legend mapping groups to colors (default True).
        - shade_missing: bool
            Whether to shade contiguous runs of NaN values (default True).
        - dynamic_ylim: bool
            Whether to auto-scale y-limits via `set_dynamic_ylim` (default True).
            Automatically accounts for std whiskers when `show_std` is True.
        - mean_col: str
            Column holding the plotted values. Defaults to `f"{metric}_mean"`
            if that column exists in `df`, otherwise falls back to `metric`
            itself (so both naming conventions work unchanged).
        - show_std: bool
            Whether to overlay std error bars via `ax.errorbar`, drawn as
            dots with whiskers on top of the bars (default False).
        - std_col: str
            Column holding std values. Defaults to `f"{metric}_std"`.
            Ignored if `show_std` is False.

    """
    mean_col = kwargs.get("mean_col") or (
        f"{metric}_mean" if f"{metric}_mean" in df.columns else metric
    )

    if df.empty or df[mean_col].isna().all():
        msg = f"DataFrame for metric '{metric}' has no plottable data."
        logging.warning(msg)
        return ax

    group_col = kwargs.get("group_col", "Site")
    colors_map = kwargs.get("colors") or get_group_colors(df, group_col)
    hatch_metrics = kwargs.get("hatch_metrics", {"%buzzes", "FBR", "%FBh"})
    legend = kwargs.get("legend", True)
    shade_missing = kwargs.get("shade_missing", True)
    dynamic_ylim = kwargs.get("dynamic_ylim", True)

    show_std = kwargs.get("show_std", False)
    std_col = kwargs.get("std_col", f"{metric}_std")

    if ax is None:
        _, ax = plt.subplots()

    bar_colors = df[group_col].map(colors_map)

    ax.bar(df[x].astype(str), df[mean_col], color=bar_colors, zorder=2, edgecolor="black", linewidth=0.5)

    if metric in hatch_metrics:
        for bar in ax.patches:
            bar.set_hatch("/")

    std_values = None
    if show_std:
        if std_col not in df.columns:
            msg = f"show_std=True but column '{std_col}' not found in df."
            logging.warning(msg)
        else:
            std_values = df[std_col]
            ax.errorbar(
                df[x].astype(str), df[mean_col], std_values,
                fmt=".", color="black", elinewidth=2, capthick=10,
                errorevery=1, alpha=0.5, ms=4, capsize=2,
            )

    if shade_missing:
        _shade_missing_bars(ax, df[mean_col].isna().to_numpy())

    if dynamic_ylim and df[mean_col].notna().any():
        if std_values is not None:
            padded = (df[mean_col].fillna(0) + std_values.fillna(0)).to_frame(mean_col)
            set_dynamic_ylim(ax, padded)
        else:
            set_dynamic_ylim(ax, df[[mean_col]].fillna(0))

    if legend and len(colors_map) > 1:
        handles = [Patch(facecolor=c, label=g) for g, c in colors_map.items()]
        ax.legend(
            handles=handles,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            title=group_col,
        )

    ax.set_title(f"{metric} per {x}")
    ax.set_ylabel(metric)
    ax.set_xlabel(x)

    return ax
