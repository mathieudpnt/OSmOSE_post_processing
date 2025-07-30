"""Plot functions used for DataAplose objects."""

from __future__ import annotations

from collections import Counter
from itertools import cycle
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from numpy import ceil, histogram, polyfit, zeros
from pandas import DataFrame, Series, Timedelta, Timestamp, date_range
from scipy.stats import pearsonr
from seaborn import scatterplot

from post_processing import logger
from post_processing.utils.def_func import (
    add_season_period,
    get_coordinates,
    get_labels_and_annotators,
    get_sun_times,
    shade_no_effort,
)
from post_processing.utils.premiers_resultats_utils import timedelta_to_str

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pandas.tseries import offsets

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def histo(
    df: DataFrame,
    ax: plt.Axes,
    bin_size: Timedelta | offsets.BaseOffset,
    time_bin: Timedelta,
    **kwargs: bool | str | list[str] | tuple[float, float] | list[Timestamp] | Series,
) -> None:
    """Seasonality plot.

    Parameters
    ----------
    df: DataFrame
        Data to plot.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object on which to draw the histogram.
    bin_size: Timedelta | offsets.BaseOffset
        The size of the histogram bins.
    time_bin: Timedelta | offsets.BaseOffset
        The size of detections.
    **kwargs: Additional keyword arguments depending on the mode.
        - legend: bool
            Whether to show the legend.
        - color: str | list[str]
            Color or list of colors for the histogram bars.
            If not provided, default colors will be used.
        - season: bool
            Whether to show the season.
        - coordinates: tuple[float, float]
            The coordinates of the plotted detections.
        - effort: Series
            The list of timestamps corresponding to the observation effort.
            If provided, data will be normalized by observation effort.

    """
    legend = kwargs.get("legend", False)
    color = kwargs.get("color", False)
    season = kwargs.get("season", False)
    effort = kwargs.get("effort", False)
    lat, lon = kwargs.get("coordinates")

    labels, annotators = zip(*[col.rsplit("-", 1) for col in df.columns], strict=False)
    labels = list(labels)
    annotators = list(annotators)

    if isinstance(bin_size, Timedelta):
        begin = min(df.index).floor(bin_size)
        end = max(df.index + bin_size).ceil(bin_size)
        freq = bin_size
    else:
        begin = bin_size.rollback(min(df.index)).normalize()
        end = bin_size.rollforward(max(df.index)).normalize()
        bin_str = str(bin_size.n) + bin_size.name
        freq = Timedelta(bin_str.split("-")[0])

    if not color:
        color = get_colors(df)

    if len(df.columns) > 1 and legend:
        legend_labels = get_legend(labels, annotators)
    else:
        legend_labels = None

    if effort:
        effort_df, origin_timebin = effort
        effort_series = Series(
            data=effort_df.iloc[:, 0].values,
            index=[interval.left for interval in effort_df.iloc[:, 0].index]
        )
        for col in df.columns:
            effort_ratio = effort_series * (origin_timebin / time_bin)
            effort_ratio = Series(
                np.where((effort_ratio > 0) & (effort_ratio < 1), 1.0, effort_ratio),
                index=effort_series.index,
                name=effort_series.name
            )
            df[f"{col}"] = (df[col] / effort_ratio.reindex(df[col].index)).clip(upper=1)


    n_groups = len(labels) if legend_labels else 1
    bar_width = freq / n_groups
    bin_starts = mdates.date2num(df.index)

    for i in range(n_groups):
        offset = i * bar_width.total_seconds() / 86400  # convert to days for plot

        bar_kwargs = {
            "width": bar_width.total_seconds() / 86400,
            "align": "edge",
            "edgecolor": "black",
            "color": color[i],
            "zorder": 2,
        }
        if legend_labels:
            bar_kwargs["label"] = legend_labels[i]

        ax.bar(bin_starts + offset, df.iloc[:, i], **bar_kwargs)

    if len(df.columns) > 1 and legend:
        ax.legend(labels=legend_labels, bbox_to_anchor=(1.01, 1), loc="upper left")

    bin_size_str = timedelta_to_str(bin_size) if isinstance(bin_size, Timedelta) else bin_str
    title = (
        f"Detections{(' normalized by effort' if effort else '')}"
        f"\n(detections: {timedelta_to_str(time_bin)}"
        f" - bin size: {bin_size_str})"
    )
    ax.set_ylabel(title)

    if effort:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_yticks(np.arange(0, 1.02, 0.2))
    else:
        max_val = df.to_numpy().max()
        ax.set_ylim(0, int(ceil(1.05 * max_val)))
        ax.set_yticks(
            range(
                0,
                int(ceil(max_val)) + 1,
                max(1, int(ceil(max_val)) // 4),
            ),
        )

    ax.title.set_text(
        f"annotator: {', '.join(set(annotators))}\nlabel: {', '.join(set(labels))}",
    )
    ax.set_xlim(begin, end)

    if season:
        if not lat or not lon:
            get_coordinates()
        add_season_period(ax, northern=lat >= 0)

    if effort:
        if not isinstance(bin_size, Timedelta):
            bar_width = Timedelta(bin_str.split("-")[0])
        else:
            bar_width = bin_size
        shade_no_effort(
            ax=ax,
            bin_starts=df.index,
            observed=effort_series,
            bar_width=bar_width,
        )

def map_detection_timeline(
    df: DataFrame,
    ax: Axes,
    *,
    coordinates: tuple[float, float] | None = None,
    mode: str = "scatter",
    **kwargs: bool,
) -> None:
    """Plot daily detection patterns for a given annotator and label.

    Parameters
    ----------
    df: DataFrame
        data to plot
    ax : plt.Axes
        The matplotlib axis to draw on.
    coordinates : tuple[float, float]
        The latitude and longitude.
    mode : {'scatter', 'heatmap'}
        'scatter': Plot each detection as a point by time of day.
        'heatmap': Plot hourly detection rate as a heatmap.
    **kwargs: Additional keyword arguments depending on the mode.
        -show_rise_set : bool, default True
            Whether to overlay sunrise and sunset lines.
        -season: bool
            Whether to show the season.

    """
    lat, lon = coordinates
    if lat is None or lon is None:
        lat, lon = get_coordinates()

    show_rise_set = kwargs.get("show_rise_set", False)
    season = kwargs.get("season", False)

    datetime_list = df["start_datetime"]
    labels, annotators = get_labels_and_annotators(df)
    time_bin = df["end_time"].iloc[0]
    begin = min(datetime_list) - Timedelta("1d")
    end = max(datetime_list) + Timedelta(time_bin, "s") + Timedelta("1d")

    # compute sunrise and sunset decimal hour at dataset location
    sunrise, sunset, _, _, _, _ = get_sun_times(
        start=begin,
        stop=end,
        lat=lat,
        lon=lon,
    )

    dates = date_range(begin.normalize(), end.normalize(), freq="D")

    if mode == "scatter":
        for ann in set(annotators):
            for lbl in set(labels):
                group = df[(df["annotator"] == ann) & (df["annotation"] == lbl)]

                if group.empty:
                    continue

                detect_time_dec = [
                    ts.hour + ts.minute / 60 + ts.second / 3600
                    for ts in group["start_datetime"]
                ]

                ax.scatter(
                    group["start_datetime"],
                    detect_time_dec,
                    label=f"{ann} - {lbl}",
                    marker="x",
                    linewidths=1,
                    alpha=0.7,
                )

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            frameon=True,
            framealpha=0.6,
        )

    elif mode == "heatmap":
        mat = zeros((24, len(dates)))
        date_to_index = {date: i for i, date in enumerate(dates)}

        for dt in datetime_list:
            date = dt.normalize()
            hour = dt.hour
            if date in date_to_index:
                j = date_to_index[date]
                mat[hour, j] += 1  # count detections per hour per day

        im = ax.imshow(
            mat,
            extent=(begin, end, 0, 24),
            vmin=0,
            vmax=3600 / time_bin,
            aspect="auto",
            origin="lower",
        )
        fig = ax.get_figure()
        cbar = fig.colorbar(im, ax=ax, pad=0.1)
        cbar.ax.set_ylabel(f"Detections per {time_bin!s}s bin")

    else:
        msg = f"Unsupported mode '{mode}'. Use 'scatter' or 'heatmap'."
        raise ValueError(msg)

    if show_rise_set:
        ax.plot(dates, sunrise, color="darkorange", label="Sunrise")
        ax.plot(dates, sunset, color="royalblue", label="Sunset")
        ax.legend(
            loc="center left",
            frameon=1,
            framealpha=0.6,
            bbox_to_anchor=(1.01, 0.95),
        )

    ax.set_xlim(begin, end)
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 2))
    ax.set_ylabel("Hour")
    ax.set_xlabel("Date")
    ax.grid(color="k", linestyle="-", linewidth=0.2)

    ax.set_title(
        f"Time of detections ({mode})\n"
        f"annotator: {', '.join(set(annotators))} - "
        f"label: {', '.join(set(labels))}\n "
        f"timezone: {begin.tz}",
    )

    if season:
        northern = lat >= 0
        add_season_period(ax, northern=northern)


def overview(df: DataFrame) -> None:
    """Overview of an APLOSE formatted DataFrame.

    Parameters
    ----------
    df: DataFrame
        The Dataframe to analyse.

    """
    summary_label = (
        df.groupby("annotation")["annotator"]  # noqa: PD010
        .apply(Counter)
        .unstack(fill_value=0)
    )

    summary_annotator = (
        df.groupby("annotator")["annotation"]  # noqa: PD010
        .apply(Counter)
        .unstack(fill_value=0)
    )

    dataset = df["dataset"].iloc[0]

    fig, axs = plt.subplots(2, 1)
    axs[0] = summary_label.plot(
        kind="bar",
        ax=axs[0],
        edgecolor="black",
        linewidth=0.5,
    )
    axs[1] = summary_annotator.plot(
        kind="bar",
        ax=axs[1],
        edgecolor="black",
        linewidth=0.5,
    )

    for a in axs:
        a.legend(
            loc="center left",
            frameon=1,
            framealpha=0.6,
            bbox_to_anchor=(1.01, 0.5),
        )
        a.tick_params(axis="both", rotation=0)
        a.set_ylabel("Number of annotated calls")
        a.yaxis.grid(color="gray", linestyle="--")
        a.set_axisbelow(True)

    # labels
    _wrap_xtick_labels(axs[0], max_chars=10)
    axs[1].set_xlabel("Annotator")

    # titles
    axs[0].set_title("Number of annotations per label")
    axs[1].set_title("Number of annotations per annotator")
    fig.suptitle(f"{dataset}")

    # log
    msg = f"- Overview of the detections -\n {summary_label}"
    logger.info(msg)


def _wrap_xtick_labels(ax: plt.Axes, max_chars: int = 10) -> None:
    """Wrap x-axis tick labels at max_chars per line."""

    def wrap_text(text: str) -> str:
        lines = []
        while len(text) > max_chars:
            break_index = text.rfind(" ", 0, max_chars + 1)
            if break_index == -1:
                break_index = max_chars  # force break
            lines.append(text[:break_index])
            text = text[break_index:].lstrip()
        lines.append(text)  # remaining part
        return "\n".join(lines)

    new_labels = [wrap_text(label.get_text()) for label in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels, rotation=0)


def agreement(
    df: DataFrame,
    bin_size: Timedelta | offsets.BaseOffset,
    ax: plt.Axes,
) -> None:
    """Compute and visualize agreement between two annotators.

    This function compares annotation timestamps from two annotators over a time range.
    It also fits and plots a linear regression line and displays the coefficient
    of determination (R²) on the plot.

    Parameters
    ----------
    df : DataFrame
        APLOSE-formatted DataFrame.
        It must contain The annotations of two annotators.

    bin_size : Timedelta | offsets.BaseOffset
        The size of each time bin for aggregating annotation timestamps.

    ax : matplotlib.axes.Axes
        Matplotlib axes object where the scatterplot and regression line will be drawn.

    """
    labels, annotators = get_labels_and_annotators(df)

    datetimes1 = list(
        df[(df["annotator"] == annotators[0]) & (df["annotation"] == labels[0])][
            "start_datetime"
        ],
    )
    datetimes2 = list(
        df[(df["annotator"] == annotators[1]) & (df["annotation"] == labels[1])][
            "start_datetime"
        ],
    )

    # scatter plot
    n_annot_max = bin_size.total_seconds() / df["end_time"].iloc[0]

    start = df["start_datetime"].min()
    stop = df["start_datetime"].max()

    freq = (
        bin_size if isinstance(bin_size, Timedelta) else str(bin_size.n) + bin_size.name
    )

    bins = date_range(
        start=start.floor(bin_size),
        end=stop.ceil(bin_size),
        freq=freq,
    )

    hist1, _ = histogram(datetimes1, bins=bins)
    hist2, _ = histogram(datetimes2, bins=bins)

    df_hist = (
        DataFrame(
            {
                annotators[0]: hist1,
                annotators[1]: hist2,
            },
        )
        / n_annot_max
    )

    scatterplot(data=df_hist, x=annotators[0], y=annotators[1], ax=ax)

    coefficients = polyfit(df_hist[annotators[0]], df_hist[annotators[1]], 1)
    poly = np.poly1d(coefficients)
    ax.plot(df_hist[annotators[0]], poly(df_hist[annotators[0]]), lw=1)

    ax.set_xlabel(f"{annotators[0]}\n{labels[0]}")
    ax.set_ylabel(f"{annotators[1]}\n{labels[1]}")
    ax.grid(linestyle="-", linewidth=0.2)

    # Pearson correlation (R²)
    r, _ = pearsonr(df_hist[annotators[0]], df_hist[annotators[1]])
    ax.text(0.05, 0.85, f"R² = {r**2:.2f}", transform=ax.transAxes)


def timeline(
    df: DataFrame,
    ax: plt.Axes,
    **kwargs: list[str],
) -> None:
    """Plot detections on a timeline.

    Parameters
    ----------
    df: DataFrame
        APLOSE DataFrame
    ax : matplotlib.axes.Axes
        Matplotlib axes object where the scatterplot and regression line will be drawn.
    **kwargs: Additional keyword arguments depending on the mode.
        - color: str | list[str]
            Color or list of colors for the histogram bars.
            If not provided, default colors will be used.

    """
    color = kwargs.get("color")

    labels, _ = get_labels_and_annotators(df)

    color = (
        color
        if color
        else [c for _, c in zip(range(len(labels)), cycle(default_colors))]
    )

    for i, label in enumerate(labels):
        time_det = df[(df["annotation"] == label)]["start_datetime"].to_list()
        l_data = len(time_det)
        x = np.ones((l_data, 1), int) * i
        ax.scatter(time_det, x, color=color[i])

    ax.grid(color="k", linestyle="-", linewidth=0.2)
    ax.set_yticks(np.arange(0, len(labels), 1))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("Date")
    ax.set_xlim(
        df["start_datetime"].min().floor("1d"),
        df["end_datetime"].max().ceil("1d"),
    )

def get_colors(df: DataFrame) -> list[str]:
    """Return default plot colors."""
    return [c for _, c in zip(range(len(df.columns)), cycle(default_colors))]

def get_legend(annotators: str | list[str], labels: str | list[str]) -> list[str]:
    """Return plot legend."""
    if len(set(labels)) > 1 and len(set(annotators)) == 1:
        return labels
    if len(set(annotators)) > 1 and len(set(labels)) == 1:
        return annotators
    return [f"{ant}\n{lbl}" for ant, lbl in zip(annotators, labels, strict=False)]

bin_size = Timedelta()