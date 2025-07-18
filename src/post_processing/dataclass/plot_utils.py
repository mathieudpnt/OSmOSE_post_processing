"""Plot functions used for DataAplose objects."""

from __future__ import annotations

import logging
from collections import Counter
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes  # noqa: TC002
from numpy import ceil, histogram, integer, ndarray, polyfit, zeros
from pandas import DataFrame, Timedelta, date_range
from pandas.tseries import offsets  # noqa: TC002
from scipy.stats import pearsonr
from seaborn import scatterplot

from post_processing.def_func import get_coordinates, get_sun_times, t_rounder
from post_processing.premiers_resultats_utils import get_resolution_str

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
from post_processing import logger


def _get_legend(annotator: str | list[str], label: str | list[str]) -> list[str]:
    """Return plot legend."""
    if len(annotator) > 1 and len(label) > 1:
        legend = [f"{ant} - {lbl}" for ant, lbl in zip(annotator, label, strict=False)]
    elif len(annotator) == 1:
        legend = label
    elif len(label) == 1:
        legend = annotator
    else:
        msg = "Legend error"
        raise ValueError(msg)
    return legend


def histo(
    df: DataFrame,
    ax: plt.Axes,
    res_bin: int | offsets.BaseOffset,
    color: str | None = None,
    *,
    legend: bool = True,
) -> None:
    """Seasonality plot.

    Parameters
    ----------
    df: DataFrame
        Data to plot.
    res_bin: int | offsets.BaseOffset
        The size of the histogram bins.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object on which to draw the histogram.
    color : str or list of str, optional
        Color or list of colors for the histogram bars.
        If not provided, default colors will be used.
    legend : bool, default=True
        Whether to display the legend on the plot.

    """
    datetime_list = df["start_datetime"]
    annotator = list(set(df["annotator"]))
    label = list(set(df["annotation"]))
    time_bin = df["end_time"].iloc[0].astype(int)
    begin = min(datetime_list) - Timedelta("1d")
    end = max(datetime_list) + Timedelta(time_bin, "s") + Timedelta("1d")
    grouped_by_annotator = [
        (annotator, group["start_datetime"])
        for annotator, group in df.groupby("annotator")
    ]
    annotators_list, series_list = zip(*grouped_by_annotator, strict=False)

    if isinstance(res_bin, (int, integer)):
        bins = date_range(
            start=t_rounder(t=begin, res=res_bin),
            end=t_rounder(t=end, res=res_bin),
            freq=str(res_bin) + "s",
        )
    else:
        bins = date_range(
            start=begin.normalize(),
            end=end.normalize(),
            freq=str(res_bin.n) + res_bin.name,
        )

    color = (
        color
        if color
        else [c for _, c in zip(range(len(annotator)), cycle(default_colors))]
    )

    val1, _, _ = ax.hist(
        series_list,
        bins=bins,
        edgecolor="black",
        zorder=2,
        histtype="bar",
        stacked=False,
        color=color,
    )

    if val1.ndim > 1 and legend:
        ax.legend(labels=annotators_list, loc="upper right")

    time_bin_str = get_resolution_str(time_bin)
    resolution_bin_str = (
        str(res_bin.n) + res_bin.name
        if not isinstance(res_bin, int)
        else get_resolution_str(res_bin)
    )

    ax.set_ylabel(
        f"Detections\n(resolution: {time_bin_str} - bin size: {resolution_bin_str})",
    )

    ax.set_ylim(0, int(ceil(1.05 * ndarray.max(val1))))
    ax.set_yticks(
        range(
            0,
            int(ceil(ndarray.max(val1))) + 1,
            max(1, int(ceil(ndarray.max(val1))) // 4),
        ),
    )
    ax.title.set_text(
        f"annotator: {', '.join(set(annotator))}\nlabel: {', '.join(set(label))}",
    )


def map_detection_timeline(
    df: DataFrame,
    ax: Axes = None,
    coordinates: tuple[float, float] | None = None,
    *,
    mode: str = "scatter",
    show_rise_set: bool = True,
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
    show_rise_set : bool, default True
        Whether to overlay sunrise and sunset lines.

    """
    if ax is None:
        fig, ax = plt.subplots()

    lat, lon = coordinates
    if lat is None or lon is None:
        lat, lon = get_coordinates()

    datetime_list = df["start_datetime"]
    annotator = list(set(df["annotator"]))
    label = list(set(df["annotation"]))
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
        detect_time_dec = [
            ts.hour + ts.minute / 60 + ts.second / 3600 for ts in datetime_list
        ]
        ax.scatter(
            datetime_list,
            detect_time_dec,
            marker="x",
            linewidths=1,
            color="silver",
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
        f"annotator: {', '.join(set(annotator))} - "
        f"label: {', '.join(set(label))}\n "
        f"timezone: {begin.tz}",
    )


def overview(df: DataFrame) -> None:
    """Overview of an APLOSE formatted DataFrame.

    Parameters
    ----------
    df: DataFrame
        The Dataframe to analyse.

    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

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

    msg = f"- Overview of the detections -\n\n {summary_label}"
    logger.info(msg)

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
    bin_size: Timedelta,
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

    bin_size : Timedelta
        The size of each time bin for aggregating annotation timestamps.

    ax : matplotlib.axes.Axes
        Matplotlib axes object where the scatterplot and regression line will be drawn.

    """
    annotators = df["annotator"].unique().tolist()
    labels = df["annotation"].unique().tolist()
    if len(labels) == 1:
        labels = [labels[0]] * 2

    num_annotators, num_labels = 2, 2
    if len(annotators) != num_annotators:
        msg = f"Two annotators needed, DataFrame contains {len(annotators)} annotators"
        raise ValueError(msg)
    if len(labels) != num_labels:
        msg = f"Two labels needed, DataFrame contains {len(labels)} labels"
        raise ValueError(msg)

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
    time_bins = date_range(
        start=t_rounder(t=start, res=bin_size),
        end=t_rounder(t=stop, res=bin_size),
        freq=bin_size,
    )

    hist1, _ = histogram(datetimes1, bins=time_bins)
    hist2, _ = histogram(datetimes2, bins=time_bins)

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
