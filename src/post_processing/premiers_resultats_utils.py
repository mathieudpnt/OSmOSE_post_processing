import datetime
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import easygui
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from scipy.stats import pearsonr

from src.post_processing.def_func import (
    get_datetime_format,
    get_duration,
    load_detections,
    read_yaml,
    suntime_hour,
    t_rounder,
)


def load_parameters_from_yaml(
    file: Path,
) -> (
    pd.DataFrame,
    list[int],
    list[str],
    list[str],
    list[int],
    pd.Timestamp,
    pd.Timestamp,
    list[datetime.timezone],
):
    """Load parameters from yaml.

    Parameters
    ----------
    file: Path
        To yaml file

    Returns
    -------
    df : APLOSE formatted detections
    time_bin:  list, time bin(s) in seconds
    annotators: list of annotators
    labels: list of labels
    f_max: list of maximum frequencies
    datetime_begin: pd.Timestamp
    datetime_end: pd.Timestamp
    tz: timezone

    """
    parameters = read_yaml(file=file)

    df_detections = pd.DataFrame()
    for f in parameters:
        df_detections = pd.concat(
            [df_detections, load_detections(**parameters[f])], ignore_index=True
        )

    # converted to UTC in case of several timezones
    if len(list({dt.tz for dt in df_detections["start_datetime"]})) > 1:
        df_detections["start_datetime"] = [
            dt.tz_convert("UTC") for dt in df_detections["start_datetime"]
        ]
        df_detections["end_datetime"] = [
            dt.tz_convert("UTC") for dt in df_detections["end_datetime"]
        ]

    return (df_detections.sort_values(by="start_datetime").reset_index(drop=True),)


def select_reference(param: Iterable, param_str: str = "") -> str | int | float:
    """Select a reference parameter (label, annotator...) from a list.

    Parameters
    ----------
    param : Iterable
        List of the choices
    param_str: str
        Message to display

    Returns
    -------
    the user-defined reference parameter

    """
    if len(list(set(param))) > 1:
        selection = easygui.buttonbox(
            msg=f"{param_str} selection",
            title="Parameter selection",
            choices=[str(elem) for elem in list(set(param))],
        )
    else:
        selection = next(iter(set(param)))

    if not selection:
        msg = "select_reference() has been cancelled"
        raise TypeError(msg)

    return selection


def set_plot_resolution(
    time_bin: int,
    start: pd.Timestamp,
    stop: pd.Timestamp,
) -> (pd.DatetimeIndex, str, int, mpl.dates.DayLocator):
    """Compute the time_vector for a user-defined resolution.

    Select appropriate date ticks and datetime format for plots

    Parameters
    ----------
    time_bin : int
        Length in seconds of a time bin,
        it is similar to the precision of the future plots

    start : pd.Timestamp
        begin datetime

    stop : pd.Timestamp
        end datetime

    Returns
    -------
    datetime_index : pd.DatetimeIndex,
        computed timestamps according to user-defined resolution
    y_axis_plot_legend : str
        y-axis legend
    number_max_of_annotation : int
        maximum number of annotations possible according to resolution and time bin
    date_interval : mpl.dates.DayLocator
        x-ticks resolution

    """
    resolution_bin = get_duration(msg="Enter the x-axis bin resolution")
    resolution_x_ticks = get_duration(
        msg="Enter the x-axis tick resolution",
        default="2h",
    )

    date_interval = resolution_x_ticks

    number_max_of_annotation = int(resolution_bin / time_bin)
    datetime_index = pd.date_range(
        start=t_rounder(t=start, res=resolution_bin),
        end=t_rounder(t=stop, res=resolution_bin),
        freq=str(resolution_bin) + "s",
    )

    resolution_bin_str = _get_resolution_str(resolution_bin)
    time_bin_str = _get_resolution_str(time_bin)
    y_axis_plot_legend = (
        f"Detections\n(resolution: {time_bin_str} - bin size: {resolution_bin_str})"
    )

    return (
        datetime_index,
        y_axis_plot_legend,
        number_max_of_annotation,
        date_interval,
    )


def _set_yaxis(ax: Axes, max_annotation_number: int) -> str:
    """Change ax properties.

    Whether the plot is visualized in percentage or in raw values.

    Parameters
    ----------
    ax: Axes
        Plot axes

    max_annotation_number: int
        Maximum number of annotations possible

    """
    choice_percentage = easygui.buttonbox(
        msg="Do you want your scripts plot in percentage or in raw values ?",
        choices=["percentage", "raw values"],
        default_choice="percentage",
    )

    if isinstance(ax, Iterable):
        y_max = int(
            max(
                [
                    max([patch.get_height() for patch in ax[i].patches])
                    for i in range(len(ax))
                ],
            ),
        )
    else:
        y_max = int(max([patch.get_height() for patch in ax.patches]))

    resolution = easygui.integerbox(
        msg=f"Select a y-ticks resolution\n"
        f"(mode='{choice_percentage}' - highest_bin/max={y_max}/{max_annotation_number})",
        title="Plot resolution",
        default=10,
        lowerbound=1,
        upperbound=int(1e6),
    )

    # change the y scale
    if choice_percentage == "percentage":
        bars = np.arange(0, 101, resolution)
        y_pos = [max_annotation_number * pos / 100 for pos in bars]

        if isinstance(ax, Iterable):
            [a.set_yticks(y_pos, bars) for a in ax]
            [a.set_ylabel("") for a in ax]
        else:
            ax.set_yticks(y_pos, bars)
            current_label = ax.get_ylabel().split("\n")
            current_label[0] = current_label[0] + " (%)"
            new_label = "\n".join(current_label)
            ax.set_ylabel(new_label)
    elif isinstance(ax, Iterable):
        [
            a.set_yticks(
                np.arange(0, max_annotation_number + resolution, resolution),
            )
            for a in ax
        ]
        [a.set_ylabel("") for a in ax]
    else:
        ax.set_yticks(np.arange(0, max_annotation_number + resolution, resolution))
        current_label = ax.get_ylabel().split("\n")
        current_label[0] = current_label[0] + " (N)"
        new_label = "\n".join(current_label)
        ax.set_ylabel(new_label)

    # set y-axis limit
    if isinstance(ax, Iterable):
        [a.set_ylim([0, min(int(1.4 * y_max), max_annotation_number)]) for a in ax]
    else:
        ax.set_ylim([0, min(int(1.4 * y_max), max_annotation_number)])

    return choice_percentage


def overview_plot(df: pd.DataFrame) -> None:
    """Overview of an APLOSE formatted dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        APLOSE formatted result DataFrame with detections and associated timestamps

    """
    summary_label = (
        df.groupby("annotation")["annotator"].apply(Counter).pivot_table(fill_value=0)
    )
    summary_annotator = (
        df.groupby("annotator")["annotation"].apply(Counter).pivot_table(fill_value=0)
    )

    print(f"\n- Overview of the detections -\n\n {summary_label}")

    fig, ax = plt.subplots(2, 1)
    ax[0] = summary_label.plot(kind="bar", ax=ax[0], edgecolor="black", linewidth=1)
    ax[1] = summary_annotator.plot(kind="bar", ax=ax[1], edgecolor="black", linewidth=1)

    for a in ax:
        a.legend(loc="best", frameon=1, framealpha=0.6)
        a.tick_params(axis="both", rotation=0)
        a.set_ylabel("Number of annotated calls")
        a.yaxis.grid(color="gray", linestyle="--")
        a.set_axisbelow(True)

    # labels
    ax[0].set_xlabel("Labels")
    ax[1].set_xlabel("Annotator")

    # titles
    ax[0].set_title("Number of annotations per label")
    ax[1].set_title("Number of annotations per annotator")

    plt.tight_layout()


def plot_hourly_detection_rate(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    *,
    show_rise_set: bool = True,
) -> None:
    """Compute the hourly detection rate for an APLOSE DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        APLOSE formatted result DataFrame with detections and associated timestamps

    lat: float
        latitude

    lon: float
        longitude

    show_rise_set : bool, default True
        display the sunrise and sunset lines

    """
    datetime_begin = df["start_datetime"].iloc[0]
    datetime_end = df["end_datetime"].iloc[-1]

    bin_ref = select_reference(df["end_time"], "time bin")

    # compute sunrise and sunset decimal hour at the dataset location
    sunrise, sunset, _, _, _, _ = suntime_hour(
        start=datetime_begin,
        stop=datetime_end,
        lat=lat,
        lon=lon,
    )

    df["date"] = [ts.normalize() for ts in df["start_datetime"]]
    df["hour"] = [ts.hour for ts in df["start_datetime"]]

    det_groupby = df.groupby(["date", "hour"]).size()
    idx_day_group_by = det_groupby.index.get_level_values(0)
    idx_hour_group_by = det_groupby.index.get_level_values(1)

    dates = pd.date_range(
        datetime_begin.normalize(),
        datetime_end.normalize(),
        freq="D",
    )
    m = np.zeros((24, len(dates)))
    for idx_j, j in enumerate(dates):
        # Search for detection in day = j
        f = [idx for idx, det in enumerate(idx_day_group_by) if det == j]
        if f:
            for ff in f:
                hour = idx_hour_group_by[ff]
                m[int(hour), idx_j] = det_groupby.iloc[ff]

    # plot
    fig, ax = plt.subplots()
    im = ax.imshow(
        m,
        extent=[datetime_begin, datetime_end, 0, 24],
        vmin=0,
        vmax=3600 / bin_ref,
        aspect="auto",
        origin="lower",
    )
    # colorbar
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel(f"{bin_ref!s}s bins positive to detection")

    if show_rise_set:
        plt.plot(dates, sunrise, color="darkorange", linewidth=1)
        plt.plot(dates, sunset, color="royalblue", linewidth=1)

    # axes settings
    ax.xaxis_date()
    resolution_x_ticks = get_duration(
        msg="Enter the x-axis tick resolution",
        default="2h",
    )

    if type(resolution_x_ticks) is int:
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=resolution_x_ticks))
    elif resolution_x_ticks.base == "MS":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=resolution_x_ticks.n))
    else:
        msg = "date locator not supported"
        raise ValueError(msg)

    date_formatter = mdates.DateFormatter(
        fmt=get_datetime_format(),
        tz=datetime_begin.tz,
    )
    ax.xaxis.set_major_formatter(date_formatter)
    ax.set_ylabel("Hour")
    ax.set_xlabel("Date")
    plt.xlim(datetime_begin, datetime_end)
    plt.tight_layout()


def scatter_detections(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    *,
    show_rise_set: bool = True,
) -> None:
    """Scatter plot of the detections from an APLOSE formatted DataFrame.

    Additionally, sunrise and sunset lines can be plotted
    if show_rise_set is set to True (default value)

    Parameters
    ----------
    df : pd.DataFrame
        APLOSE formatted result DataFrame with detections and associated timestamps

    lat : float
        latitude

    lon : float
        longitude

    show_rise_set : bool, default True
        display the sunrise and sunset lines

    """
    datetime_begin = df["start_datetime"].iloc[0]
    datetime_end = df["end_datetime"].iloc[-1]

    # compute sunrise and sunset decimal hour at the dataset location
    hour_sunrise, hour_sunset, _, _, _, _ = suntime_hour(
        start=datetime_begin,
        stop=datetime_end,
        lat=lat,
        lon=lon,
    )

    dates = pd.date_range(datetime_begin.date(), datetime_end.date(), freq="D")

    # decimal hours of detections
    hour_det = [
        ts.hour + ts.minute / 60 + ts.second / 3600 for ts in df["start_datetime"]
    ]

    # plot
    fig, ax = plt.subplots()
    plt.scatter(
        df["start_datetime"],
        hour_det,
        marker="x",
        linewidths=1,
        color="silver",
    )
    if show_rise_set:
        plt.plot(dates, hour_sunrise, color="darkorange")
        plt.plot(dates, hour_sunset, color="royalblue")

    # axes settings
    plt.xlim(datetime_begin, datetime_end)
    plt.ylim(0, 24)

    resolution_x_ticks = get_duration(
        msg="Enter the x-axis tick resolution",
        default="2h",
    )

    if type(resolution_x_ticks) is int:
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=resolution_x_ticks))
    elif resolution_x_ticks.base == "MS":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=resolution_x_ticks.n))
    else:
        msg = f"date locator not supported"
        raise ValueError(msg)

    date_formatter = mdates.DateFormatter(
        fmt=get_datetime_format(),
        tz=datetime_begin.tz,
    )
    ax.xaxis.set_major_formatter(date_formatter)
    ax.grid(color="k", linestyle="-", linewidth=0.2)
    ax.set_ylabel("Hour")
    ax.set_xlabel("Date")

    # title
    plt.title(
        f"Time of detections within each day for dataset {select_reference(df['dataset'])}",
    )

    plt.tight_layout()


def single_plot(df: pd.DataFrame) -> None:
    """Plot the detections of an APLOSE DataFrame for a single label.

    Parameters
    ----------
    df: pd.DataFrame
        APLOSE DataFrame

    """
    # selection of the references parameters
    datetime_begin = df["start_datetime"].iloc[0]
    datetime_end = df["end_datetime"].iloc[-1]
    annotators = list(set(df["annotator"]))
    annot_ref = select_reference(annotators, "annotator")
    label_ref = select_reference(
        df[df["annotator"] == annot_ref]["annotation"],
        "label",
    )

    if df[(df["start_time"] == 0) & (df["end_time"] == max(df["end_time"]))].empty:
        msg = "DataFrame contains no weak detection, consider reshaping it first"
        raise ValueError(msg)

    time_bin_ref = int(
        select_reference(
            df[(df["annotator"] == annot_ref) & (df["is_box"] == 0)]["end_time"],
            "time bin",
        ),
    )

    # set plot resolution
    time_vector, y_label_legend, n_annot_max, date_locator = set_plot_resolution(
        time_bin=time_bin_ref,
        start=datetime_begin,
        stop=datetime_end,
    )

    df_1annot_1label = df[
        (df["annotator"] == annot_ref) & (df["annotation"] == label_ref)
    ]

    # plot
    fig, ax = plt.subplots()
    [hist_y, hist_x, _] = ax.hist(
        df_1annot_1label["start_datetime"],
        bins=time_vector,
        edgecolor="black",
        zorder=2,
    )

    # title
    plt.title(f"annotator: {annot_ref}\nlabel: {label_ref}")

    # axes settings
    if type(date_locator) is int:
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=date_locator))
    elif date_locator.name in ["MS", "ME", "BME", "BMS"]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=date_locator.n))
    else:
        msg = f"date locator not supported"
        raise ValueError(msg)

    date_formatter = mdates.DateFormatter(
        fmt=get_datetime_format(),
        tz=datetime_begin.tz,
    )
    ax.xaxis.set_major_formatter(date_formatter)
    plt.xlim(time_vector[0], time_vector[-1])
    ax.grid(linestyle="--", linewidth=0.2, axis="both", zorder=1)
    ax.set_ylabel(y_label_legend)
    _set_yaxis(ax=ax, max_annotation_number=n_annot_max)

    plt.tight_layout()


def multilabel_plot(df: pd.DataFrame) -> None:
    """Plot the detections of an APLOSE formatted DataFrame for all labels.

    Parameters
    ----------
    df: pd.DataFrame
        APLOSE DataFrame

    """
    datetime_begin = df["start_datetime"].iloc[0]
    datetime_end = df["end_datetime"].iloc[-1]
    annotators = list(set(df["annotator"]))
    annot_ref = select_reference(annotators, "annotator")
    time_bin_ref = select_reference(
        df[df["annotator"] == annot_ref]["end_time"],
        "time bin",
    )
    list_labels = list(set(df[df["annotator"] == annot_ref]["annotation"]))

    if not len(list_labels) > 1:
        msg = f"Only {len(list_labels)!s} labels detected, multilabel plot cancelled"
        raise ValueError(msg)

    time_vector, y_label_legend, n_annot_max, date_locator = set_plot_resolution(
        time_bin=time_bin_ref,
        start=datetime_begin,
        stop=datetime_end,
    )

    fig, ax = plt.subplots(
        nrows=len(list_labels),
    )

    date_formatter = mdates.DateFormatter(
        fmt=get_datetime_format(),
        tz=datetime_begin.tz,
    )
    for i, label in enumerate(list_labels):
        df_1annot_1label = df[
            (df["annotator"] == annot_ref) & (df["annotation"] == label)
        ]

        ax[i].hist(
            df_1annot_1label["start_datetime"],
            bins=time_vector,
            linewidth=1,
        )

        ax[i].set_title(label)
        if type(date_locator) is int:
            ax[i].xaxis.set_major_locator(mdates.SecondLocator(interval=date_locator))
        elif date_locator.name in ["MS", "ME", "BME", "BMS"]:
            ax[i].xaxis.set_major_locator(mdates.MonthLocator(interval=date_locator.n))
        else:
            msg = f"date locator not supported"
            raise ValueError(msg)

        ax[i].xaxis.set_major_formatter(date_formatter)
        ax[i].set_xlim(time_vector[0], time_vector[-1])
        ax[i].grid(linestyle="--", linewidth=0.2, axis="both")

    choice = _set_yaxis(ax=ax, max_annotation_number=n_annot_max)
    current_label = y_label_legend.split("\n")
    current_label[0] = (
        current_label[0] + " (%)"
        if choice == "percentage"
        else current_label[0] + "(N)"
    )
    new_label = "\n".join(current_label)

    fig.suptitle(f"Annotator : {annot_ref}")
    fig.supylabel(new_label, ha="center")
    plt.tight_layout()


def multiuser_plot(df: pd.DataFrame):
    df = df.sort_values(by="start_datetime")
    datetime_begin = df["start_datetime"].iloc[0]
    datetime_end = df["end_datetime"].iloc[-1]
    annotators = list(set(df["annotator"]))

    if len(annotators) < 2:
        msg = "Only 1 annotator detected, multiuser plot cancelled"
        raise ValueError(msg)

    annot_ref1 = select_reference(annotators, "annotator 1")
    annot_ref2 = select_reference(
        [elem for elem in annotators if elem != annot_ref1],
        "annotator 2",
    )

    label_ref1 = select_reference(
        df[df["annotator"] == annot_ref1]["annotation"],
        "label 1",
    )
    if label_ref1 not in list(set(df[df["annotator"] == annot_ref2]["annotation"])):
        label_ref2 = select_reference(
            df[df["annotator"] == annot_ref2]["annotation"],
            "label 2",
        )
    else:
        label_ref2 = label_ref1

    time_bin_ref1 = select_reference(
        df[df["annotator"] == annot_ref1]["end_time"],
        "time bin 1",
    )
    time_bin_ref2 = select_reference(
        df[df["annotator"] == annot_ref2]["end_time"],
        "time bin 2",
    )

    if time_bin_ref1 != time_bin_ref2:
        msg = (
            f"The timebin of the detections {annot_ref1}/{label_ref1} "
            f"is {time_bin_ref1}s whereas the timebin for {annot_ref2}/{label_ref2} "
            f"is {time_bin_ref2}s"
        )
        raise ValueError(msg)
    time_bin_ref = time_bin_ref1

    # set plot resolution
    time_vector, y_label_legend, n_annot_max, date_locator = set_plot_resolution(
        time_bin=time_bin_ref,
        start=datetime_begin,
        stop=datetime_end,
    )

    df1_1annot_1label = df[
        (df["annotator"] == annot_ref1) & (df["annotation"] == label_ref1)
    ]
    df2_1annot_1label = df[
        (df["annotator"] == annot_ref2) & (df["annotation"] == label_ref2)
    ]

    fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [8, 2]})

    hist_plot = ax[0].hist(
        [df1_1annot_1label["start_datetime"], df2_1annot_1label["start_datetime"]],
        bins=time_vector,
        label=[annot_ref1, annot_ref2],
        lw=10,
    )
    ax[0].legend(loc="best")

    fig.suptitle(f"[{annot_ref1}/{label_ref1}] VS [{annot_ref2}/{label_ref2}]")

    # axes settings
    if type(date_locator) is int:
        ax[0].xaxis.set_major_locator(mdates.SecondLocator(interval=date_locator))
    elif date_locator.name in ["MS", "ME", "BME", "BMS"]:
        ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=date_locator.n))
    else:
        msg = f"date locator not supported"
        raise ValueError(msg)

    date_formatter = mdates.DateFormatter(
        fmt=get_datetime_format(),
        tz=datetime_begin.tz,
    )
    ax[0].xaxis.set_major_formatter(date_formatter)
    ax[0].set_xlim(time_vector[0], time_vector[-1])
    ax[0].grid(linestyle="--", linewidth=0.2, axis="both")
    ax[0].set_ylabel(y_label_legend)
    _set_yaxis(ax=ax[0], max_annotation_number=n_annot_max)
    ax[0].grid(linestyle="--", linewidth=0.2, axis="both")

    # accord inter-annot
    list1 = list(df1_1annot_1label["start_datetime"])
    list2 = list(df2_1annot_1label["start_datetime"])

    unique_annotations = len([elem for elem in list1 if elem not in list2]) + len(
        [elem for elem in list2 if elem not in list1],
    )
    common_annotations = len([elem for elem in list1 if elem in list2])
    agreement = common_annotations / (unique_annotations + common_annotations)
    ax[0].text(
        0.05,
        0.9,
        f"agreement={100 * agreement:.0f}%",
        transform=ax[0].transAxes,
    )

    # scatter
    df_corr = pd.DataFrame(
        hist_plot[0] / n_annot_max,
        index=[annot_ref1, annot_ref2],
    ).transpose()

    sns.scatterplot(x=df_corr[annot_ref1], y=df_corr[annot_ref2], ax=ax[1])

    z = np.polyfit(df_corr[annot_ref1], df_corr[annot_ref2], 1)
    p = np.poly1d(z)
    plt.plot(df_corr[annot_ref1], p(df_corr[annot_ref1]), lw=1)

    ax[1].set_xlabel(f"{annot_ref1}\n{label_ref1}")
    ax[1].set_ylabel(f"{annot_ref2}\n{label_ref2}")
    ax[1].grid(linestyle="-", linewidth=0.2, axis="both")

    r, p = pearsonr(df_corr[annot_ref1], df_corr[annot_ref2])
    ax[1].text(0.05, 0.9, f"R²={r * r:.2f}", transform=ax[1].transAxes)

    plt.tight_layout()


def plot_detection_timeline(df: pd.DataFrame) -> None:
    """Plot detections on a timeline.

    Parameters
    ----------
    df: pd.DataFrame
        APLOSE DataFrame

    """
    labels = sorted(list(set(df["annotation"])))

    fig, ax = plt.subplots()

    for i, label in enumerate(labels[::-1]):
        time_det = df[(df["annotation"] == label)]["start_datetime"].to_list()
        l_data = len(time_det)
        x = np.ones((l_data, 1), int) * i
        plt.scatter(time_det, x, s=12)

    xtick_resolution = get_duration(
        msg="Enter x-tick resolution",
        default="1d",
    )
    locator = mdates.SecondLocator(interval=xtick_resolution)
    ax.xaxis.set_major_locator(locator)

    datetime_format = get_datetime_format(
        msg="Enter x-tick format",
        default="%d/%m",
    )
    formatter = mdates.DateFormatter(datetime_format)
    ax.xaxis.set_major_formatter(formatter)

    plt.grid(color="k", linestyle="-", linewidth=0.2)
    ax.set_yticks(np.arange(0, len(labels), 1))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("Date")
    plt.xlim(
        df["start_datetime"].min().normalize(),
        df["end_datetime"].max().normalize() + pd.Timedelta(days=1),
    )

    plt.tight_layout()


def _map_datetimes_to_vector(df: pd.DataFrame, timestamps: [int]) -> np.ndarray:
    """Map datetime ranges to a binary vector based on overlap with timestamp intervals.

    Parameters
    ----------
    df: pandas DataFrame
        APLOSE dataframe

    timestamps: list of int
        unix timestamps

    Returns
    -------
    An array of 0/1 based of the overlap

    """
    detection_time_beg = sorted({x.timestamp() for x in df["start_datetime"]})
    detection_time_end = sorted({y.timestamp() for y in df["end_datetime"]})

    timebin = df["end_time"].iloc[0]
    timestamp_beg = timestamps
    timestamp_end = [ts + timebin for ts in timestamps]

    vec, ranks, k = np.zeros(len(timestamps), dtype=int), [], 0
    for i in range(len(detection_time_beg)):
        for j in range(k, len(timestamps) - 1):
            if int(detection_time_beg[i] * 1e7) in range(
                int(timestamp_beg[j] * 1e7),
                int(timestamp_end[j] * 1e7),
            ) or int(detection_time_end[i] * 1e7) in range(
                int(timestamp_beg[j] * 1e7),
                int(timestamp_end[j] * 1e7),
            ):
                ranks.append(j)
                k = j
                break
            continue
    ranks = sorted(set(ranks))
    vec[np.isin(range(len(timestamps)), ranks)] = 1

    return vec


def get_detection_perf(
    df: pd.DataFrame,
    timestamps: [pd.Timestamp] = None,
    start: pd.Timestamp = None,
    stop: pd.Timestamp = None,
) -> None:
    """Compute detection performances.

    Performances are computed with a reference annotator
    in comparison with a second annotator/detector

    Parameters
    ----------
    df: pd.DataFrame
        APLOSE formatted detection/annotation DataFrame

    timestamps: list[pd.Timestamp]
        list of datetimes to base the computation on

    start: pd.Timestamp
        begin datetime, optional

    stop: pd.Timestamp
        end datetime, optional

    """
    datetime_begin = df["start_datetime"].min()
    datetime_end = df["start_datetime"].max()
    df_freq = str(df["end_time"].max()) + "s"
    annotations = df["annotation"].unique().tolist()
    annotators = df["annotator"].unique().tolist()

    min_annotators = 2
    if len(annotators) < min_annotators:
        msg = "At least 2 annotators needed"
        raise ValueError(msg)

    if not timestamps:
        timestamps = [
            ts.timestamp()
            for ts in pd.date_range(
                start=datetime_begin,
                end=datetime_end,
                freq=df_freq,
            )
        ]
    else:
        timestamps = [ts.timestamp() for ts in timestamps]

    if start and stop and start > stop:
        msg = f"Start timestamp {start} must be smaller than stop timestamp {stop}"
        raise ValueError(msg)
    if start:
        timestamps = [t for t in timestamps if t >= start.timestamp()]
        if not timestamps:
            msg = f"No annotation superior than start timestamp {start}"
            raise ValueError(msg)
    if stop:
        timestamps = [t for t in timestamps if t <= stop.timestamp()]
        if not timestamps:
            msg = f"No annotation anterior than stop timestamp {stop}"
            raise ValueError(msg)

    # df1 - REFERENCE
    selected_annotator1 = select_reference(annotators, "annotator")
    selected_label1 = select_reference(annotations)
    selected_annotations1 = df[
        (df["annotator"] == selected_annotator1) & (df["annotation"] == selected_label1)
    ]
    vec1 = _map_datetimes_to_vector(df=selected_annotations1, timestamps=timestamps)

    # df2
    selected_annotator2 = select_reference(
        [annot for annot in annotators if annot != selected_annotator1],
    )
    selected_label2 = select_reference(annotations)
    selected_annotations2 = df[
        (df["annotator"] == selected_annotator2) & (df["annotation"] == selected_label2)
    ]
    vec2 = _map_datetimes_to_vector(selected_annotations2, timestamps)

    # DETECTION PERFORMANCES
    true_pos, false_pos, true_neg, false_neg, error = 0, 0, 0, 0, 0
    for i in range(len(timestamps)):
        if vec1[i] == 0 and vec2[i] == 0:
            true_neg += 1
        elif vec1[i] == 1 and vec2[i] == 1:
            true_pos += 1
        elif vec1[i] == 0 and vec2[i] == 1:
            false_pos += 1
        elif vec1[i] == 1 and vec2[i] == 0:
            false_neg += 1
        else:
            error += 1

    if error != 0:
        msg = f"Error : {error}"
        raise ValueError(msg)

    print("\n\n### Detection results ###", end="\n")
    print(f"True positive : {true_pos}")
    print(f"True negative : {true_neg}")
    print(f"False positive : {false_pos}")
    print(f"False negative : {false_neg}")

    if true_pos + false_pos == 0 or false_neg + true_pos == 0:
        msg = "Precision/Recall computation impossible"
        raise ValueError(msg)

    print(f"\nPRECISION : {true_pos / (true_pos + false_pos):.2f}")
    print(f"RECALL : {true_pos / (false_neg + true_pos):.2f}")

    # f-score : 2 * (precision * recall) / (precision + recall)
    f_score = (
        2
        * ((true_pos / (true_pos + false_pos)) * (true_pos / (false_neg + true_pos)))
        / ((true_pos / (true_pos + false_pos)) + (true_pos / (false_neg + true_pos)))
    )
    print(f"F-SCORE : {f_score:.2f}")

    print(
        f"File 1 : {selected_annotator1}/{selected_label1} \n"
        f"File 2 : {selected_annotator2}/{selected_label2}",
    )


def _get_resolution_str(bin_sec: int) -> str:
    """From a resolution in seconds to corresponding string.

    In day/hour/minute/second resolution

    Parameters
    ----------
    bin_sec: int
        in seconds

    """
    if bin_sec // 86400 >= 1:
        bin_str = str(int(bin_sec // 86400)) + "D"
    elif bin_sec // 3600 >= 1:
        bin_str = str(int(bin_sec // 3600)) + "h"
    elif bin_sec // 60 >= 1:
        bin_str = str(int(bin_sec // 60)) + "min"
    else:
        bin_str = str(bin_sec) + "s"

    return bin_str
