from typing import Iterable
from pathlib import Path
import pytz
import pandas as pd
import numpy as np
import easygui
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from def_func import t_rounder, suntime_hour, get_duration, read_yaml, sort_detections
from collections import Counter
import seaborn as sns
from scipy.stats import pearsonr


def load_parameters_from_yaml():
    """Loads parameters from results\premier_resultats_parameters.yaml

    Returns
    -------
    df : APLOSE formatted detections
    time_bin:  list, time bin(s) in seconds
    annotators: list of annotators
    labels: list of labels
    fmax: list of maximum frequencies
    datetime_begin: pd.Timestamp
    datetime_end: pd.Timestamp
    """
    parameters = read_yaml(Path(r".\results\premiers_resultats_parameters.yaml"))

    df = pd.DataFrame()
    for file in parameters:
        df = pd.concat([df, sort_detections(**parameters[file])], ignore_index=True)

    time_bin = list(set(df["end_time"]))
    fmax = list(set(df["end_frequency"]))
    annotators = list(set(df["annotator"]))
    labels = list(set(df["annotation"]))
    tz_data = [df["start_datetime"].iloc[0].tz]

    datetime_begin = df["start_datetime"].iloc[0]
    datetime_end = df["end_datetime"].iloc[-1]

    return df, time_bin, annotators, labels, fmax, datetime_begin, datetime_end, tz_data


def select_tick_resolution(
    ax: str, default: int, lowerbound: int, upperbound: int
) -> int:
    return easygui.integerbox(
        msg=f"Choose the {ax}-axis tick resolution",
        title="Tick resolution",
        default=default,
        lowerbound=lowerbound,
        upperbound=upperbound,
    )


def select_reference(param: list[str] | pd.Series, param_str: str = ""):
    """Selection of a reference parameter (label, annotator...) from a list

    Parameters
    ----------
    param : list[str], list of the choices

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
        selection = list(set(param))[0]

    if not selection:
        raise TypeError("select_reference() has been cancelled")

    return selection


def set_plot_resolution(
    time_bin: int,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    tz: pytz.FixedOffset = None,
) -> (pd.DatetimeIndex, str, int, mpl.dates.DayLocator, mpl.dates.DateFormatter):
    """Compute the time_vector for a user-defined resolution and select appropriate date ticks and datetime format for plots
    Parameters
    ----------
    time_bin : int, length in seconds of a time bin, it is similar to the precision of the future plots
    start : pd.Timestamp, begin datetime
    stop : pd.Timestamp, end datetime
    tz : pytz.FixedOffset, timezone object, if not declared, the 'start' timezone is selected

    Returns
    -------
    datetime_index : pd.DatetimeIndex, computed timestamps according to user-defined resolution
    y_axis_plot_legend : str, string used for the y-axis legend
    number_max_of_annotation : maximum number of annotations possible according to resolution and time bin
    """
    if not tz:
        tz = start.tz

    resolution_bin = get_duration(msg="Enter the x-axis bin resolution")
    resolution_x_ticks = get_duration(
        msg="Enter the x-axis tick resolution", default="2h"
    )

    date_interval = mdates.SecondLocator(interval=resolution_x_ticks)
    date_formatter = mdates.DateFormatter("%H:%M", tz=tz)

    number_max_of_annotation = int(resolution_bin / time_bin)
    datetime_index = pd.date_range(
        start=t_rounder(start, res=resolution_bin),
        end=t_rounder(stop, res=resolution_bin),
        freq=str(resolution_bin) + "s",
    )
    if resolution_bin // 86400 > 1:
        resolution_bin_str = int(resolution_bin // 86400)
        base_str = "D"
    elif resolution_bin // 3600 > 1:
        resolution_bin_str = int(resolution_bin // 3600)
        base_str = "h"
    elif resolution_bin // 60 > 1:
        resolution_bin_str = int(resolution_bin // 60)
        base_str = "min"
    else:
        resolution_bin_str = resolution_bin
        base_str = "s"

    y_axis_plot_legend = f"Detections\n(bin resolution: {resolution_bin_str}{base_str})"

    return (
        datetime_index,
        y_axis_plot_legend,
        number_max_of_annotation,
        date_interval,
        date_formatter,
    )


def set_y_axis(ax: mpl.axes, max_annotation_number: int):
    """Changes ax properties whether the plot is visualized in percentage or in raw values

    Parameters
    ----------
    ax
    max_annotation_number
    resolution
    """
    choice_percentage = easygui.buttonbox(
        msg="Do you want your results plot in percentage or in raw values ?",
        choices=["percentage", "raw values"],
    )

    resolution = easygui.integerbox(
        msg=f"Select a y-ticks resolution\n(mode='{choice_percentage}' / max_annotation_number={max_annotation_number})",
        title="Plot resolution",
        default=10,
        lowerbound=1,
        upperbound=1440,
    )

    # change the y scale
    if choice_percentage == "percentage":
        bars = np.arange(0, 101, resolution)
        y_pos = [max_annotation_number * pos / 100 for pos in bars]

        if isinstance(ax, Iterable):
            [a.set_yticks(y_pos, bars) for a in ax]
            [a.set_ylabel("%") for a in ax]
        else:
            ax.set_yticks(y_pos, bars)
            ax.set_ylabel("%")
    else:
        if isinstance(ax, Iterable):
            [
                a.set_yticks(
                    np.arange(0, max_annotation_number + resolution, resolution)
                )
                for a in ax
            ]
        else:
            ax.set_yticks(np.arange(0, max_annotation_number + resolution, resolution))

    if isinstance(ax, Iterable):
        [a.set_ylim([0, max_annotation_number]) for a in ax]
    else:
        ax.set_ylim([0, max_annotation_number])
    return choice_percentage


def overview_plot(df: pd.DataFrame):
    """Overview of an APLOSE formatted dataframe

    Parameters
    ----------
    df: pd.DataFrame, APLOSE formatted result DataFrame with detections and associated timestamps
    """
    summary_label = (
        df.groupby("annotation")["annotator"].apply(Counter).unstack(fill_value=0)
    )
    summary_annotator = (
        df.groupby("annotator")["annotation"].apply(Counter).unstack(fill_value=0)
    )

    print(f"\n- Overview of the detections -\n\n {summary_label}")

    fig, ax = plt.subplots(2, 1)
    ax[0] = summary_label.plot(kind="bar", ax=ax[0], edgecolor="black", linewidth=1)
    ax[1] = summary_annotator.plot(kind="bar", ax=ax[1], edgecolor="black", linewidth=1)

    for a in ax:
        # legend
        a.legend(loc="best", frameon=1, framealpha=0.6)
        # ticks
        a.tick_params(axis="both", rotation=0)
        a.set_ylabel("Number of annotated calls")
        # y-grids
        a.yaxis.grid(color="gray", linestyle="--")
        a.set_axisbelow(True)

    # labels
    ax[0].set_xlabel("Labels")
    ax[1].set_xlabel("Annotator")

    # titles
    ax[0].set_title("Number of annotations per label")
    ax[1].set_title("Number of annotations per annotator")

    plt.tight_layout()
    plt.show()

    return


def plot_hourly_detection_rate(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    date_format: str = "%H:%M",
    show_rise_set: bool = True,
):
    """Computes the hourly detection rate for an APLOSE formatted result DataFrame

    Parameters
    ----------
    df : pd.DataFrame, APLOSE formatted result DataFrame with detections and associated timestamps
    lat: float, latitude
    lon: float, longitude
    date_format: str, default '%H:%M'
    show_rise_set : bool, default True, display the sunrise and sunset lines
    """
    datetime_begin = df["start_datetime"].iloc[0]
    datetime_end = df["end_datetime"].iloc[-1]
    tz = datetime_begin.tz

    bin = select_reference(df["end_time"], "time bin")

    # compute sunrise and sunset decimal hour at the dataset location
    sunrise, sunset, _, _, _, _ = suntime_hour(
        start=datetime_begin, stop=datetime_end, lat=lat, lon=lon
    )

    df["date"] = [ts.normalize() for ts in df["start_datetime"]]
    df["hour"] = [ts.hour for ts in df["start_datetime"]]

    det_groupby = df.groupby(["date", "hour"]).size()
    idx_day_groupby = det_groupby.index.get_level_values(0)
    idx_hour_groupby = det_groupby.index.get_level_values(1)

    dates = pd.date_range(
        datetime_begin.normalize(), datetime_end.normalize(), freq="D"
    )
    M = np.zeros((24, len(dates)))
    for idx_j, j in enumerate(dates):
        # Search for detection in day = j
        f = [idx for idx, det in enumerate(idx_day_groupby) if det == j]
        if f:
            for ff in f:
                hour = idx_hour_groupby[ff]
                M[int(hour), idx_j] = det_groupby.iloc[ff]

    # plot
    fig, ax = plt.subplots()
    im = ax.imshow(
        M,
        extent=[datetime_begin, datetime_end, 0, 24],
        vmin=0,
        vmax=3600 / bin,
        aspect="auto",
        origin="lower",
    )
    # colorbar
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel(f"{str(bin)}s bins positive to detection")

    if show_rise_set:
        plt.plot(dates, sunrise, color="darkorange", linewidth=1)
        plt.plot(dates, sunset, color="royalblue", linewidth=1)

    # axes settings
    ax.xaxis_date()
    resolution_x_ticks = get_duration(
        msg="Enter the x-axis tick resolution", default="2h"
    )
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=resolution_x_ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt=date_format, tz=tz))
    ax.set_ylabel("Hour")
    ax.set_xlabel("Date")
    plt.xlim(datetime_begin, datetime_end)
    plt.tight_layout()
    plt.show()

    return


def scatter_detections(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    date_format: str = "%H:%M",
    show_rise_set: bool = True,
):
    """Plot scatter of the detections from an APLOSE formatted DataFrame.
    Additionally, sunrise and sunset lines can be plotted if show_rise_set is set to True (default value)

    Parameters
    ----------
    df : pd.DataFrame, APLOSE formatted result DataFrame with detections and associated timestamps
    lat : float, latitude
    lon : float, longitude
    date_format : string template, default '%H:%M'
    show_rise_set : bool, default True, display the sunrise and sunset lines
    """
    datetime_begin = df["start_datetime"].iloc[0]
    datetime_end = df["end_datetime"].iloc[-1]
    tz = datetime_begin.tz

    # compute sunrise and sunset decimal hour at the dataset location
    hour_sunrise, hour_sunset, _, _, _, _ = suntime_hour(
        start=datetime_begin, stop=datetime_end, lat=lat, lon=lon
    )

    dates = pd.date_range(datetime_begin.date(), datetime_end.date(), freq="D")

    # decimal hours of detections
    hour_det = [
        ts.hour + ts.minute / 60 + ts.second / 3600 for ts in df["start_datetime"]
    ]

    # plot
    fig, ax = plt.subplots()
    plt.scatter(
        df["start_datetime"], hour_det, marker="x", linewidths=1, color="silver"
    )
    if show_rise_set:
        plt.plot(dates, hour_sunrise, color="darkorange")
        plt.plot(dates, hour_sunset, color="royalblue")

    # axes settings
    plt.xlim(datetime_begin, datetime_end)
    plt.ylim(0, 24)

    resolution_x_ticks = get_duration(
        msg="Enter the x-axis tick resolution", default="2h"
    )
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=resolution_x_ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt=date_format, tz=tz))
    ax.grid(color="k", linestyle="-", linewidth=0.2)
    ax.set_ylabel("Hour")
    ax.set_xlabel("Date")

    # title
    plt.title(
        f"Time of detections within each day for dataset {select_reference(df['dataset'])}"
    )

    plt.tight_layout()
    plt.show()

    return


def single_plot(df: pd.DataFrame):
    """Plot the detections of an APLOSE formatted DataFrame for a single label

    Parameters
    ----------
    df: pd.DataFrame, APLOSE formatted result DataFrame with detections and associated timestamps
    """
    # selection of the references parameters
    datetime_begin = df["start_datetime"].iloc[0]
    datetime_end = df["end_datetime"].iloc[-1]
    annotators = list(set(df["annotator"]))
    annot_ref = select_reference(annotators, "annotator")
    label_ref = select_reference(
        df[df["annotator"] == annot_ref]["annotation"], "label"
    )
    time_bin_ref = select_reference(
        df[df["annotator"] == annot_ref]["end_time"], "time bin"
    )

    # set plot resolution
    time_vector, y_label_legend, n_annot_max, mdate1, mdate2 = set_plot_resolution(
        time_bin=time_bin_ref, start=datetime_begin, stop=datetime_end
    )

    df_1annot_1label = df[
        (df["annotator"] == annot_ref) & (df["annotation"] == label_ref)
    ]

    # plot
    fig, ax = plt.subplots()
    [hist_y, hist_x, _] = ax.hist(
        df_1annot_1label["start_datetime"],
        bins=time_vector,
    )

    # title
    plt.title(f"annotator: {annot_ref}\nlabel: {label_ref}")

    # axes settings
    ax.xaxis.set_major_locator(mdate1)
    ax.xaxis.set_major_formatter(mdate2)
    plt.xlim(time_vector[0], time_vector[-1])
    ax.grid(linestyle="--", linewidth=0.2, axis="both")
    ax.set_ylabel(y_label_legend)
    set_y_axis(ax, n_annot_max)

    plt.tight_layout()
    plt.show()

    return


def multilabel_plot(df: pd.DataFrame):
    """Plot the detections of an APLOSE formatted DataFrame for a all labels

    Parameters
    ----------
    df: pd.DataFrame, APLOSE formatted result DataFrame with detections and associated timestamps
    """
    datetime_begin = df["start_datetime"].iloc[0]
    datetime_end = df["end_datetime"].iloc[-1]
    annotators = list(set(df["annotator"]))
    annot_ref = select_reference(annotators, "annotator")
    time_bin_ref = select_reference(
        df[df["annotator"] == annot_ref]["end_time"], "time bin"
    )
    list_labels = list(set(df[df["annotator"] == annot_ref]["annotation"]))

    if not len(list_labels) > 1:
        raise ValueError(
            f"Only {str(len(list_labels))} labels detected, multilabel plot cancelled"
        )

    time_vector, y_label_legend, n_annot_max, mdate1, mdate2 = set_plot_resolution(
        time_bin=time_bin_ref, start=datetime_begin, stop=datetime_end, tz=None
    )

    fig, ax = plt.subplots(
        nrows=len(list_labels),
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
        ax[i].xaxis.set_major_locator(mdate1)
        ax[i].xaxis.set_major_formatter(mdate2)
        ax[i].set_xlim(time_vector[0], time_vector[-1])
        ax[i].grid(linestyle="--", linewidth=0.2, axis="both")

    set_y_axis(ax, n_annot_max)
    fig.suptitle(
        f"Annotator : {annot_ref}",
        weight="bold",
    )
    fig.supylabel(y_label_legend, ha="center")
    plt.tight_layout()
    plt.show()

    return


def multiuser_plot(df: pd.DataFrame):

    datetime_begin = df["start_datetime"].iloc[0]
    datetime_end = df["end_datetime"].iloc[-1]
    annotators = list(set(df["annotator"]))

    if len(annotators) < 2:
        raise ValueError(
            "Only 1 annotator detected, multiuser plot cancelled"
        )

    annot_ref1 = select_reference(annotators, "annotator 1")
    annot_ref2 = select_reference(
        [elem for elem in annotators if elem != annot_ref1], "annotator 2"
    )

    label_ref1 = select_reference(
        df[df["annotator"] == annot_ref1]["annotation"], "label 1"
    )
    if label_ref1 not in list(set(df[df["annotator"] == annot_ref2]["annotation"])):
        label_ref2 = select_reference(
            df[df["annotator"] == annot_ref2]["annotation"], "label 2"
        )
    else:
        label_ref2 = label_ref1

    time_bin_ref1 = select_reference(
        df[df["annotator"] == annot_ref1]["end_time"], "time bin 1"
    )
    time_bin_ref2 = select_reference(
        df[df["annotator"] == annot_ref2]["end_time"], "time bin 2"
    )

    if time_bin_ref1 != time_bin_ref1:
        raise ValueError(
            f"The timebin of the detections {annot_ref1}/{label_ref1} is {time_bin_ref1}s"
            f" whereas the timebin for {annot_ref2}/{label_ref2} is {time_bin_ref2}s"
        )
    else:
        time_bin_ref = time_bin_ref1

    # set plot resolution
    time_vector, y_label_legend, n_annot_max, mdate1, mdate2 = set_plot_resolution(
        time_bin=time_bin_ref, start=datetime_begin, stop=datetime_end
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
    ax[0].legend(loc="upper right")

    fig.suptitle(f"[{annot_ref1}/{label_ref1}] VS [{annot_ref2}/{label_ref2}]")
    # axes settings
    ax[0].xaxis.set_major_locator(mdate1)
    ax[0].xaxis.set_major_formatter(mdate2)
    ax[0].set_xlim(time_vector[0], time_vector[-1])
    ax[0].grid(linestyle="--", linewidth=0.2, axis="both")
    ax[0].set_ylabel(y_label_legend)
    set_y_axis(ax[0], n_annot_max)
    ax[0].grid(linestyle="--", linewidth=0.2, axis="both")

    # accord inter-annot
    list1 = list(df1_1annot_1label["start_datetime"])
    list2 = list(df2_1annot_1label["start_datetime"])

    unique_annotations = len([elem for elem in list1 if elem not in list2]) + len(
        [elem for elem in list2 if elem not in list1]
    )
    common_annotations = len([elem for elem in list1 if elem in list2])
    agreement = (common_annotations) / (unique_annotations + common_annotations)
    ax[0].text(
        0.05, 0.9, f"agreement={100 * agreement:.0f}%", transform=ax[0].transAxes
    )

    # scatter
    df_corr = pd.DataFrame(
        hist_plot[0] / n_annot_max, index=[annot_ref1, annot_ref2]
    ).transpose()
    sns.scatterplot(x=df_corr[annot_ref1], y=df_corr[annot_ref2], ax=ax[1])

    z = np.polyfit(df_corr[annot_ref1], df_corr[annot_ref2], 1)
    p = np.poly1d(z)
    plt.plot(df_corr[annot_ref1], p(df_corr[annot_ref1]), lw=1)

    ax[1].set_xlabel(f"{annot_ref1}\n{label_ref1}")
    ax[1].set_ylabel(f"{annot_ref2}\n{label_ref2}")
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].grid(linestyle="-", linewidth=0.2, axis="both")

    r, p = pearsonr(df_corr[annot_ref1], df_corr[annot_ref2])
    ax[1].text(0.05, 0.9, f"R²={r * r:.2f}", transform=ax[1].transAxes)

    plt.tight_layout()
    plt.show()

    return
