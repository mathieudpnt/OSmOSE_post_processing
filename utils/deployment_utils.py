import pytz
from typing import List
import pandas as pd
import datetime as dt
import numpy as np
import easygui
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.dates as mdates
from utils.Deployment import Deployment
from utils.def_func import reshape_timebin, t_rounder

mpl.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["figure.figsize"] = [10, 4]


def plot_single(
    data: Deployment,
    file: str,
    timebin: int = 60,
    begin_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    fmin_filter: int = None,
    fmax_filter: int = None,
) -> None:
    assert isinstance(data, Deployment), "not a Deployment class"
    assert hasattr(data, "df_" + file), f"{file} results not available"
    if fmin_filter is not None:
        assert isinstance(fmin_filter, int), "An integer must be passed for fmin_filter"
    if fmax_filter is not None:
        assert isinstance(fmax_filter, int), "An integer must be passed for fmin_filter"
    if all(isinstance(f, int) for f in [fmin_filter, fmax_filter]):
        assert fmin_filter < fmax_filter, "fmin_filter must be < than fmax_filter"

    df = getattr(data, f"df_{file}")

    if fmin_filter is not None:
        df = df[df["start_frequency"] >= fmin_filter]
        assert (
            len(df) > 0
        ), f"No detection found in {file[0]} DataFrame after fmin filtering at {fmin_filter} Hz, upload aborted"

    if fmax_filter is not None:
        df = df[df["start_frequency"] <= fmax_filter]
        assert (
            len(df) > 0
        ), f"No detection found in {file[0]} DataFrame after fmax filtering at {fmax_filter} Hz, upload aborted"

    annotator = getattr(data, f"{file}_annotator")
    annotation = getattr(data, f"{file}_annotation")

    if not begin_date:
        begin_date = getattr(data, "start_date")
    if not end_date:
        end_date = getattr(data, "end_date")

    df = df[(df["start_datetime"] >= begin_date) & (df["end_datetime"] <= end_date)]
    assert len(df) > 0, f"No annotation found between {begin_date} and {end_date}"

    timestamp = getattr(data, "segment_timestamp")
    timestamp2 = list(
        dict.fromkeys([ts for ts in timestamp if begin_date <= ts <= end_date])
    )
    tz = pytz.FixedOffset(begin_date.utcoffset().total_seconds() // 60)

    if len(annotator) > 1 and not isinstance(annotator, str):
        annot_ref = easygui.buttonbox("Select annotator", "Single plot", annotator)
    elif isinstance(annotator, str):
        annot_ref = annotator

    df_reshaped = reshape_timebin(df=df, timebin_new=timebin, timestamp=timestamp2)

    # list of the labels corresponding to the selected user
    if isinstance(annotation, str):
        label_ref = annotation
    else:
        label_ref = easygui.buttonbox(
            f"Select a label for {annot_ref}", "Single plot", annotation
        )

    res_min = easygui.integerbox(
        "Enter the bin size (min) ",
        "Time resolution",
        default=10,
        lowerbound=1,
        upperbound=86400,
    )
    delta, start_vec, end_vec = (
        pd.Timedelta(seconds=60 * res_min),
        t_rounder(begin_date, res=600),
        t_rounder(end_date + pd.Timedelta(seconds=timebin), res=600),
    )
    time_vector = [
        start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)
    ]
    n_annot_max = (
        res_min * 60
    ) / timebin  # max nb of annotated time_bin max per res_min slice
    df_1annot_1label = df_reshaped[
        (df_reshaped["annotator"] == annot_ref)
        & (df_reshaped["annotation"] == label_ref)
    ]

    fig, ax = plt.subplots(1, 1)
    ax.hist(df_1annot_1label["start_datetime"], bins=time_vector, edgecolor="black")
    bars = range(0, 101, 10)  # from 0 to 100 step 10
    y_pos = np.linspace(0, n_annot_max, num=len(bars))
    ax.set_yticks(y_pos, bars)
    ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel(f"positive detection rate\n{timebin}s window per {res_min}min bin")
    ax.tick_params(axis="y")
    fig.suptitle(f"{data.name}\n[{annot_ref}/{label_ref}]")

    if data.duration < pd.Timedelta(12, "hour"):
        t_inter = 1
        date_fmt = "%H:%M"
    elif data.duration < pd.Timedelta(24, "hour"):
        t_inter = 2
        date_fmt = "%H:%M"
    elif data.duration < pd.Timedelta(4, "day"):
        t_inter = 4
        date_fmt = "%m/%d - %H:%M"
    elif data.duration < pd.Timedelta(7, "day"):
        t_inter = 24
        date_fmt = "%m/%d"

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=t_inter))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt, tz=tz))
    ax.set_xlim(time_vector[0], time_vector[-1])
    ax.grid(linestyle="-", linewidth=0.2, axis="both")

    plt.tight_layout()
    plt.show()

    return df_1annot_1label["start_datetime"], time_vector


def get_agreement(
    data: Deployment,
    file: str,
    timebin: int = 60,
    begin_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    fmin_filter: int = None,
    fmax_filter: int = None,
) -> None:
    assert isinstance(data, Deployment), "not a Deployment class"
    assert hasattr(data, "df_" + file), f"{file} results not available"
    if fmin_filter is not None:
        assert isinstance(fmin_filter, int), "An integer must be passed for fmin_filter"
    if fmax_filter is not None:
        assert isinstance(fmax_filter, int), "An integer must be passed for fmin_filter"
    if all(isinstance(f, int) for f in [fmin_filter, fmax_filter]):
        assert fmin_filter < fmax_filter, "fmin_filter must be < than fmax_filter"

    df = getattr(data, f"df_{file}")

    if fmin_filter is not None:
        df = df[df["start_frequency"] >= fmin_filter]
        assert (
            len(df) > 0
        ), f"No detection found in {file[0]} DataFrame after fmin filtering at {fmin_filter} Hz, upload aborted"

    if fmax_filter is not None:
        df = df[df["start_frequency"] <= fmax_filter]
        assert (
            len(df) > 0
        ), f"No detection found in {file[0]} DataFrame after fmax filtering at {fmax_filter} Hz, upload aborted"

    annotator = getattr(data, f"{file}_annotator")
    annotation = getattr(data, f"{file}_annotation")
    if not begin_date:
        begin_date = getattr(data, "start_date")
    if not end_date:
        end_date = getattr(data, "end_date")
    timestamp = getattr(data, "segment_timestamp")
    timestamp2 = [ts for ts in timestamp if begin_date <= ts <= end_date]

    tz = pytz.FixedOffset(begin_date.utcoffset().total_seconds() // 60)

    assert (
        isinstance(annotator, List) and len(annotator) == 2
    ), "Agreement plot cancelled, not enough annotators to make a comparison"
    if len(annotator) > 2:
        annot_ref1 = easygui.buttonbox("Select annotator 1", "Plot label", annotator)
        annot_ref2 = easygui.buttonbox(
            "Select an annotator",
            "Plot label",
            [elem for elem in annotator if elem != annot_ref1],
        )

    annot_ref1 = annotator[0]
    annot_ref2 = annotator[1]

    df_reshaped = reshape_timebin(df=df, timebin_new=timebin, timestamp=timestamp2)

    # list of the labels corresponding to the selected user
    if not isinstance(annotation, str):
        label_ref = easygui.buttonbox(
            "Select the label to plot", "Single plot", annotation
        )
    else:
        label_ref = annotation

    res_min = easygui.integerbox(
        "Enter the bin size (min) ",
        "Time resolution",
        default=10,
        lowerbound=1,
        upperbound=86400,
    )
    delta, start_vec, end_vec = (
        dt.timedelta(seconds=60 * res_min),
        t_rounder(begin_date, res=600),
        t_rounder(end_date + dt.timedelta(seconds=timebin), res=600),
    )
    time_vector = [
        start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)
    ]
    n_annot_max = (
        res_min * 60
    ) / timebin  # max nb of annoted time_bin max per res_min slice

    df1_1annot_1label = df_reshaped[
        (df_reshaped["annotator"] == annot_ref1)
        & (df_reshaped["annotation"] == label_ref)
    ]
    df2_1annot_1label = df_reshaped[
        (df_reshaped["annotator"] == annot_ref2)
        & (df_reshaped["annotation"] == label_ref)
    ]

    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [8, 2]})

    hist_plot = axs[0].hist(
        [df1_1annot_1label["start_datetime"], df2_1annot_1label["start_datetime"]],
        bins=time_vector,
        label=[annot_ref1, annot_ref2],
        edgecolor="black",
    )
    axs[0].legend(loc="upper right")

    bars = range(0, 101, 10)  # from 0 to 100 step 10
    y_pos = np.linspace(0, n_annot_max, num=len(bars))
    axs[0].set_yticks(y_pos, bars)
    axs[0].tick_params(axis="x", rotation=60)
    axs[0].set_ylabel(
        f"positive detection rate\n{timebin}s window per {res_min}min bin"
    )
    axs[0].tick_params(axis="y")
    fig.suptitle(f"[{annot_ref1}/{label_ref}] VS [{annot_ref2}/{label_ref}]")

    axs[0].xaxis.set_major_locator(mdates.HourLocator(interval=4))
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    axs[0].set_xlim(time_vector[0], time_vector[-1])
    axs[0].grid(linestyle="-", linewidth=0.2, axis="both")

    # agreement
    list1 = list(df1_1annot_1label["start_datetime"])
    list2 = list(df2_1annot_1label["start_datetime"])

    unique_annotations = len([elem for elem in list1 if elem not in list2]) + len(
        [elem for elem in list2 if elem not in list1]
    )
    common_annotations = len([elem for elem in list1 if elem in list2])
    agreement = (common_annotations) / (unique_annotations + common_annotations)
    axs[0].text(
        0.05, 0.9, f"agreement={100 * agreement:.0f}%", transform=axs[0].transAxes
    )

    # scatter
    df_corr = pd.DataFrame(
        hist_plot[0] / n_annot_max, index=[annot_ref1, annot_ref2]
    ).transpose()
    sns.scatterplot(x=df_corr[annot_ref1], y=df_corr[annot_ref2], ax=axs[1])

    z = np.polyfit(df_corr[annot_ref1], df_corr[annot_ref2], 1)
    p = np.poly1d(z)
    plt.plot(df_corr[annot_ref1], p(df_corr[annot_ref1]), lw=1)

    axs[1].set_xlabel("{0}\n{1}".format(annot_ref1, label_ref))
    axs[1].set_ylabel("{0}\n{1}".format(annot_ref2, label_ref))
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].grid(linestyle="-", linewidth=0.2, axis="both")

    r, p = stats.pearsonr(df_corr[annot_ref1], df_corr[annot_ref2])
    axs[1].text(0.05, 0.9, f"R²={r * r:.2f}", transform=axs[1].transAxes)

    plt.tight_layout()
    plt.show()

    return


def get_perf(
    data: Deployment,
    file: List[str],
    timebin: int = 60,
    begin_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    fmin_filter: List[int] = None,
    fmax_filter: List[int] = None,
) -> None:
    assert isinstance(data, Deployment), "data arg is not a Deployment class"
    assert isinstance(file, List), "file arg is not a list"
    assert len(file) == 2, f"{len(file)} arg were passed instead of 2 for file arg"
    assert hasattr(data, "df_" + file[0]), f"{file[0]} results not available"
    assert hasattr(data, "df_" + file[1]), f"{file[1]} results not available"
    if fmin_filter is not None:
        assert isinstance(
            fmin_filter, List
        ), "A list of 2 integers must be passed for fmin_filter"
        assert (
            len(fmin_filter) == 2
        ), "A list of 2 integers must be passed for fmin_filter"
    if fmax_filter is not None:
        assert isinstance(
            fmax_filter, List
        ), "A list of 2 integers must be passed for fmin_filter"
        assert (
            len(fmax_filter) == 2
        ), "A list of 2 integers must be passed for fmax_filter"
    if all(isinstance(f, List) for f in [fmin_filter, fmax_filter]) and all(
        len(f) == 2 for f in [fmin_filter, fmax_filter]
    ):
        for i in range(2):
            assert (
                fmin_filter[i] < fmax_filter[i]
            ), "fmin_filter must be < than fmax_filter"

    df1 = getattr(data, f"df_{file[0]}")
    df2 = getattr(data, f"df_{file[1]}")

    if fmin_filter is not None:
        df1 = df1[df1["start_frequency"] >= fmin_filter[0]]
        df2 = df2[df2["start_frequency"] >= fmin_filter[1]]
        assert (
            len(df1) > 0
        ), f"No detection found in {file[0]} DataFrame after fmin filtering at {fmin_filter} Hz, upload aborted"
        assert (
            len(df2) > 0
        ), f"No detection found in {file[1]} DataFrame after fmin filtering at {fmin_filter} Hz, upload aborted"

    if fmax_filter is not None:
        df1 = df1[df1["start_frequency"] <= fmax_filter[0]]
        df2 = df2[df2["start_frequency"] <= fmax_filter[1]]
        assert (
            len(df1) > 0
        ), f"No detection found in {file[0]} DataFrame after fmax filtering at {fmax_filter} Hz, upload aborted"
        assert (
            len(df2) > 0
        ), f"No detection found in {file[1]} DataFrame after fmax filtering at {fmax_filter} Hz, upload aborted"

    annotator1 = getattr(data, f"{file[0]}_annotator")
    annotator2 = getattr(data, f"{file[1]}_annotator")

    annotation1 = getattr(data, f"{file[0]}_annotation")
    annotation2 = getattr(data, f"{file[1]}_annotation")

    if not begin_date:
        begin_date = getattr(data, "start_date")
    if not end_date:
        end_date = getattr(data, "end_date")

    timestamp = getattr(data, "segment_timestamp")
    timestamp2 = [ts for ts in timestamp if begin_date <= ts <= end_date]

    t = t_rounder(t=max(timestamp2[0], begin_date), res=timebin)
    t2 = t_rounder(min(timestamp2[-1], end_date), res=timebin) + pd.Timedelta(
        seconds=timebin
    )
    time_vector = [
        ts.timestamp() for ts in pd.date_range(start=t, end=t2, freq=str(timebin) + "s")
    ]

    tz = pytz.FixedOffset(begin_date.utcoffset().total_seconds() // 60)

    # annotator
    if len(annotator1) > 1 and not isinstance(annotator1, str):
        selected_annotator1 = easygui.buttonbox(
            "Select annotator", "Performances", annotator1
        )
    elif isinstance(annotator1, str):
        selected_annotator1 = annotator1

    if len(annotator2) > 1 and not isinstance(annotator2, str):
        selected_annotator2 = easygui.buttonbox(
            "Select annotator", "Performances", annotator2
        )
    elif isinstance(annotator2, str):
        selected_annotator2 = annotator2

    # label
    if len(annotation1) > 1 and not isinstance(annotation1, str):
        selected_label1 = easygui.buttonbox(
            "Select annotation", "Performances", annotation1
        )
    elif isinstance(annotation1, str):
        selected_label1 = annotation1

    if len(annotation2) > 1 and not isinstance(annotation2, str):
        selected_label2 = easygui.buttonbox(
            "Select annotation", "Performances", annotation2
        )
    elif isinstance(annotation2, str):
        selected_label2 = annotation2

    # df1 - REFERENCE
    df1_reshaped = reshape_timebin(df=df1, timebin_new=timebin, timestamp=timestamp)

    selected_annotations1 = df1_reshaped[
        (df1_reshaped["annotator"] == selected_annotator1)
        & (df1_reshaped["annotation"] == selected_label1)
        & (df1_reshaped["start_datetime"] >= begin_date)
        & (df1_reshaped["end_datetime"] <= end_date)
    ]

    times1_beg = sorted(
        list(set(x.timestamp() for x in selected_annotations1["start_datetime"]))
    )
    times1_end = sorted(
        list(set(y.timestamp() for y in selected_annotations1["end_datetime"]))
    )

    vec1, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
    for i in range(len(times1_beg)):
        for j in range(k, len(time_vector) - 1):
            if (
                times1_beg[i] >= time_vector[j] and times1_beg[i] < time_vector[j + 1]
            ) or (
                times1_end[i] > time_vector[j] and times1_end[i] <= time_vector[j + 1]
            ):
                ranks.append(j)
                k = j
                break
            else:
                continue
    ranks = sorted(list(set(ranks)))
    vec1[np.isin(range(len(time_vector)), ranks)] = 1

    # df2
    df2_reshaped = reshape_timebin(df=df2, timebin_new=timebin, timestamp=timestamp)

    selected_annotations2 = df2_reshaped[
        (df2_reshaped["annotator"] == selected_annotator2)
        & (df2_reshaped["annotation"] == selected_label2)
        & (df2_reshaped["start_datetime"] >= begin_date)
        & (df2_reshaped["end_datetime"] <= end_date)
    ]

    times2_beg = [i.timestamp() for i in selected_annotations2["start_datetime"]]
    times2_end = [i.timestamp() for i in selected_annotations2["end_datetime"]]

    vec2, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
    for i in range(len(times2_beg)):
        for j in range(k, len(time_vector) - 1):
            if (
                times2_beg[i] >= time_vector[j] and times2_beg[i] < time_vector[j + 1]
            ) or (
                times2_end[i] > time_vector[j] and times2_end[i] <= time_vector[j + 1]
            ):
                ranks.append(j)
                k = j
                break
            else:
                continue
    ranks = sorted(list(set(ranks)))
    vec2[np.isin(range(len(time_vector)), ranks)] = 1

    # DETECTION PERFORMANCES
    true_pos, false_pos, true_neg, false_neg, error = 0, 0, 0, 0, 0
    for i in range(len(time_vector)):
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
    f_score = (
        2
        * ((true_pos / (true_pos + false_pos)) * (true_pos / (false_neg + true_pos)))
        / ((true_pos / (true_pos + false_pos)) + (true_pos / (false_neg + true_pos)))
    )

    if error != 0:
        raise ValueError(f"Error : {error}")

    print("\n-- Detection results --", end="\n")
    print(f" - Timebin : {timebin}s\n")
    print(f" - True positive : {true_pos}")
    print(f" - True negative : {true_neg}")
    print(f" - False positive : {false_pos}")
    print(f" - False negative : {false_neg}")
    print(f"\n - Precision : {true_pos / (true_pos + false_pos):.2f}")
    print(f" - Recall : {true_pos / (false_neg + true_pos):.2f}")
    print(f" - F-score : {f_score:.2f}")

    # plot
    res_min = easygui.integerbox(
        "Enter the bin size (min) ",
        "Time resolution",
        default=10,
        lowerbound=1,
        upperbound=86400,
    )
    delta, start_vec, end_vec = (
        pd.Timedelta(seconds=60 * res_min),
        t_rounder(begin_date, res=600),
        t_rounder(end_date + pd.Timedelta(seconds=timebin), res=600),
    )
    bin_vector = [
        start_vec + i * delta for i in range(int((end_vec - start_vec) / delta) + 1)
    ]
    n_annot_max = (
        res_min * 60
    ) / timebin  # max nb of annoted time_bin max per res_min slice

    fig, ax = plt.subplots(1, 1, dpi=200, figsize=(10, 4))

    ax.hist(
        [
            selected_annotations1["start_datetime"],
            selected_annotations2["start_datetime"],
        ],
        bins=bin_vector,
        label=[selected_label1, selected_label2],
        edgecolor="black",
    )
    ax.legend(loc="upper right")

    bars = range(0, 101, 10)  # from 0 to 100 step 10
    y_pos = np.linspace(0, n_annot_max, num=len(bars))
    ax.set_yticks(y_pos, bars)
    ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel(f"positive detection rate\n({timebin}s window per {res_min}min bin)")
    ax.tick_params(axis="y")
    fig.suptitle(
        f"[{selected_annotator1}/{selected_label1}] VS [{selected_annotator2}/{selected_label2}]"
    )

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax.set_xlim(bin_vector[0], bin_vector[-1])
    ax.grid(linestyle="-", linewidth=0.2, axis="both")

    plt.tight_layout()
    plt.show()
    return
