from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from numpy import ceil
from pandas import DataFrame, Series, date_range, Timedelta
from pandas.tseries import offsets
import logging
from collections import Counter

from src.post_processing.def_func import get_datetime_format, get_duration, t_rounder
from src.post_processing.premiers_resultats_utils import (
    get_resolution_str,
    select_reference,
)


def get_locator_from_offset(offset) -> mdates.DateLocator:
    """Map a pandas offset object to the appropriate matplotlib DateLocator."""
    if isinstance(offset, int):
        return mdates.SecondLocator(interval=offset)

    offset_to_locator = {
        (
            offsets.MonthEnd,
            offsets.MonthBegin,
            offsets.BusinessMonthEnd,
            offsets.BusinessMonthBegin,
        ): mdates.MonthLocator,
        (offsets.Day,): mdates.DayLocator,
        (offsets.Hour,): mdates.HourLocator,
        (offsets.Minute,): mdates.MinuteLocator,
    }

    for offset_classes, locator_cls in offset_to_locator.items():
        if isinstance(offset, offset_classes):
            return locator_cls(interval=offset.n)

    raise ValueError(f"Unsupported offset type: {type(offset)}")


class DataAplose:
    """A class to handle annotation data for APLOSE.

    Attributes
    ----------
    df (DataFrame): An APLOSE formatted DataFrame
    annotators ([str]): Annotator list.
    labels ([str]): Label list.
    begin (Timestamp): Earliest detection.
    end (Timestamp): Latest detection.
    _time_bin (int): Internal time bin resolution used for plotting.
    _resolution_bin (int): Bin resolution in seconds for histogram plots.
    _resolution_x_ticks (int or pd.offsets.DateOffset): Resolution for x-axis ticks.

    """

    def __init__(self, df: DataFrame) -> None:
        """Construct all the necessary attributes for the DataAplose object."""
        self.df = df
        self.annotators = list(set(self.df["annotator"]))
        self.labels = list(set(self.df["annotation"]))
        self.begin = min(self.df["start_datetime"])
        self.end = max(self.df["end_datetime"])
        self._time_bin = None
        self._resolution_bin = None
        self._resolution_x_ticks = None


    def __str__(self) -> str:
        """Return string representation of DataAplose object."""
        return (
            f"annotators: {self.annotators}\n"
            f"labels: {self.labels}\n"
            f"begin: {self.begin}\n"
            f"end: {self.end}"
        )


    def __getitem__(self, item: int) -> Series:
        """Allow indexing to retrieve rows from the underlying DataFrame."""
        return self.df.iloc[item]


    def plot(self, annotator: str, label: str, ax: plt.Axes) -> None:
        """Seasonality plot."""
        if annotator not in self.annotators:
            msg = f'Annotator "{annotator}" not in APLOSE DataFrame'
            raise ValueError(msg)
        if label not in self.labels:
            msg = f'Label "{label}" not in APLOSE DataFrame'
            raise ValueError(msg)
        if self.df[self.df["is_box"] == 0].empty:
            msg = "DataFrame contains no weak detection, consider reshaping it first"
            raise ValueError(msg)

        df_1annot_1label = self.df[
            (self.df["annotator"] == annotator) & (self.df["annotation"] == label)
            ]

        bins = date_range(
            start=t_rounder(t=self.begin, res=self._resolution_bin),
            end=t_rounder(t=self.end, res=self._resolution_bin),
            freq=str(self._resolution_bin) + "s",
        )

        val1, _, _ = ax.hist(
            df_1annot_1label["start_datetime"],
            bins=bins,
            edgecolor="black",
            zorder=2,
        )

        ax.set_ylim(0, 1.05 * max(val1))
        ax.set_yticks(range(0, int(ceil(max(val1))) + 1, max(1, int(ceil(max(val1))) // 4)))
        ax.title.set_text(f"annotator: {annotator}\nlabel: {label}")


    def set_ax(self, ax: plt.Axes, bin_size: Timedelta = None, xticks_res: Timedelta | offsets.DateOffset = None) -> plt.Axes:
        """Set up axis configuration for plot."""
        self._time_bin = int(
            select_reference(self.df[self.df["is_box"] == 0]["end_time"], "time bin"),
        )
        time_bin_str = get_resolution_str(self._time_bin)
        self._resolution_bin = get_duration(msg="Enter bin resolution") if not bin_size else bin_size.total_seconds()
        resolution_bin_str = get_resolution_str(self._resolution_bin)
        self._resolution_x_ticks = get_duration(
            msg="Enter x-axis tick resolution",
            default="2h",
        ) if not xticks_res else xticks_res.total_seconds()
        ax.xaxis.set_major_locator(get_locator_from_offset(offset=self._resolution_x_ticks))
        date_formatter = mdates.DateFormatter(fmt=get_datetime_format(), tz=self.begin.tz)
        ax.xaxis.set_major_formatter(date_formatter)
        ax.set_xlim(self.begin, self.end)
        ax.grid(linestyle="--", linewidth=0.2, axis="both", zorder=1)
        ax.set_ylabel(f"Detections\n(resolution: {time_bin_str} - bin size: {resolution_bin_str})")

        return ax


    def copy_ax(self, source_ax: plt.Axes, target_ax: plt.Axes) -> None:
        """Duplicate axis configuration."""
        # x-axis properties
        target_ax.set_xticks(source_ax.get_xticks())
        target_ax.set_xlim(source_ax.get_xlim())
        target_ax.xaxis.set_major_locator(source_ax.xaxis.get_major_locator())
        target_ax.xaxis.set_major_formatter(source_ax.xaxis.get_major_formatter())

        # labels
        target_ax.set_ylabel(source_ax.get_ylabel())
        target_ax.set_xlabel(source_ax.get_xlabel())

        # grid properties
        for dim in ["x", "y"]:
            gridlines = getattr(source_ax, f"{dim}axis").get_gridlines()

            if gridlines:
                line = gridlines[0]
                visible = line.get_visible()
                linestyle = line.get_linestyle()
                linewidth = line.get_linewidth()
                color = line.get_color()
                zorder = line.get_zorder()

                getattr(target_ax, f"{dim}axis").grid(
                    visible=visible,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=color,
                    zorder=zorder,
                )

    def overview(self) -> None:
        """Overview of an APLOSE formatted DataFrame."""

        logging.basicConfig(level=logging.INFO, format="%(message)s")

        summary_label = (
            self.df.groupby("annotation")["annotator"]  # noqa: PD010
            .apply(Counter)
            .unstack(fill_value=0)
        )

        summary_annotator = (
            self.df.groupby("annotator")["annotation"]  # noqa: PD010
            .apply(Counter)
            .unstack(fill_value=0)
        )

        msg = f"- Overview of the detections -\n\n {summary_label}"
        logging.info(msg)

        fig, axs = plt.subplots(2, 1)
        axs[0] = summary_label.plot(kind="bar", ax=axs[0], edgecolor="black", linewidth=1)
        axs[1] = summary_annotator.plot(
            kind="bar", ax=axs[1], edgecolor="black", linewidth=1
        )

        for a in axs:
            a.legend(loc="best", frameon=1, framealpha=0.6)
            a.tick_params(axis="both", rotation=0)
            a.set_ylabel("Number of annotated calls")
            a.yaxis.grid(color="gray", linestyle="--")
            a.set_axisbelow(True)

        # labels
        axs[0].set_xlabel("Labels")
        axs[1].set_xlabel("Annotator")

        # titles
        axs[0].set_title("Number of annotations per label")
        axs[1].set_title("Number of annotations per annotator")