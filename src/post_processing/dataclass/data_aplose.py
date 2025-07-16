from __future__ import annotations

import matplotlib.dates as mdates
from matplotlib.dates import MO
import matplotlib.pyplot as plt
from numpy import ceil, ndarray
from pandas import DataFrame, Series, date_range, Timedelta
from pandas.tseries import offsets
import logging
from collections import Counter
from itertools import cycle

from post_processing.def_func import get_datetime_format, get_duration, t_rounder, get_coordinates
from post_processing.premiers_resultats_utils import (
    get_resolution_str,
    select_reference,
    get_sun_times
)

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def _get_locator_from_offset(offset) -> mdates.DateLocator:
    """Map a pandas offset object to the appropriate matplotlib DateLocator."""
    if isinstance(offset, int):
        return mdates.SecondLocator(interval=offset)

    offset_to_locator = {
        (
            offsets.MonthEnd,
            offsets.MonthBegin,
            offsets.BusinessMonthEnd,
            offsets.BusinessMonthBegin,
        ): lambda offset: mdates.MonthLocator(interval=offset.n),
        (offsets.Week,): lambda offset: mdates.WeekdayLocator(byweekday=mdates.MO, interval=offset.n),
        (offsets.Day,): lambda offset: mdates.DayLocator(interval=offset.n),
        (offsets.Hour,): lambda offset: mdates.HourLocator(interval=offset.n),
        (offsets.Minute,): lambda offset: mdates.MinuteLocator(interval=offset.n),
    }

    for offset_classes, locator_fn in offset_to_locator.items():
        if isinstance(offset, offset_classes):
            return locator_fn(offset)

    raise ValueError(f"Unsupported offset type: {type(offset)}")

def _wrap_xtick_labels(ax: plt.Axes, max_chars=10):
    """Wrap x-axis tick labels at max_chars per line without breaking words unless necessary."""
    def wrap_text(text):
        lines = []
        while len(text) > max_chars:
            break_index = text.rfind(' ', 0, max_chars + 1)
            if break_index == -1:
                break_index = max_chars  # force break
            lines.append(text[:break_index])
            text = text[break_index:].lstrip()
        lines.append(text)  # remaining part
        return '\n'.join(lines)

    new_labels = [wrap_text(label.get_text()) for label in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels, rotation=0)


class DataAplose:
    """A class to handle APLOSE formatted data."""

    def __init__(self, df: DataFrame) -> None:
        """Initialize a DataAplose object from a DataFrame.

        Parameters
        ----------
        df: DataFrame
            APLOSE formatted DataFrame.
        """
        self.df = df
        self.annotators = list(set(self.df["annotator"]))
        self.labels = list(set(self.df["annotation"]))
        self.begin = min(self.df["start_datetime"])
        self.end = max(self.df["end_datetime"])
        self.dataset = list(set(self.df["dataset"]))
        self.lat = None
        self.lon = None
        self._time_bin = None
        self._resolution_bin = None
        self._resolution_x_ticks = None


    def __str__(self) -> str:
        """Return string representation of DataAplose object."""
        return (
            f"annotators: {self.annotators}\n"
            f"labels: {self.labels}\n"
            f"begin: {self.begin}\n"
            f"end: {self.end}\n"
            f"dataset: {', '.join(self.dataset)}"
        )


    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the audio data."""
        return self.df.shape


    @property
    def lat(self) -> float:
        """Shape of the audio data."""
        return self._lat


    @lat.setter
    def lat(self, value):
        self._lat = value


    @property
    def lon(self) -> float:
        """Shape of the audio data."""
        return self._lon


    @lon.setter
    def lon(self, value):
        self._lon = value


    @property
    def coordinates(self) -> tuple[float, float]:
        """Shape of the audio data."""
        return (self.lat, self.lon)


    def __getitem__(self, item: int) -> Series:
        """Return the row from the underlying DataFrame."""
        return self.df.iloc[item]


    def plot(self, annotator: str | [str], label: str | [str], ax: plt.Axes, color: str = None, legend: bool = True) -> None:
        """Seasonality plot.

        Parameters
        ----------
        annotator: str | [str]
            Annotator or list of annotators used to plot data.
        label: str | [str]
            Label or list of labels used to plot data.
        ax : matplotlib.axes.Axes
            Matplotlib Axes object on which to draw the histogram.
        color : str or list of str, optional
            Color or list of colors for the histogram bars. If not provided, default colors will be used.
        legend : bool, default=True
            Whether to display the legend on the plot.
        """
        if type(label) == str:
            label = [label] if type(annotator) == str else [label] * len(annotator)
            if legend: legend = annotator
        if type(annotator) == str:
            annotator = [annotator] if type(label) == str else [annotator] * len(label)
            if legend: legend = label
        if not len(annotator) == len(label):
            msg = f"Length of annotator ({len(annotator)}) and label ({len(label)}) must match."
            raise ValueError(msg)

        for ant, lbl in zip(annotator, label):
            if ant not in self.annotators:
                msg = f'Annotator "{ant}" not in APLOSE DataFrame'
                raise ValueError(msg)
            if lbl not in self.labels:
                msg = f'Label "{lbl}" not in APLOSE DataFrame'
                raise ValueError(msg)
            if self.df[(self.df["is_box"] == 0) & (self.df["annotator"] == ant) & (self.df["annotation"] == lbl)].empty:
                msg = f"DataFrame with annotator '{ant}' / label '{lbl}' contains no weak detection, consider reshaping it first"
                raise ValueError(msg)

        datetime_list = [
            self.df[(self.df["annotator"] == ant) & (self.df["annotation"] == lbl)]["start_datetime"]
            for ant, lbl in zip(annotator, label)
        ]

        if type(self._resolution_bin) == int:
            bins = date_range(
                start=t_rounder(t=self.begin, res=self._resolution_bin),
                end=t_rounder(t=self.end, res=self._resolution_bin),
                freq=str(self._resolution_bin) + "s",
            )
        else:
            bins = date_range(
                start=self.begin.normalize(),
                end=self.end.normalize(),
                freq=str(self._resolution_bin.n) + self._resolution_bin.name,
            )

        color = color if color else [c for _, c in zip(range(len(datetime_list)), cycle(default_colors))]

        val1, _, _ = ax.hist(
            datetime_list,
            bins=bins,
            edgecolor="black",
            zorder=2,
            histtype="bar",
            stacked=False,
            color=color,
        )

        if val1.ndim > 1 and legend:
            ax.legend(loc="upper right", labels=legend)

        time_bin_str = get_resolution_str(self._time_bin)
        resolution_bin_str = str(self._resolution_bin.n) + self._resolution_bin.name if not type(self._resolution_bin) == int else get_resolution_str(self._resolution_bin)

        ax.set_ylabel(f"Detections\n(resolution: {time_bin_str} - bin size: {resolution_bin_str})")

        ax.set_ylim(0, int(ceil(1.05 * ndarray.max(val1))))
        ax.set_yticks(range(0, int(ceil(ndarray.max(val1))) + 1, max(1, int(ceil(ndarray.max(val1))) // 4)))
        ax.title.set_text(f"annotator: {', '.join(set(annotator))}\nlabel: {', '.join(set(label))}")


    def set_ax(self, ax: plt.Axes, bin_size: Timedelta | offsets.BaseOffset = None, xticks_res: Timedelta | offsets.BaseOffset = None, date_format: str = None) -> plt.Axes:
        """Configure a Matplotlib axis for time-based plot.

        Sets up the x-axis with appropriate limits, tick spacing, formatting, and grid styling.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object to configure.
        bin_size : Timedelta | offsets.BaseOffset, optional
            Resolution of the histogram bins. If a `Timedelta`, it is converted to seconds.
        xticks_res : Timedelta | offsets.BaseOffset, optional
            Resolution of the x-axis major ticks.
            If not provided, user will be prompted.
        date_format : str, optional
            Date format string for x-axis tick labels (e.g., "%b", "%Y-%m-%d %H:%M").
            If not provided, user will be prompted.

        Returns
        -------
        matplotlib.axes.Axes
            The configured Axes object, ready for plotting.
        """
        self._time_bin = int(
            select_reference(self.df[self.df["is_box"] == 0]["end_time"], "time bin"),
        )

        self._resolution_bin = int(bin_size.total_seconds()) if type(bin_size) == Timedelta else bin_size

        if not xticks_res:
            self._resolution_x_ticks = get_duration(
                msg="Enter x-axis tick resolution",
                default="2h",
            )
        self._resolution_x_ticks = xticks_res

        ax.xaxis.set_major_locator(_get_locator_from_offset(offset=self._resolution_x_ticks))
        date_format = date_format if date_format else get_datetime_format()
        date_formatter = mdates.DateFormatter(fmt=date_format, tz=self.begin.tz)
        ax.xaxis.set_major_formatter(date_formatter)
        ax.set_xlim(self.begin, self.end)
        ax.grid(linestyle="--", linewidth=0.2, axis="both", zorder=1)

        return ax


    def copy_ax(self, source_ax: plt.Axes, target_ax: plt.Axes) -> None:
        """Duplicate axis configuration.

        Attributes of source_ax are copied to target_ax

        Parameters
        ----------
        source_ax : matplotlib.axes.Axes
            The Axes object to duplicate.
        target_ax : matplotlib.axes.Axes
            The Axes object to configure.
        """
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
        axs[0] = summary_label.plot(kind="bar", ax=axs[0], edgecolor="black", linewidth=0.5)
        axs[1] = summary_annotator.plot(
            kind="bar", ax=axs[1], edgecolor="black", linewidth=0.5
        )

        for a in axs:
            a.legend(loc="center left", frameon=1, framealpha=0.6, bbox_to_anchor=(1.01, 0.5))
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

    def scatter_detections(self, ax: plt.Axes, annotator: str, label: str, *, show_rise_set: bool = True) -> None:
        """Scatter plot of the detections from an APLOSE formatted DataFrame.

        Additionally, sunrise and sunset lines can be plotted
        if show_rise_set is set to True (default value)

        Parameters
        ----------
        annotator: str
            The selected annotator.
        label: str
            The selected label.
        ax: matplotlib.axes.Axes
            Ax to apply the scatter plot to.
        show_rise_set : bool, default True
            display the sunrise and sunset lines

        """
        if not self.lat or not self.lon:
            self.lat, self.lon = get_coordinates()

        if annotator not in self.annotators:
            msg = f'Annotator "{annotator}" not in APLOSE DataFrame'
            raise ValueError(msg)
        if label not in self.labels:
            msg = f'Label "{label}" not in APLOSE DataFrame'
            raise ValueError(msg)
        if self.df[(self.df["annotator"] == annotator) & (self.df["annotation"] == label)].empty:
            msg = f"DataFrame with annotator '{annotator}' / label '{label}' contains no detection."
            raise ValueError(msg)

        datetime_list = self.df[(self.df["annotator"] == annotator) & (self.df["annotation"] == label)]["start_datetime"]
        begin = min(datetime_list) - Timedelta('1d')
        end = max(datetime_list) + Timedelta(self._time_bin, 's') + Timedelta('1d')

        # compute sunrise and sunset decimal hour at the dataset location
        hour_sunrise, hour_sunset, _, _, _, _ = get_sun_times(
            start=begin,
            stop=end,
            lat=self.lat,
            lon=self.lon,
        )

        dates = date_range(begin.date(), end.date(), freq="D")

        # decimal hours of detections
        hour_det = [
            ts.hour + ts.minute / 60 + ts.second / 3600 for ts in datetime_list
        ]

        # plot
        ax.scatter(
            datetime_list,
            hour_det,
            marker="x",
            linewidths=1,
            color="silver",
        )
        if show_rise_set:
            ax.plot(dates, hour_sunrise, color="darkorange", label="Sunrise")
            ax.plot(dates, hour_sunset, color="royalblue", label="Sunset")
            ax.legend(loc="center left", frameon=1, framealpha=0.6, bbox_to_anchor=(1.01, 0.5))


        ax.set_xlim(begin, end)
        ax.set_ylim(0, 24)
        ax.set_yticks(range(0, 24, 2))

        ax.grid(color="k", linestyle="-", linewidth=0.2)
        ax.set_ylabel("Hour")
        ax.set_xlabel("Date")

        ax.set_title(
            f"Time of detections within each day\n"
            f"annotator: {annotator}\nlabel: {label}\n"
            f"timezone: {begin.tz}",
        )
