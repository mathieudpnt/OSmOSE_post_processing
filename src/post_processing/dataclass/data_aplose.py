"""`data_aplose` module provides the `DataAplose` class.

DataAplose class is used for handling, analyzing, and visualizing
APLOSE-formatted annotation data. It includes utilities to bin detections,
plot time-based distributions, and manage metadata such as annotators and labels.

Features
--------
- Load and validate APLOSE-formatted annotations.
- Set up matplotlib axes with time-based formatting.
- Plot detection profiles (scatter, heatmap).
- Compute detection rates by time of day.
- Filter annotations by label and annotator.

"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pandas import DataFrame, Series, Timedelta, Timestamp
from pandas.tseries import offsets

from post_processing.dataclass.metrics_utils import detection_perf
from post_processing.dataclass.plot_utils import (
    agreement,
    histo,
    map_detection_timeline,
    overview,
)
from post_processing.def_func import (
    get_datetime_format,
    get_duration,
)
from post_processing.premiers_resultats_utils import (
    select_reference,
)

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _get_locator_from_offset(
    offset: int | Timedelta | offsets.BaseOffset,
) -> mdates.DateLocator:
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
        (offsets.Week,): lambda offset: mdates.WeekdayLocator(
            byweekday=offset.weekday,
            interval=offset.n,
        ),
        (offsets.Day,): lambda offset: mdates.DayLocator(interval=offset.n),
        (offsets.Hour,): lambda offset: mdates.HourLocator(interval=offset.n),
        (offsets.Minute,): lambda offset: mdates.MinuteLocator(interval=offset.n),
    }

    for offset_classes, locator_fn in offset_to_locator.items():
        if isinstance(offset, offset_classes):
            return locator_fn(offset)

    msg = f"Unsupported offset type: {type(offset)}"
    raise ValueError(msg)


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
        """Shape of DataFrame."""
        return self.df.shape

    @property
    def lat(self) -> float:
        """Return latitude."""
        return self._lat

    @lat.setter
    def lat(self, value: float) -> None:
        self._lat = value

    @property
    def lon(self) -> float:
        """Return longitude."""
        return self._lon

    @lon.setter
    def lon(self, value: float) -> None:
        self._lon = value

    @property
    def coordinates(self) -> tuple[float, float]:
        """Shape of the audio data."""
        return self.lat, self.lon

    def __getitem__(self, item: int) -> Series:
        """Return the row from the underlying DataFrame."""
        return self.df.iloc[item]

    def filter_df(
        self,
        annotator: str | list[str],
        label: str | list[str],
    ) -> DataFrame:
        """Filter DataFrame based on annotator and label.

        Parameters
        ----------
        annotator: str | list[str]
            The annotator or list of annotators to filter.
        label: str | list[str]
            The label or list of labels to filter.

        """
        if isinstance(label, str):
            label = [label] if isinstance(annotator, str) else [label] * len(annotator)
        if isinstance(annotator, str):
            annotator = (
                [annotator] if isinstance(label, str) else [annotator] * len(label)
            )
        if len(annotator) != len(label):
            msg = (
                f"Length of annotator ({len(annotator)}) and"
                f" label ({len(label)}) must match."
            )
            raise ValueError(msg)

        for ant, lbl in zip(annotator, label, strict=False):
            if ant not in self.annotators:
                msg = f'Annotator "{ant}" not in APLOSE DataFrame'
                raise ValueError(msg)
            if lbl not in self.labels:
                msg = f'Label "{lbl}" not in APLOSE DataFrame'
                raise ValueError(msg)
            if self.df[
                (self.df["is_box"] == 0)
                & (self.df["annotator"] == ant)
                & (self.df["annotation"] == lbl)
            ].empty:
                msg = (
                    f"DataFrame with annotator '{ant}' / label '{lbl}'"
                    f" contains no weak detection."
                )
                raise ValueError(msg)
        pairs = list(zip(annotator, label, strict=False))
        return self.df[
            self.df[["annotator", "annotation"]].apply(tuple, axis=1).isin(pairs)
        ]

    def set_ax(
        self,
        ax: plt.Axes,
        x_ticks_res: Timedelta | offsets.BaseOffset = None,
        date_format: str | None = None,
    ) -> plt.Axes:
        """Configure a Matplotlib axis for time-based plot.

        Sets up x-axis with appropriate limits, tick spacing,
        formatting, and grid styling.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object to configure.
        x_ticks_res : Timedelta | offsets.BaseOffset, optional
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

        if not x_ticks_res:
            self._resolution_x_ticks = get_duration(
                msg="Enter x-axis tick resolution",
                default="2h",
            )
        self._resolution_x_ticks = x_ticks_res

        ax.xaxis.set_major_locator(
            _get_locator_from_offset(offset=self._resolution_x_ticks),
        )
        date_format = date_format if date_format else get_datetime_format()
        date_formatter = mdates.DateFormatter(fmt=date_format, tz=self.begin.tz)
        ax.xaxis.set_major_formatter(date_formatter)
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
                line_style = line.get_linestyle()
                line_width = line.get_linewidth()
                color = line.get_color()
                z_order = line.get_zorder()

                getattr(target_ax, f"{dim}axis").grid(
                    visible=visible,
                    linestyle=line_style,
                    linewidth=line_width,
                    color=color,
                    zorder=z_order,
                )

    def overview(self) -> None:
        """Overview of an APLOSE formatted DataFrame."""
        overview(self.df)

    def detection_perf(
        self,
        annotators: [str, str],
        labels: [str, str],
        timestamps: [Timestamp] = None,
    ) -> (float, float, float):
        """Compute performances metrics for detection.

        Performances are computed with a reference annotator in
        comparison with a second annotator/detector.
        Precision and recall are computed in regard
        with a reference annotator/label pair.

        Parameters
        ----------
        annotators: [str, str]
            List of the two annotators to compare.
            First annotator is chosen as reference.
        labels: [str, str]
            List of the two labels to compare.
            First label is chosen as reference.
        timestamps: list[Timestamp], optional
            A list of Timestamps to base the computation on.

        Returns
        -------
        precision: float
        recall: float
        f_score: float

        """
        df_filtered = self.filter_df(
            annotators,
            labels,
        )
        if isinstance(annotators, str):
            annotators = [annotators]
        if isinstance(labels, str):
            labels = [labels]
        ref = (annotators[0], labels[0])
        return detection_perf(
            df=df_filtered,
            ref=ref,
            timestamps=timestamps,
        )

    def plot(
        self,
        mode: str,
        ax: plt.Axes,
        *,
        annotator: str | list[str],
        label: str | list[str],
        **kwargs: bool | Timedelta | offsets.BaseOffset | str | list[str],
    ) -> None:
        """Plot filtered annotation data using the specified mode.

        Supports multiple plot types depending on the mode:
          - "histogram": Plots a histogram of annotation data.
          - "scatter" / "heatmap": Maps detections on a timeline.
          - "agreement": Plots inter-annotator agreement regression.

        Args:
            mode: str
                Type of plot to generate.
                Must be one of {"histogram", "scatter", "heatmap", "agreement"}.
            ax: plt.Axes
                Matplotlib Axes object to plot on.
            annotator: str | list[str]
                The selected annotator or list of annotators.
            label: str | list[str]
                The selected label or list of labels.
            **kwargs: Additional keyword arguments depending on the mode.
                - legend: bool
                    Whether to show the legend.
                - season: bool
                    Whether to show the season.
                - show_rise_set: bool
                    Whether to show sunrise and sunset times.
                - color: str | list[str]
                    Color(s) for the bars.
                - bin_size: Timedelta | offsets.BaseOffset
                    Bin size for the histogram.
                - effort: list[Timestamp]
                    The list of timestamps corresponding to the observation effort.
                    If provided, data will be normalized by observation effort.

        """
        df_filtered = self.filter_df(
            annotator,
            label,
        )

        if mode == "histogram":
            legend = kwargs.get("legend", True)
            color = kwargs.get("color")
            bin_size = kwargs.get("bin_size")
            season = kwargs.get("season", False)
            effort = kwargs.get("effort")

            return histo(
                df=df_filtered,
                ax=ax,
                bin_size=bin_size,
                legend=legend,
                color=color,
                season=season,
                effort=effort,
                coordinates=(self.lat, self.lon),
            )

        if mode in {"scatter", "heatmap"}:
            show_rise_set = kwargs.get("show_rise_set", True)
            season = kwargs.get("season", False)

            return map_detection_timeline(
                df=df_filtered,
                ax=ax,
                coordinates=(self.lat, self.lon),
                mode=mode,
                show_rise_set=show_rise_set,
                season=season,
            )

        if mode == "agreement":
            bin_size = kwargs.get("bin_size")
            return agreement(df=df_filtered, bin_size=bin_size, ax=ax)

        msg = f"Unsupported plot mode: {mode}"
        raise ValueError(msg)
