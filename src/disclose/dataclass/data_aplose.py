"""`data_aplose` module provides the `DataAplose` class.

DataAplose class is used for handling, analyzing, and visualizing
APLOSE-formatted detection data. It includes utilities to bin detections,
plot time-based distributions, and manage metadata such as annotators and labels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pandas import (
    DataFrame,
    Series,
    Timedelta,
    Timestamp,
    concat,
    date_range,
    read_csv,
    NA,
    cut,
)
from pandas.tseries import offsets

from disclose.dataclass.data_aplose_config import DataAploseConfig
from disclose.utils.core import get_count, build_time_vector
from disclose.utils.filtering import (
    get_annotators,
    get_dataset,
    get_labels,
    get_timezone,
    load_detections,
    _build_detection_vector,
)
from disclose.dataclass.recording_period import RecordingPeriod
from disclose.utils.metric import detection_perf
from disclose.utils.visualisation import (
    heatmap,
    histo,
    overview,
    plot_annotator_agreement,
    scatter,
    timeline,
)

if TYPE_CHECKING:
    from datetime import tzinfo

    from pandas.tseries.offsets import BaseOffset


default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _get_locator_from_offset(
    offset: int | Timedelta | BaseOffset,
) -> mdates.DateLocator:
    """Map a pandas' offset object to the appropriate matplotlib DateLocator."""
    if isinstance(offset, int):
        return mdates.SecondLocator(interval=offset)

    if isinstance(offset, Timedelta):
        total_seconds = int(offset.total_seconds())
        if total_seconds % 3600 == 0:
            return mdates.HourLocator(interval=total_seconds // 3600)
        if total_seconds % 60 == 0:
            return mdates.MinuteLocator(interval=total_seconds // 60)
        return mdates.SecondLocator(interval=total_seconds)

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


APLOSE_COLUMNS = [
    "dataset",
    "filename",
    "start_datetime",
    "end_datetime",
    "start_time",
    "end_time",
    "label",
    "annotator",
]


def empty_dataframe() -> DataFrame:
    return DataFrame(columns=Series(APLOSE_COLUMNS))


@dataclass
class DataAplose:
    """A class to handle APLOSE formatted data."""

    df: DataFrame = field(default_factory=empty_dataframe)
    config: DataAploseConfig = field(default_factory=DataAploseConfig.empty)

    lat: float | None = None
    lon: float | None = None

    _start_datetime: Timestamp | None = field(default=None, init=False, repr=False)
    _end_datetime: Timestamp | None = field(default=None, init=False, repr=False)
    _annotator: list[str] | None = field(default=None, init=False, repr=False)
    _label: list[str] | None = field(default=None, init=False, repr=False)
    _dataset: list[str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._start_datetime: Timestamp | None = None
        self._end_datetime: Timestamp | None = None
        self._annotator: str | list[str] | None = None
        self._label: str | list[str] | None = None
        self._dataset: str | list[str] | None = None

    def __str__(self) -> str:
        """Return string representation of DataAplose object."""
        return (
            f"start_datetime: {self.start_datetime}\n"
            f"end_datetime: {self.end_datetime}\n"
            f"annotator: {self.annotator}\n"
            f"label: {self.label}\n"
            f"dataset: {self.dataset}"
        )

    def __repr__(self) -> str:
        """Return string representation of DataAplose object."""
        return self.__str__()

    @property
    def annotator(self) -> str | list[str] | None:
        if self.df.empty:
            return None
        return get_annotators(self.df)

    @annotator.setter
    def annotator(self, value: list[str]) -> None:
        self._annotator = value

    @property
    def label(self) -> str | list[str] | None:
        if self.df.empty:
            return None
        return get_labels(self.df)

    @label.setter
    def label(self, value: str | list[str]) -> None:
        self._label = value

    @property
    def dataset(self) -> str | list[str] | None:
        if self.df.empty:
            return None
        return get_dataset(self.df)

    @dataset.setter
    def dataset(self, value: str | list[str]) -> None:
        self._dataset = value

    @property
    def start_datetime(self):
        if self._start_datetime is not None:
            return self._start_datetime
        if self.config.start_datetime:
            return min(self.config.start_datetime, self.df["start_datetime"].min())
        if not self.df.empty:
            return self.df["start_datetime"].min()
        return None

    @start_datetime.setter
    def start_datetime(self, value: Timestamp | None = None) -> None:
        self._start_datetime = value

    @property
    def end_datetime(self):
        if self._end_datetime is not None:
            return self._end_datetime
        if self.config.end_datetime:
            return max(self.config.end_datetime, self.df["end_datetime"].max())
        if not self.df.empty:
            return self.df["end_datetime"].max()
        return None

    @end_datetime.setter
    def end_datetime(self, value: Timestamp | None = None) -> None:
        self._end_datetime = value

    @property
    def coordinates(self) -> tuple[float | None, float | None]:
        """Coordinates of the audio data."""
        return self.lat, self.lon

    @coordinates.setter
    def coordinates(self, value: tuple[float, float]) -> None:
        if not isinstance(value, tuple) or len(value) != 2:  # noqa: PLR2004
            msg = "Coordinates must be a tuple of two floats: (lat, lon)."
            raise ValueError(msg)
        self.lat, self.lon = value

    @classmethod
    def from_config(cls, config: DataAploseConfig):
        df = load_detections(config)

        obj = cls(df=df, config=config)

        obj.reshape(
            config.start_datetime,
            config.end_datetime,
        )

        return obj

    @classmethod
    def from_dict(
        cls,
        config: dict | list[dict],
        *,
        merge: bool = True,
    ) -> DataAplose | list[DataAplose]:
        configs = DataAploseConfig.from_dict(config)

        if isinstance(configs, DataAploseConfig):
            return cls.from_config(configs)

        datasets = [cls.from_config(conf) for conf in configs]

        if merge:
            return cls.merge(datasets)

        return datasets

    @classmethod
    def merge(
        cls,
        datasets: list[DataAplose],
    ) -> DataAplose:
        if not datasets:
            return cls()

        df = (
            concat(
                [d.df for d in datasets],
                ignore_index=True,
            )
            .sort_values([
                "start_datetime",
                "end_datetime",
                "annotator",
                "label",
            ])
            .reset_index(drop=True)
        )

        config = DataAploseConfig.merge([d.config for d in datasets])

        obj = cls(df=df, config=config)

        if isinstance(get_timezone(df), list):
            obj.change_tz("UTC")
            msg = (
                "Several timezones found in DataFrame,"
                " all timestamps are converted to UTC."
            )
            logging.info(msg)

        return obj

    def reshape(
        self,
        start_datetime: Timestamp | None = None,
        end_datetime: Timestamp | None = None,
    ) -> DataAplose:
        """Reshape the DataAplose with a new beginning and/or end."""
        if not any([start_datetime, end_datetime]):
            msg = "No begin/end timestamps provided for reshape of DataAplose instance."
            logging.debug(msg)
            return self

        if start_datetime is None:
            start_datetime = self.df["start_datetime"].min()

        if end_datetime is None:
            end_datetime = self.df["end_datetime"].max()

        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

        tz = get_timezone(self.df)

        if start_datetime:
            self.start_datetime = start_datetime
            if not start_datetime.tz:
                self.start_datetime = start_datetime.tz_localize(tz)

        if end_datetime:
            self.end_datetime = end_datetime
            if not end_datetime.tz:
                self.end_datetime = end_datetime.tz_localize(tz)

        if self.start_datetime >= self.end_datetime:
            msg = "Begin timestamp is not anterior than end timestamp."
            raise ValueError(msg)

        self.df = self.df[
            (self.df["start_datetime"] >= self.start_datetime)
            & (self.df["end_datetime"] <= self.end_datetime)
        ]

        if self.df.empty:
            return self

        self.dataset = get_dataset(self.df)
        self.label = get_labels(self.df)
        self.annotator = get_annotators(self.df)

        return self

    def change_tz(self, tz: str | tzinfo) -> None:
        """Change the timezone of a DataAplose instance.

        Examples
        --------
        >>> import pytz
        >>> data = DataAplose(...)
        >>> data.change_tz(pytz.timezone("Etc/GMT-2"))

        >>> data = DataAplose(...)
        >>> data.change_tz("UTC")

        >>> data = DataAplose(...)
        >>> data.change_tz("UTC+02:00")

        """
        self.df["start_datetime"] = [
            elem.tz_convert(tz) for elem in self.df["start_datetime"]
        ]
        self.df["end_datetime"] = [
            elem.tz_convert(tz) for elem in self.df["end_datetime"]
        ]

        self.start_datetime = self.start_datetime.tz_convert(tz)
        self.end_datetime = self.end_datetime.tz_convert(tz)

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

        Returns
        -------
        The filtered DataFrame.

        Raises
        ------
        ValueError
            If annotator or label are not valid or if the filtered Dataframe is empty.

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
            if ant not in self.annotator:
                msg = f'Annotator "{ant}" not in APLOSE DataFrame'
                raise ValueError(msg)
            if lbl not in self.label:
                msg = f'Label "{lbl}" not in APLOSE DataFrame'
                raise ValueError(msg)
            if self.df[(self.df["annotator"] == ant) & (self.df["label"] == lbl)].empty:
                msg = (
                    f"DataFrame with annotator '{ant}' / label '{lbl}'"
                    f" contains no detection."
                )
                raise ValueError(msg)
        config = list(zip(annotator, label, strict=False))
        return self.df[
            self.df[["annotator", "label"]].apply(tuple, axis=1).isin(config)
        ].reset_index(drop=True)

    def set_ax(
        self,
        ax: plt.Axes,
        x_ticks_res: Timedelta | offsets.BaseOffset,
        date_format: str,
    ) -> plt.Axes:
        """Configure a Matplotlib axis for time-based plot.

        Sets up x-axis with appropriate limits, tick spacing,
        formatting, and grid styling.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object to configure.
        x_ticks_res : Timedelta | offsets.BaseOffset
            Resolution of the x-axis major ticks.
        date_format : str
            Date format string for x-axis tick labels (e.g., "%b", "%Y-%m-%d %H:%M").

        Returns
        -------
        matplotlib.axes.Axes
            The configured Axes object, ready for plotting.

        """
        ax.xaxis.set_major_locator(
            _get_locator_from_offset(offset=x_ticks_res),
        )
        date_formatter = mdates.DateFormatter(
            fmt=date_format, tz=self.start_datetime.tz
        )
        ax.xaxis.set_major_formatter(date_formatter)
        ax.grid(linestyle="--", linewidth=0.2, axis="both", zorder=1)

        return ax

    def overview(self, annotator: str | list[str] | None = None) -> None:
        """Overview of an APLOSE formatted DataFrame."""
        if not annotator:
            annotator = self.annotator
        overview(self.df, annotator)

    def detection_perf(
        self,
        annotator: tuple[str, str] | list[str],
        label: tuple[str, str] | list[str],
    ) -> tuple[float, float, float]:
        """Compute performance metrics for detection.

        Precision and recall are computed in regard to a reference annotator/label pair.

        Parameters
        ----------
        annotator: [str, str]
            List of the two annotators to compare.
            The first annotator is chosen as a reference.
        label: [str, str]
            List of the two labels to compare.
            The first label is chosen as a reference.

        Returns
        -------
        precision: float
        recall: float
        f_score: float

        """
        df_filtered = self.filter_df(
            annotator,
            label,
        )
        if isinstance(annotator, str):
            annotator = [annotator]
        if isinstance(label, str):
            label = [label]
        ref = (annotator[0], label[0])

        if len(set(df_filtered["end_time"])) > 1:
            msg = "Multiple time bins detected in DataFrame."
            raise ValueError(msg)
        timebin = Timedelta(df_filtered["end_time"].iloc[0], "s")

        if self.config.recording_file:
            effort = RecordingPeriod.from_config(config=self.config, bin_size=timebin)

        return detection_perf(
            df=df_filtered,
            ref=ref,
            time=date_range(self.start_datetime, self.end_datetime, freq=timebin),
            effort=effort,
        )

    def plot(
        self,
        mode: str,
        ax: plt.Axes,
        *,
        annotator: str | list[str],
        label: str | list[str],
        **kwargs: bool | Timedelta | BaseOffset | str | list[str] | RecordingPeriod,
    ) -> None:
        """Plot filtered data using the specified mode.

        Supports multiple plot types depending on the mode:
          - "histogram": Plot a histogram of data.
          - "scatter" / "heatmap": Map hourly detections on a timeline.
          - "agreement": Plot inter-annotator agreement regression.
          - "timeline": Plot a timeline of data.

        Parameters
        ----------
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
            - bin_size: Timedelta | BaseOffset
                Bin size for the histogram.
            - effort: bool
                The timestamp intervals corresponding to the observation effort.
                If provided by the `recording_file` argument, data will be normalized by observation effort.

        """
        df_filtered = self.filter_df(
            annotator,
            label,
        )

        dates = date_range(self.start_datetime, self.end_datetime)
        bin_size = kwargs.get("bin_size")
        legend = kwargs.get("legend", True)
        color = kwargs.get("color")
        season = kwargs.get("season")
        effort = kwargs.get("effort", False)
        if effort:
            effort = RecordingPeriod.from_config(config=self.config, bin_size=bin_size)
        show_rise_set = kwargs.get("show_rise_set", True)

        if mode == "histogram":
            ax.set_xlim(self.start_datetime, self.end_datetime)
            if not bin_size:
                msg = "'bin_size' missing for histogram plot."
                raise ValueError(msg)
            df_counts = get_count(df_filtered, bin_size)
            detection_size = Timedelta(max(df_filtered["end_time"]), "s")
            histo(
                df=df_counts,
                ax=ax,
                bin_size=bin_size,
                time_bin=detection_size,
                legend=legend,
                color=color,
                season=season,
                effort=effort,
                coordinates=(self.lat, self.lon),
            )
            return

        if mode == "heatmap":
            ax.set_xlim(self.start_datetime, self.end_datetime)
            heatmap(
                df=df_filtered,
                ax=ax,
                bin_size=bin_size,
                time_range=dates,
                show_rise_set=show_rise_set,
                season=season,
                effort=effort,
                coordinates=self.coordinates,
            )
            return

        if mode == "scatter":
            ax.set_xlim(self.start_datetime, self.end_datetime)
            scatter(
                df=df_filtered,
                ax=ax,
                time_range=dates,
                show_rise_set=show_rise_set,
                season=season,
                coordinates=self.coordinates,
                effort=effort,
            )
            return

        if mode == "agreement":
            if not bin_size:
                msg = "'bin_size' missing for agreement plot."
                raise ValueError(msg)
            df_counts = get_count(df_filtered, bin_size)
            plot_annotator_agreement(df=df_counts, bin_size=bin_size, ax=ax)
            return

        if mode == "timeline":
            ax.set_xlim(self.start_datetime, self.end_datetime)
            color = kwargs.get("color")
            df_filtered = self.filter_df(
                annotator,
                label,
            )
            timeline(df=df_filtered, ax=ax, color=color)
            return

        msg = f"Unsupported plot mode: {mode}"
        raise ValueError(msg)

    def add_dpm(self) -> None:
        """Add the detection per minute `DPM` column to DataFrame."""
        df_dpm = read_csv(
            self.config.detection_file, parse_dates=["start_datetime", "end_datetime"]
        )
        self.df["dpm_count"] = NA
        ts_detect_beg = df_dpm["start_datetime"].to_list()
        ts_detect_end = df_dpm["end_datetime"].to_list()

        annotators = (
            [self.annotator] if isinstance(self.annotator, str) else self.annotator
        )

        for ann in annotators:
            if ann.lower() in {"fpod", "cpod"}:
                df_sel = self.df[(self.df["annotator"] == ann)]
                time_vector = build_time_vector(df_sel, self.config.timebin_new)
                bins = list(time_vector) + [time_vector[-1] + self.config.timebin_new]

                detect_vec = _build_detection_vector(
                    time_vector, ts_detect_beg, ts_detect_end
                )
                counts = (
                    cut(ts_detect_beg, bins=bins, right=False)
                    .value_counts()
                    .sort_index()
                )

                dpm_values = [
                    counts.iloc[i] for i, detected in enumerate(detect_vec) if detected
                ]

                self.df.loc[df_sel.index, "dpm_count"] = dpm_values
