"""`recording_period` module provides the `RecordingPeriod` dataclass.

RecordingPeriod class returns a Timestamp list corresponding to recording periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from osekit.config import TIMESTAMP_FORMATS_EXPORTED_FILES
from osekit.utils.timestamp_utils import strptime_from_text
from pandas import (
    DataFrame,
    DatetimeIndex,
    Timedelta,
    concat,
    cut,
    date_range,
    read_csv,
)

from post_processing.dataclass.data_aplose import DataAplose
from post_processing.utils.def_func import (
    is_non_fixed_frequency,
)
from post_processing.utils.filtering_utils import (
    find_delimiter,
    get_timezone,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import Timestamp
    from pandas.tseries import offsets

    from post_processing.dataclass.detection_filters import DetectionFilters


@dataclass(frozen=True)
class RecordingPeriod:
    """A class to handle recording periods."""

    @classmethod
    def from_path(
        cls,
        config: tuple[list[Path], list[DetectionFilters]],
        date_format: str = TIMESTAMP_FORMATS_EXPORTED_FILES,
        *,
        dataset: str,
        bin_size: Timedelta | offsets.BaseOffset,
    ) -> tuple[DataFrame, Timedelta]:
        """Return a list of Timestamps corresponding to recording periods."""
        files, filters = config

        df = DataAplose.from_filters(config=config).df
        delim = find_delimiter(config[0][0])
        df_origin = read_csv(config[0][0], delimiter=delim)
        origin_bin = Timedelta(max(df_origin["end_time"]), "s")

        if not isinstance(bin_size, Timedelta):
            bin_str = str(bin_size.n) + bin_size.name
            if not is_non_fixed_frequency(bin_str):
                bin_size = Timedelta(bin_str)

        result = DataFrame()
        path = [fil.timestamp_file for fil in filters]
        timestamp_file = DataFrame()

        for p in path:
            delim = find_delimiter(p)
            timestamp_file = concat([timestamp_file, read_csv(p, delimiter=delim)])

        if "timestamp" in timestamp_file.columns:
            msg = "Parsing 'timestamp' column not implemented yet."
            raise NotImplementedError(msg)

        if "filename" in timestamp_file.columns:
            datasets = list(set(timestamp_file["dataset"]))
            timestamps_list = [
                [
                    strptime_from_text(ts, date_format)
                    for ts in timestamp_file[timestamp_file["dataset"] == d]["filename"]
                ]
                for d in datasets
            ]

            for i in range(len(timestamps_list)):
                # localize timestamps if naïve
                if any(
                    ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None
                    for ts in timestamps_list[i]
                ):
                    tz = get_timezone(df)
                    timestamps_list[i] = [
                        ts.tz_localize(tz) for ts in timestamps_list[i]
                    ]

            all_timestamps = [ts for sublist in timestamps_list for ts in sublist]
            time_vector = get_time_vector(all_timestamps, bin_size)

            for i in range(len(timestamps_list)):
                binned = cut(timestamps_list[i], time_vector)

                if isinstance(bin_size, Timedelta):
                    max_annot = bin_size / origin_bin
                else:
                    max_annot = Timedelta(bin_str.split("-")[0]) / origin_bin

                result[f"{datasets[i]}"] = (binned.value_counts().
                                            sort_index().
                                            clip(upper=max_annot)
                                            )

        return result[dataset], origin_bin



def get_time_vector(timestamp_list: list[Timestamp],
                    bin_size: Timedelta | offsets.BaseOffset,
                    ) -> DatetimeIndex:
    """"Return time vector given a bin size."""
    if isinstance(bin_size, Timedelta):
        return date_range(start=min(timestamp_list).floor(bin_size),
                          end=max(timestamp_list).ceil(bin_size),
                          freq=bin_size)
    return date_range(start=bin_size.rollback(min(timestamp_list)).normalize(),
                             end=bin_size.rollforward(max(timestamp_list)).normalize(),
                             freq=bin_size
                             )