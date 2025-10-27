"""`recording_period` module provides `RecordingPeriod` dataclass.

RecordingPeriod class returns a Timestamp list corresponding to recording periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from osekit.config import TIMESTAMP_FORMATS_EXPORTED_FILES
from osekit.utils.timestamp_utils import strptime_from_text
from pandas import (
    Series,
    Timedelta,
    cut,
    read_csv,
)

from post_processing.utils.core_utils import (
    get_time_range_and_bin_size,
    localize_timestamps,
)
from post_processing.utils.filtering_utils import (
    find_delimiter,
)

if TYPE_CHECKING:
    from pandas.tseries.offsets import BaseOffset

    from post_processing.dataclass.detection_filter import DetectionFilter


@dataclass(frozen=True)
class RecordingPeriod:
    """A class to handle recording periods."""

    counts: Series
    timebin_origin: Timedelta

    @classmethod
    def from_path(
        cls,
        config: DetectionFilter,
        date_format: str = TIMESTAMP_FORMATS_EXPORTED_FILES,
        *,
        bin_size: Timedelta | BaseOffset,
    ) -> RecordingPeriod:
        """Return a list of Timestamps corresponding to recording periods."""
        timestamp_file = config.timestamp_file
        delim = find_delimiter(timestamp_file)
        timestamp_df = read_csv(timestamp_file, delimiter=delim)

        if "timestamp" in timestamp_df.columns:
            msg = "Parsing 'timestamp' column not implemented yet."
            raise NotImplementedError(msg)

        if "filename" in timestamp_df.columns:
            timestamps = [
                    strptime_from_text(ts, date_format)
                    for ts in timestamp_df["filename"]
                ]
            timestamps = localize_timestamps(timestamps, config.timezone)
            time_vector, bin_size = get_time_range_and_bin_size(timestamps, bin_size)

            binned = cut(timestamps, time_vector)
            max_annot = bin_size / config.timebin_origin

            return cls(counts=binned.value_counts().sort_index().clip(upper=max_annot),
                       timebin_origin=config.timebin_origin,
                       )

        msg = "Could not parse timestamps."
        raise ValueError(msg)
