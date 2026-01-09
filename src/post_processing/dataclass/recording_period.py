"""`recording_period` module provides `RecordingPeriod` dataclass.

RecordingPeriod class returns a Timestamp list corresponding to recording periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from pandas import (
    Series,
    Timedelta,
)

from post_processing.utils.filtering_utils import (
    find_delimiter,
)

if TYPE_CHECKING:
    from pandas.tseries.offsets import BaseOffset


@dataclass(frozen=True)
class RecordingPeriod:
    counts: Series
    timebin_origin: Timedelta

    @classmethod
    def from_path(
        cls,
        config,
        *,
        bin_size: Timedelta | BaseOffset,
    ) -> RecordingPeriod:
        """Vectorized creation of recording coverage from CSV with start/end datetime.

        CSV must have columns 'start_recording' and 'end_recording'.
        bin_size can be a Timedelta (e.g., pd.Timedelta("1H"))
        or a pandas offset (e.g., "1D").
        """
        # 1. Read CSV and parse datetime
        timestamp_file = config.timestamp_file
        delim = find_delimiter(timestamp_file)
        df = pd.read_csv(
            config.timestamp_file,
            parse_dates=["start_recording", "end_recording"],
            delimiter=delim,
        )

        if df.empty:
            raise ValueError("CSV is empty.")

        # 2. Normalize timezones if needed
        df["start_recording"] = (
            pd.to_datetime(df["start_recording"], utc=True).dt.tz_convert(None)
        )
        df["end_recording"] = (
            pd.to_datetime(df["end_recording"], utc=True).dt.tz_convert(None)
        )

        # Build fine-grained timeline (timebin_origin resolution)
        origin = config.timebin_origin
        time_index = pd.date_range(
            start=df["start_recording"].min(),
            end=df["end_recording"].max(),
            freq=origin,
        )

        # Initialize effort vector
        effort = pd.Series(0, index=time_index)

        # Vectorized interval coverage
        tvals = time_index.values[:, None]
        start_vals = df["start_recording"].values
        end_vals = df["end_recording"].values

        covered = (tvals >= start_vals) & (tvals < end_vals)
        effort[:] = covered.any(axis=1).astype(int)

        # Aggregate effort into bin_size
        counts = effort.resample(bin_size).sum()
        counts.index = pd.interval_range(
            start=counts.index[0],
            periods=len(counts),
            freq=bin_size,
            closed="left",
        )
        return cls(counts=counts, timebin_origin=origin)
