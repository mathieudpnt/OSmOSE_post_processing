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
    """Represents recording effort over time, aggregated into bins."""

    counts: Series
    timebin_origin: Timedelta

    @classmethod
    def from_path(
        cls,
        config,
        *,
        bin_size: Timedelta | BaseOffset,
    ) -> RecordingPeriod:
        """Vectorized creation of recording coverage from CSV with start/end datetimes.

        This method reads a CSV with columns:
        - 'start_recording'
        - 'end_recording'
        - 'start_deployment'
        - 'end_deployment'

        It computes the **effective recording interval** as the intersection between
        recording and deployment periods, builds a fine-grained timeline at
        `timebin_origin` resolution, and aggregates effort into `bin_size` bins.

        Parameters
        ----------
        config
            Configuration object containing at least:
            - `timestamp_file`: path to CSV
            - `timebin_origin`: Timedelta resolution of detections
        bin_size : Timedelta or BaseOffset
            Size of the aggregation bin (e.g., pd.Timedelta("1H") or "1D").

        Returns
        -------
        RecordingPeriod
            Object containing `counts` (Series indexed by IntervalIndex) and
            `timebin_origin`.

        """
        # 1. Read CSV and parse datetime columns
        timestamp_file = config.timestamp_file
        delim = find_delimiter(timestamp_file)
        df = pd.read_csv(
            config.timestamp_file,
            parse_dates=[
                "start_recording",
                "end_recording",
                "start_deployment",
                "end_deployment",
            ],
            delimiter=delim,
        )

        if df.empty:
            raise ValueError("CSV is empty.")

        # Ensure all required columns are present
        required_columns = {
            "start_recording",
            "end_recording",
            "start_deployment",
            "end_deployment",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(sorted(missing))}",
            )

        # 2. Normalize timezones: convert to UTC, then remove tz info (naive)
        for col in [
            "start_recording",
            "end_recording",
            "start_deployment",
            "end_deployment",
        ]:
            df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert(None)

        # 3. Compute effective recording intervals (intersection)
        df["effective_start_recording"] = df[
            ["start_recording", "start_deployment"]
        ].max(axis=1)

        df["effective_end_recording"] = df[
            ["end_recording", "end_deployment"]
        ].min(axis=1)

        # Remove rows with no actual recording interval
        df = df.loc[df["effective_start_recording"] < df["effective_end_recording"]].copy()

        if df.empty:
            raise ValueError("No valid recording intervals after deployment intersection.")

        # 4. Build fine-grained timeline at `timebin_origin` resolution
        origin = config.timebin_origin
        time_index = pd.date_range(
            start=df["effective_start_recording"].min(),
            end=df["effective_end_recording"].max(),
            freq=origin,
        )

        # Initialize effort vector (0 = no recording, 1 = recording)
        # Compare each timestamp to all intervals in a vectorized manner
        effort = pd.Series(0, index=time_index)

        # 5. Vectorized interval coverage
        tvals = time_index.values[:, None]
        start_vals = df["effective_start_recording"].values
        end_vals = df["effective_end_recording"].values

        # Boolean matrix: True if timestamp is within any recording interval
        covered = (tvals >= start_vals) & (tvals < end_vals)
        effort[:] = covered.any(axis=1).astype(int)

        # 6. Aggregate effort into user-defined bin_size
        counts = effort.resample(bin_size).sum()

        # Replace index with IntervalIndex for downstream compatibility
        counts.index = pd.interval_range(
            start=counts.index[0],
            periods=len(counts),
            freq=bin_size,
            closed="left",
        )

        return cls(counts=counts, timebin_origin=origin)
