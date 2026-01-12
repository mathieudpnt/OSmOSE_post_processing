import pandas as pd
from pandas.tseries import frequencies

from post_processing.dataclass.recording_period import RecordingPeriod


def test_recording_period_with_gaps(recording_planning_config):
    """RecordingPeriod correctly represents long gaps with no recording effort.

    The planning contains two recording blocks separated by ~3 weeks with no
    recording at all. Weekly aggregation must reflect:
    - weeks with full effort,
    - weeks with partial effort,
    - weeks with zero effort.
    """
    recording_period = RecordingPeriod.from_path(
        config=recording_planning_config,
        bin_size=frequencies.to_offset("1W"),
    )

    counts = recording_period.counts

    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------
    assert not counts.empty
    assert counts.index.is_interval()
    assert counts.min() >= 0

    # One week = 7 * 24 hours (origin = 1min)
    full_week_minutes = 7 * 24 * 60

    # ------------------------------------------------------------------
    # Helper: find the bin covering a given timestamp
    # ------------------------------------------------------------------
    def bin_covering(ts: pd.Timestamp) -> pd.Interval:
        for interval in counts.index:
            if interval.left <= ts < interval.right:
                return interval
        raise AssertionError(f"No bin covers timestamp {ts}")

    # ------------------------------------------------------------------
    # Week fully inside the long gap → zero effort
    # ------------------------------------------------------------------
    gap_ts = pd.Timestamp("2024-04-21")

    gap_bin = bin_covering(gap_ts)
    assert counts.loc[gap_bin] == 0

    # ------------------------------------------------------------------
    # Week fully inside recording → full effort
    # ------------------------------------------------------------------
    full_effort_ts = pd.Timestamp("2024-02-04")

    full_bin = bin_covering(full_effort_ts)
    assert counts.loc[full_bin] == full_week_minutes

    # ------------------------------------------------------------------
    # Week overlapping recording stop → partial effort
    # ------------------------------------------------------------------
    partial_ts = pd.Timestamp("2024-04-14")

    partial_bin = bin_covering(partial_ts)
    assert counts.loc[partial_bin] == 1560
