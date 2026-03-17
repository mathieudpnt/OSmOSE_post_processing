import pytest
from pandas import DataFrame, Timedelta

from post_processing.dataclass.detection import Detection


@pytest.fixture
def sample_detection(sample_df: DataFrame) -> list[Detection]:
    return Detection.from_df(sample_df)


def test_valid_detection_creates_successfully(
    sample_detection: list[Detection],
    sample_df: DataFrame,
) -> None:
    det = sample_detection[0]

    assert len(sample_detection) == len(sample_df)
    assert isinstance(det, Detection)
    assert det.dataset == sample_df.iloc[0]["dataset"]
    assert det.filename == sample_df.iloc[0]["filename"]
    assert det.start_datetime == sample_df.iloc[0]["start_datetime"]
    assert det.start_time == sample_df.iloc[0]["start_time"]
    assert det.end_datetime == sample_df.iloc[0]["end_datetime"]
    assert det.end_time == sample_df.iloc[0]["end_time"]
    assert det.annotation == sample_df.iloc[0]["annotation"]
    assert det.annotator == sample_df.iloc[0]["annotator"]
    assert det.type == sample_df.iloc[0]["type"]


def test_sanity_check_end_dt_anterior(sample_df: DataFrame) -> None:
    row = sample_df.iloc[0]
    row["end_datetime"] = row["start_datetime"] - Timedelta(1)
    with pytest.raises(ValueError, match="must be strictly less than end_datetime"):
        Detection.from_series(row)


def test_sanity_check_end_time_anterior(sample_df: DataFrame) -> None:
    row = sample_df.iloc[0]
    row["end_time"] = row["start_time"] - 1
    with pytest.raises(ValueError, match="must be strictly less than "):
        Detection.from_series(row)
