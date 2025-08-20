import csv
from datetime import timezone
from pathlib import Path

import pytest
from pandas import DataFrame, Timedelta, Timestamp, date_range

from post_processing.utils.filtering_utils import (
    filter_by_annotator,
    filter_by_freq,
    filter_by_label,
    filter_by_score,
    filter_by_time,
    find_delimiter,
    get_annotators,
    get_dataset,
    get_labels,
    get_max_freq,
    get_max_time,
    get_timezone,
    read_dataframe,
    reshape_timebin,
)

#%% find delimiter

@pytest.mark.parametrize(("delimiter", "rows"), [
    (",", [["a", "b", "c"], ["1", "2", "3"]]),
    (";", [["x", "y", "z"], ["4", "5", "6"]]),
])

def test_find_delimiter_valid(tmp_path: Path,
                              delimiter: str,
                              rows: list[list[str]]
                              ) -> None:
    file = tmp_path / "test.csv"
    with file.open("w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(rows)

    detected = find_delimiter(file)
    assert detected == delimiter


def test_find_delimiter_invalid(tmp_path: Path) -> None:
    file = tmp_path / "invalid.csv"
    file.write_text("this is not really&csv&content")
    with pytest.raises(ValueError, match="Could not determine delimiter"):
        find_delimiter(file)


def test_find_delimiter_empty_file(tmp_path: Path) -> None:
    file = tmp_path / "empty.csv"
    file.write_text("")
    with pytest.raises(ValueError, match="Could not determine delimiter"):
        find_delimiter(file)

#%% filter utils

@pytest.fixture
def sample_df() -> DataFrame:
    return DataFrame([
        {
            "dataset": "dataset1",
            "filename": "filename1",
            "start_time": 0,
            "end_time": 300,
            "start_frequency": 100,
            "end_frequency": 500,
            "annotation": "lbl1",
            "annotator": "ann1",
            "start_datetime": Timestamp("2025-01-01 12:00:00+01:00"),
            "end_datetime": Timestamp("2025-01-01 12:05:00+01:00"),
            "is_box": 0,
            "score": 0.8,
        },
        {
            "dataset": "dataset1",
            "filename": "filename2",
            "start_time": 0,
            "end_time": 300,
            "start_frequency": 300,
            "end_frequency": 800,
            "annotation": "lbl2",
            "annotator": "ann2",
            "start_datetime": Timestamp("2025-01-01 13:00:00+01:00"),
            "end_datetime": Timestamp("2025-01-01 13:05:00+01:00"),
            "is_box": 0,
            "score": 0.4,
        },
        {
            "dataset": "dataset1",
            "filename": "filename3",
            "start_time": 0,
            "end_time": 300,
            "start_frequency": 600,
            "end_frequency": 1200,
            "annotation": "lbl1",
            "annotator": "ann1",
            "start_datetime": Timestamp("2025-01-01 14:00:00+01:00"),
            "end_datetime": Timestamp("2025-01-01 14:05:00+01:00"),
            "is_box": 0,
            "score": 0.9,
        },
    ])

# filter_by_time
def test_filter_by_time_begin(sample_df: DataFrame) -> None:
    df = filter_by_time(sample_df, begin=Timestamp("2025-01-01 13:00:00+01:00"), end=None)
    assert (df["start_datetime"] >= Timestamp("2025-01-01 13:00:00+01:00")).all()

def test_filter_by_time_end(sample_df: DataFrame) -> None:
    df = filter_by_time(sample_df, begin=None, end=Timestamp("2025-01-01 13:10:00+01:00"))
    assert (df["end_datetime"] <= Timestamp("2025-01-01 13:10:00+01:00")).all()

def test_filter_by_time_out_of_range(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="No detection found after '2026"):
        filter_by_time(sample_df, begin=Timestamp("2026-01-01 00:00:00+01:00"), end=None)

# filter_by_annotator
def test_filter_by_annotator_string(sample_df: DataFrame) -> None:
    df = filter_by_annotator(sample_df, "ann1")
    assert (df["annotator"] == "ann1").all()

def test_filter_by_annotator_list(sample_df: DataFrame) -> None:
    df = filter_by_annotator(sample_df, ["ann2"])
    assert set(df["annotator"]) == {"ann2"}

def test_filter_by_annotator_invalid(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="not present in annotator, upload aborted"):
        filter_by_annotator(sample_df, "BbJuni")

# filter_by_label
def test_filter_by_label_string(sample_df: DataFrame) -> None:
    df = filter_by_label(sample_df, "lbl1")
    assert set(df["annotation"]) == {"lbl1"}

def test_filter_by_label_list(sample_df: DataFrame) -> None:
    df = filter_by_label(sample_df, ["lbl2"])
    assert set(df["annotation"]) == {"lbl2"}

def test_filter_by_label_invalid(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="not present in label, upload aborted"):
        filter_by_label(sample_df, "hihi")

# filter_by_freq
def test_filter_by_freq_min(sample_df: DataFrame) -> None:
    df = filter_by_freq(sample_df, f_min=500, f_max=None)
    assert (df["start_frequency"] >= 500).all()

def test_filter_by_freq_max(sample_df: DataFrame) -> None:
    df = filter_by_freq(sample_df, f_min=None, f_max=600)
    assert (df["end_frequency"] <= 600).all()

def test_filter_by_freq_no_results(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="No detection found above 2000Hz"):
        filter_by_freq(sample_df, f_min=2000, f_max=None)

# filter_by_score
def test_filter_by_score_valid(sample_df: DataFrame) -> None:
    df = filter_by_score(sample_df, 0.5)
    assert (df["score"] >= 0.5).all()

def test_filter_by_score_no_results(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="No detection found with score above 1.0"):
        filter_by_score(sample_df, 1.0)

def test_filter_by_score_missing_column(sample_df: DataFrame) -> None:
    df = sample_df.drop(columns=["score"])
    with pytest.raises(ValueError, match="'score' column not present"):
        filter_by_score(df, 0.5)

def test_get_annotators(sample_df: DataFrame) -> None:
    annotators = get_annotators(sample_df)
    assert set(annotators) == {"ann1", "ann2"}

def test_get_labels(sample_df: DataFrame) -> None:
    labels = get_labels(sample_df)
    assert set(labels) == {"lbl1", "lbl2"}

def test_get_max_freq(sample_df: DataFrame) -> None:
    assert get_max_freq(sample_df) == 1_200

def test_get_max_time(sample_df: DataFrame) -> None:
    assert get_max_time(sample_df) == 300

def test_get_dataset(sample_df: DataFrame) -> None:
    datasets = get_dataset(sample_df)
    assert set(datasets) == {"dataset1"}

def test_get_timezone_single(sample_df: DataFrame) -> None:
    tz = get_timezone(sample_df)
    assert isinstance(tz, timezone)

#%% read DataFrame

def test_read_dataframe_comma_delimiter(tmp_path: Path) -> None:
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "start_datetime,end_datetime,annotation\n"
        "2025-01-01 12:00:00,2025-01-01 12:05:00,whale\n"
        "2025-01-01 13:00:00,2025-01-01 13:05:00,dolphin\n"
    )

    df = read_dataframe(csv_file)

    assert list(df.columns) == ["start_datetime", "end_datetime", "annotation"]
    assert len(df) == 2
    assert df.iloc[0]["annotation"] == "whale"

def test_read_dataframe_drop_duplicates_and_na(tmp_path: Path) -> None:
    csv_file = tmp_path / "test_duplicates.csv"
    csv_file.write_text(
        "start_datetime,end_datetime,annotation\n"
        "2025-01-01 12:00:00,2025-01-01 12:05:00,whale\n"
        "2025-01-01 12:00:00,2025-01-01 12:05:00,whale\n"  # duplicate
        "2025-01-01 13:00:00,2025-01-01 13:05:00,\n"      # NaN annotation
    )

    df = read_dataframe(csv_file)
    assert len(df) == 1

def test_read_dataframe_sorted_by_datetime(tmp_path: Path) -> None:
    csv_file = tmp_path / "test_unsorted.csv"
    csv_file.write_text(
        "start_datetime,end_datetime,annotation\n"
        "2025-01-01 14:00:00,2025-01-01 14:05:00,dolphin\n"
        "2025-01-01 12:00:00,2025-01-01 12:05:00,whale\n"
    )

    df = read_dataframe(csv_file)
    assert list(df["annotation"]) == ["whale", "dolphin"]

def test_read_dataframe_nrows(tmp_path: Path) -> None:
    csv_file = tmp_path / "test_nrows.csv"
    csv_file.write_text(
        "start_datetime,end_datetime,annotation\n"
        "2025-01-01 12:00:00,2025-01-01 12:05:00,whale\n"
        "2025-01-01 13:00:00,2025-01-01 13:05:00,dolphin\n"
    )

    df = read_dataframe(csv_file, nrows=1)
    assert len(df) == 1
    assert df.iloc[0]["annotation"] in {"whale", "dolphin"}

#%% reshape_timebin

def test_no_timebin_returns_original(sample_df: DataFrame) -> None:
    df_out = reshape_timebin(sample_df, timebin_new=None)
    assert df_out.equals(sample_df)

def test_no_timebin_original_timebin(sample_df: DataFrame) -> None:
    df_out = reshape_timebin(sample_df, timebin_new=Timedelta("300s"))
    expected = DataFrame({
        "dataset": ["dataset1"]*3,
        "filename": [f"filename{i+1}" for i in range(3)],
        "start_time": [0]*3,
        "end_time": [300.0]*3,
        "start_frequency": [0]*3,
        "end_frequency": [1200]*3,
        "annotation": ["lbl1", "lbl2", "lbl1"],
        "annotator": ["ann1", "ann2", "ann1"],
        "start_datetime": [Timestamp("2025-01-01 12:00:00+01:00"),
                           Timestamp("2025-01-01 13:00:00+01:00"),
                           Timestamp("2025-01-01 14:00:00+01:00")
                           ],
        "end_datetime": [Timestamp("2025-01-01 12:05:00+01:00"),
                           Timestamp("2025-01-01 13:05:00+01:00"),
                           Timestamp("2025-01-01 14:05:00+01:00")
                           ],
        "is_box": [0] * 3,

    })

    assert df_out.equals(expected)

def test_simple_reshape_hourly(sample_df: DataFrame) -> None:
    df_out = reshape_timebin(sample_df, timebin_new=Timedelta(hours=1))
    assert not df_out.empty
    assert all(df_out["end_time"] == 3600.0)
    assert df_out["end_frequency"].max() == sample_df["end_frequency"].max()
    assert set(df_out["annotation"]) <= set(sample_df["annotation"])
    assert set(df_out["annotator"]) <= set(sample_df["annotator"])

def test_reshape_daily_multiple_bins(sample_df: DataFrame) -> None:
    df_out = reshape_timebin(sample_df, timebin_new=Timedelta(days=1))
    assert not df_out.empty
    assert all(df_out["end_time"] == 86400.0)
    assert df_out["start_datetime"].min() >= sample_df["start_datetime"].min().floor("D")
    assert df_out["end_datetime"].max() <= sample_df["end_datetime"].max().ceil("D")

def test_with_manual_timestamps_vector(sample_df: DataFrame) -> None:
    t0 = sample_df["start_datetime"].min().floor("30min")
    t1 = sample_df["end_datetime"].max().ceil("30min")
    ts_vec = list(date_range(t0, t1, freq="30min"))

    df_out = reshape_timebin(
        sample_df,
        timebin_new=Timedelta(hours=1),
        timestamp=ts_vec,
    )

    assert not df_out.empty
    assert all(isinstance(t, Timestamp) for t in df_out["start_datetime"])
    assert df_out["end_time"].iloc[0] == 3600.0

def test_empty_result_when_no_matching(sample_df: DataFrame) -> None:
    with pytest.raises(ValueError, match="DataFrame is empty"):
        reshape_timebin(DataFrame(), Timedelta(hours=1))

