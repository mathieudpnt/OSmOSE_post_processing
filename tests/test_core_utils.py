import pytest
from pandas import DataFrame, Timedelta, Timestamp, date_range
from pandas.tseries import frequencies
from pytz import timezone

from post_processing.utils.core_utils import (
    get_coordinates,
    get_count,
    get_labels_and_annotators,
    get_season,
    get_sun_times,
    get_time_range_and_bin_size,
    localize_timestamps,
    round_begin_end_timestamps,
    timedelta_to_str,
)


def test_coordinates_valid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = [["42", "-71"]]

    def fake_box(msg: str, title: str, fields: list[str]) -> list[str]:
        return inputs.pop(0)

    monkeypatch.setattr("easygui.multenterbox", fake_box)

    lat, lon = get_coordinates()
    assert lat == 42.0
    assert lon == -71.0


def test_coordinates_cancelled_input(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_box(msg: str, title: str, fields: list[str]) -> None:
        return None

    monkeypatch.setattr("easygui.multenterbox", fake_box)

    with pytest.raises(TypeError, match="was cancelled"):
        get_coordinates()


def test_coordinates_invalid_then_valid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should re-prompt if invalid lat/lon entered, then succeed."""
    inputs = [["900", "50"], ["45", "100"]]  # first invalid, then valid

    def fake_box(msg: str, title: str, fields: list[str]) -> list[str]:
        return inputs.pop(0)

    monkeypatch.setattr("easygui.multenterbox", fake_box)

    lat, lon = get_coordinates()
    assert lat == 45.0
    assert lon == 100.0


def test_coordinates_non_numeric_input(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = [["abc", "-20"], ["10", "-20"]]

    def fake_box(msg: str, title: str, fields: list[str]) -> list[str]:
        return inputs.pop(0)

    monkeypatch.setattr("easygui.multenterbox", fake_box)

    lat, lon = get_coordinates()
    assert lat == 10.0
    assert lon == -20.0


@pytest.mark.parametrize(
    ("ts", "northern", "expected"),
    [
        # Northern hemisphere
        (Timestamp("2025-03-15"), True, ("spring", 2025)),
        (Timestamp("2025-06-21"), True, ("summer", 2025)),
        (Timestamp("2025-09-01"), True, ("autumn", 2025)),
        (Timestamp("2025-01-10"), True, ("winter", 2024)),  # Jan -> year - 1
        (Timestamp("2025-02-28"), True, ("winter", 2024)),  # Feb -> year - 1
        (Timestamp("2024-02-29"), True, ("winter", 2023)),  # leap year Feb
        (Timestamp("2025-12-25"), True, ("winter", 2025)),  # Dec -> current year

        # Southern hemisphere
        (Timestamp("2025-03-15"), False, ("autumn", 2025)),
        (Timestamp("2025-06-21"), False, ("winter", 2025)),
        (Timestamp("2025-09-01"), False, ("spring", 2025)),
        (Timestamp("2025-01-10"), False, ("summer", 2024)),  # Jan -> year - 1
        (Timestamp("2025-02-28"), False, ("summer", 2024)),  # Feb -> year - 1
        (Timestamp("2024-02-29"), False, ("summer", 2023)),  # leap year Feb
        (Timestamp("2025-12-25"), False, ("summer", 2025)),  # Dec -> current year
    ]
)

def test_get_season(ts: Timestamp, northern: bool, expected: tuple[str, int]) -> None:
    assert get_season(ts, northern=northern) == expected


@pytest.mark.parametrize(
    ("start", "stop", "lat", "lon"),
    [
        (Timestamp("2025-06-01 00:00:00+00:00"),
         Timestamp("2025-06-03 23:59:59+00:00"),
         49.4333,
         -1.5167),
        (Timestamp("2025-12-21 00:00:00+00:00"),
         Timestamp("2025-12-22 23:59:59+00:00"),
         -34.9011,
         -56.1645),
    ]
)
def test_get_sun_times_valid_input(start: Timestamp, stop: Timestamp, lat: float, lon: float) -> None:
    results = get_sun_times(start, stop, lat, lon)
    h_sunrise, h_sunset, dt_dawn, dt_day, dt_dusk, dt_night = results

    n_days = (stop.normalize() - start.normalize()).days + 1

    # All lists have the same length
    assert all(len(lst) == n_days for lst in results)

    # Type checks
    assert all(isinstance(h, float) for h in h_sunrise)
    assert all(isinstance(h, float) for h in h_sunset)
    assert all(isinstance(ts, Timestamp) for ts in dt_dawn)
    assert all(isinstance(ts, Timestamp) for ts in dt_day)
    assert all(isinstance(ts, Timestamp) for ts in dt_dusk)
    assert all(isinstance(ts, Timestamp) for ts in dt_night)

    # Sanity check: sunrise < sunset
    for sunrise, sunset in zip(h_sunrise, h_sunset, strict=False):
        assert 0 <= sunrise <= 24
        assert 0 <= sunset <= 24
        assert sunrise < sunset

    # Sanity check: dawn < day < dusk < night
    for dawn, day, dusk, night in zip(dt_dawn, dt_day, dt_dusk, dt_night, strict=False):
        assert dawn < day < dusk < night


@pytest.mark.parametrize(
    ("start", "stop", "lat", "lon"),
    [
        (Timestamp("2025-06-01 00:00:00"),
         Timestamp("2025-06-03 23:59:59"),
         49.4333,
         -1.5167),
    ]
)
def test_get_sun_times_naive_timestamps(start: Timestamp, stop: Timestamp, lat: float, lon: float) -> None:
    with pytest.raises(ValueError, match="start and stop must be timezone-aware"):
        get_sun_times(start, stop, lat, lon)


#%% get_count

def test_get_count_basic() -> None:
    df = DataFrame({
        "start_datetime": [
            Timestamp("2025-01-01 00:05:00"),
            Timestamp("2025-01-01 00:15:00"),
            Timestamp("2025-01-01 00:35:00"),
        ],
        "annotation": ["A", "A", "A"],
        "annotator": ["ann1", "ann1", "ann1"],
    })

    result = get_count(df, bin_size=Timedelta("30min"))

    assert list(result.index) == [
        Timestamp("2025-01-01 00:00:00"),
        Timestamp("2025-01-01 00:30:00"),
    ]
    assert result.columns == ["A-ann1"]
    assert result["A-ann1"].tolist() == [2, 1]


def test_get_count_multiple_annotators() -> None:
    df = DataFrame({
        "start_datetime": [
            Timestamp("2025-01-01 00:05:00"),
            Timestamp("2025-01-01 00:15:00"),
            Timestamp("2025-01-01 00:20:00"),
            Timestamp("2025-01-01 00:40:00"),
        ],
        "annotation": ["A", "B", "A", "B"],
        "annotator": ["ann1", "ann1", "ann1", "ann1"],
    })

    result = get_count(df, bin_size=Timedelta("30min"))

    assert set(result.columns) == {"A-ann1", "B-ann1"}
    assert result.loc[Timestamp("2025-01-01 00:00:00")].to_dict() == {
        "A-ann1": 2,
        "B-ann1": 1,
    }
    assert result.loc[Timestamp("2025-01-01 00:30:00")].to_dict() == {
        "A-ann1": 0,
        "B-ann1": 1,
    }

def test_get_count_multiple_labels() -> None:
    df = DataFrame({
        "start_datetime": [
            Timestamp("2025-01-01 00:05:00"),
            Timestamp("2025-01-01 00:15:00"),
            Timestamp("2025-01-01 00:20:00"),
            Timestamp("2025-01-01 00:40:00"),
        ],
        "annotation": ["A", "A", "A", "A"],
        "annotator": ["ann1", "ann2", "ann2", "ann1"],
    })

    result = get_count(df, bin_size=Timedelta("30min"))

    assert set(result.columns) == {"A-ann1", "A-ann2"}
    assert result.loc[Timestamp("2025-01-01 00:00:00")].to_dict() == {
        "A-ann1": 1,
        "A-ann2": 2,
    }
    assert result.loc[Timestamp("2025-01-01 00:30:00")].to_dict() == {
        "A-ann1": 1,
        "A-ann2": 0,
    }

def test_get_count_multiple_labels_annotators() -> None:
    df = DataFrame({
        "start_datetime": [
            Timestamp("2025-01-01 00:05:00"),
            Timestamp("2025-01-01 00:15:00"),
            Timestamp("2025-01-01 00:20:00"),
            Timestamp("2025-01-01 00:40:00"),
        ],
        "annotation": ["A", "B", "A", "B"],
        "annotator": ["ann1", "ann1", "ann2", "ann2"],
    })

    result = get_count(df, bin_size=Timedelta("30min"))

    assert set(result.columns) == {"A-ann1", "B-ann2"}
    assert result.loc[Timestamp("2025-01-01 00:00:00")].to_dict() == {
        "A-ann1": 1,
        "B-ann2": 0,
    }
    assert result.loc[Timestamp("2025-01-01 00:30:00")].to_dict() == {
        "A-ann1": 0,
        "B-ann2": 1,
    }

def test_get_count_empty_df() -> None:
    with pytest.raises(ValueError, match="`df` contains no data"):
        get_count(DataFrame(), Timedelta("1h"))

#%% get_labels_and_annotators

def test_get_labels_and_annotators_valid_entry() -> None:
    df = DataFrame({
        "annotation": ["A", "B"],
        "annotator": ["ann1", "ann2"],
    })
    labels, annotators = get_labels_and_annotators(df)
    assert labels == ["A", "B"]
    assert annotators == ["ann1", "ann2"]


def test_single_label_multiple_annotators() -> None:
    df = DataFrame({
        "annotation": ["A", "A"],
        "annotator": ["ann1", "ann2"],
    })
    labels, annotators = get_labels_and_annotators(df)
    assert labels == ["A", "A"]
    assert annotators == ["ann1", "ann2"]


def test_single_annotator_multiple_labels() -> None:
    df = DataFrame({
        "annotation": ["A", "B"],
        "annotator": ["ann1", "ann1"],
    })
    labels, annotators = get_labels_and_annotators(df)
    assert labels == ["A", "B"]
    assert annotators == ["ann1", "ann1"]


def test_mismatched_labels_and_annotators() -> None:
    df = DataFrame({
        "annotation": ["A", "B", "C"],
        "annotator": ["ann1", "ann2", "ann2"],
    })
    with pytest.raises(ValueError, match="annotators and .* labels must match"):
        get_labels_and_annotators(df)


def test_get_labels_and_annotators_empty_dataframe() -> None:
    with pytest.raises(ValueError, match="`df` contains no data"):
        get_labels_and_annotators(DataFrame())

#%% localize_timestamps

def test_localize_all_naive() -> None:
    tz = timezone("Europe/Paris")
    timestamps = [Timestamp("2025-01-01 12:00:00"), Timestamp("2025-01-02 13:00:00")]
    localized = localize_timestamps(timestamps, tz)
    for ts in localized:
        assert ts.tzinfo is not None
        assert ts.tz.zone == tz.zone


def test_already_localized() -> None:
    tz = timezone("Europe/Paris")
    timestamps = [
        Timestamp("2025-01-01 12:00:00", tz=tz),
        Timestamp("2025-01-02 13:00:00", tz=tz),
    ]
    localized = localize_timestamps(timestamps, tz)
    assert localized == timestamps


def test_mixed_naive_and_aware() -> None:
    tz = timezone("Europe/Paris")
    timestamps = [
        Timestamp("2025-01-01 12:00:00"),
        Timestamp("2025-01-02 13:00:00", tz=tz),
    ]
    localized = localize_timestamps(timestamps, tz)
    assert localized[0].tzinfo.zone == tz.zone
    assert localized[1].tzinfo.zone == tz.zone

#%% get_time_range_and_bin_size

def test_time_range_timedelta() -> None:
    timestamps = [Timestamp("2025-08-20 12:00:00"), Timestamp("2025-08-20 14:30:00")]
    bin_size = Timedelta("1h")
    time_range, returned_bin = get_time_range_and_bin_size(timestamps, bin_size)

    expected = date_range(start="2025-08-20 12:00:00", end="2025-08-20 15:00:00", freq="1h")
    assert (time_range == expected).all()
    assert returned_bin == bin_size


def test_time_range_baseoffset() -> None:
    timestamps = [Timestamp("2025-08-20 12:00:00"), Timestamp("2025-08-20 14:30:00")]
    bin_size = frequencies.to_offset("1h")
    time_range, returned_bin = get_time_range_and_bin_size(timestamps, bin_size)

    expected = date_range(start="2025-08-20 12:00:00", end="2025-08-20 15:00:00", freq="1h")
    assert (time_range == expected).all()
    assert returned_bin == bin_size


def test_empty_timestamp_list() -> None:
    timestamps = []
    bin_size = Timedelta("1h")
    with pytest.raises(ValueError, match="`timestamp_list` is empty"):
        get_time_range_and_bin_size(timestamps, bin_size)


def test_invalid_timestamp_list_type() -> None:
    timestamps = "not_a_list"
    bin_size = Timedelta("1h")
    with pytest.raises(TypeError, match=r"`timestamp_list` must be a list\[Timestamp\]"):
        get_time_range_and_bin_size(timestamps, bin_size)


def test_invalid_timestamp_list_content() -> None:
    timestamps = [Timestamp("2025-08-20"), "not_a_timestamp"]
    bin_size = Timedelta("1h")
    with pytest.raises(TypeError, match=r"`timestamp_list` must be a list\[Timestamp\]"):
        get_time_range_and_bin_size(timestamps, bin_size)

#%% round_begin_end_timestamps

def test_round_begin_end_timestamps_empty_list() -> None:
    with pytest.raises(ValueError, match="`timestamp_list` is empty"):
        round_begin_end_timestamps([], Timedelta("1h"))

def test_round_begin_end_timestamps_invalid_entry() -> None:
    ts_list = [
        Timestamp("2025-01-01 00:15:00"),
        Timestamp("2025-01-01 01:45:00"),
    ]
    with pytest.raises(ValueError, match="Could not get start/end timestamps."):
        round_begin_end_timestamps(ts_list, "not_a_valid_entry")

def test_round_begin_end_timestamps_invalid_entry_2() -> None:
    with pytest.raises(TypeError, match=r"timestamp_list must be a list\[Timestamp\]"):
        round_begin_end_timestamps("not_a_valid_entry", Timedelta("1h"))

def test_round_begin_end_timestamps_valid_entry() -> None:
    ts_list = [
        Timestamp("2025-01-01 00:15:00"),
        Timestamp("2025-01-01 01:45:00"),
    ]
    start, end, bin_size = round_begin_end_timestamps(ts_list, Timedelta("1h"))

    assert start == Timestamp("2025-01-01 00:00:00")
    assert end == Timestamp("2025-01-01 02:00:00")
    assert bin_size == Timedelta("1h")

def test_round_begin_end_timestamps_valid_entry_2() -> None:
    ts_list = [
        Timestamp("2025-01-01 10:15:00"),
        Timestamp("2025-01-03 18:45:00"),
    ]
    start, end, bin_size = round_begin_end_timestamps(ts_list, frequencies.to_offset("1h"))

    assert start == Timestamp("2025-01-01 10:00:00")
    assert end == Timestamp("2025-01-03 19:00:00")
    assert bin_size == Timedelta("1h")

#%% timedelta_to_str

@pytest.mark.parametrize(
    ("td", "expected"),
    [
        (Timedelta(days=2), "2D"),
        (Timedelta(hours=5), "5h"),
        (Timedelta(minutes=30), "30min"),
        (Timedelta(seconds=45), "45s"),
        (Timedelta(days=1, hours=2), "26h"),
        (Timedelta(minutes=90), "90min"),
    ]
)
def test_timedelta_to_str(td, expected) -> None:
    assert timedelta_to_str(td) == expected

#%% add_weak_detection / json2df
#TODO
