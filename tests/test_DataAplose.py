import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pytest
from pandas import DataFrame, Series, Timedelta
from pandas.tseries import frequencies

from post_processing.dataclass.data_aplose import DataAplose


def test_data_aplose_init(df_strong_and_weak_detections: DataFrame) -> None:
    data = DataAplose(df_strong_and_weak_detections)

    assert isinstance(data.df, DataFrame)
    assert sorted(data.annotators) == sorted(["annotator1", "annotator2"])
    assert sorted(data.labels) == sorted(["label1", "label2"])
    assert data.begin == df_strong_and_weak_detections["start_datetime"].min()
    assert data.end == df_strong_and_weak_detections["end_datetime"].max()


def test_filter_df_single_pair(df_strong_and_weak_detections: DataFrame) -> None:
    data = DataAplose(df_strong_and_weak_detections)
    label, annotator, legend, datetimes = data._filter_df("annotator1", "label1")  # noqa: SLF001

    assert label == ["label1"]
    assert annotator == ["annotator1"]
    assert isinstance(datetimes, list)
    assert isinstance(datetimes[0], Series)
    assert not datetimes[0].empty


def test_filter_df_multiple_pairs(df_strong_and_weak_detections: DataFrame) -> None:
    data = DataAplose(df_strong_and_weak_detections)
    label, annotator, legend, datetimes = data._filter_df(  # noqa: SLF001
        ["annotator1", "annotator2"],
        ["label1", "label2"],
    )

    assert label == ["label1", "label2"]
    assert annotator == ["annotator1", "annotator2"]
    assert len(datetimes) == 2
    assert all(isinstance(df, Series) for df in datetimes)


def test_filter_df_invalid_annotator(df_strong_and_weak_detections: DataFrame) -> None:
    data = DataAplose(df_strong_and_weak_detections)
    with pytest.raises(
        ValueError,
        match='Annotator "wrong_annotator" not in APLOSE DataFrame',
    ):
        data._filter_df("wrong_annotator", "label1")


def test_set_ax_uses_2hour_locator(df_strong_and_weak_detections: DataFrame) -> None:
    data = DataAplose(df_strong_and_weak_detections)
    fig, ax = plt.subplots()
    bin_size = Timedelta("1h")
    tick_freq = frequencies.to_offset("2h")

    ax = data.set_ax(
        ax=ax,
        bin_size=bin_size,
        x_ticks_res=tick_freq,
        date_format="%y-%m-%d",
    )

    locator = ax.xaxis.get_major_locator()
    assert isinstance(locator, mdates.HourLocator)
    assert locator._get_interval() == 2


def test_histo_methods_dont_crash(df_strong_and_weak_detections: DataFrame) -> None:
    data = DataAplose(df_strong_and_weak_detections)
    fig, ax = plt.subplots()
    bin_size = Timedelta("1h")
    tick_freq = frequencies.to_offset("2h")
    ax = data.set_ax(
        ax=ax,
        bin_size=bin_size,
        x_ticks_res=tick_freq,
        date_format="%y-%m-%d",
    )
    data.histo(ax=ax, annotator="annotator1", label="label1")
