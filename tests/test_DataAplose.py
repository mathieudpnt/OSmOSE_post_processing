import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pytest
from pandas import DataFrame, Timedelta
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
    filtered_data = data.filter_df(annotator="annotator1", label="label1")

    assert isinstance(filtered_data, DataFrame)
    assert sorted(set(filtered_data["annotation"])) == ["label1"]
    assert sorted(set(filtered_data["annotator"])) == ["annotator1"]
    assert not filtered_data.empty


def test_filter_df_multiple_pairs(df_strong_and_weak_detections: DataFrame) -> None:
    data = DataAplose(df_strong_and_weak_detections)
    filtered_data = data.filter_df(
        annotator=["annotator1", "annotator2"],
        label=["label1", "label2"],
    )

    assert isinstance(filtered_data, DataFrame)
    assert sorted(set(filtered_data["annotation"])) == ["label1", "label2"]
    assert sorted(set(filtered_data["annotator"])) == ["annotator1", "annotator2"]
    assert not filtered_data.empty


def test_filter_df_invalid_annotator(df_strong_and_weak_detections: DataFrame) -> None:
    data = DataAplose(df_strong_and_weak_detections)
    with pytest.raises(
        ValueError,
        match='Annotator "wrong_annotator" not in APLOSE DataFrame',
    ):
        data.filter_df(annotator="wrong_annotator", label="label1")


def test_filter_df_invalid_label(df_strong_and_weak_detections: DataFrame) -> None:
    data = DataAplose(df_strong_and_weak_detections)
    with pytest.raises(
        ValueError,
        match='Label "wrong_label" not in APLOSE DataFrame',
    ):
        data.filter_df(annotator="annotator2", label="wrong_label")


def test_filter_df_invalid_combination(
    df_strong_and_weak_detections: DataFrame,
) -> None:
    data = DataAplose(df_strong_and_weak_detections)
    with pytest.raises(
        ValueError,
        match="DataFrame with annotator 'annotator1' /"
        " label 'label2' contains no weak detection.",
    ):
        data.filter_df(annotator="annotator1", label="label2")


def test_filter_df_invalid_lists_size(
    df_strong_and_weak_detections: DataFrame,
) -> None:
    data = DataAplose(df_strong_and_weak_detections)
    with pytest.raises(
        ValueError,
        match=r"Length of annotator \(2\) and label \(1\) must match.",
    ):
        data.filter_df(annotator=["annotator1", "annotator2"], label=["label2"])


def test_set_ax_uses_2hour_locator(df_strong_and_weak_detections: DataFrame) -> None:
    data = DataAplose(df_strong_and_weak_detections)
    fig, ax = plt.subplots()
    tick_freq = frequencies.to_offset("2h")

    ax = data.set_ax(
        ax=ax,
        x_ticks_res=tick_freq,
        date_format="%y-%m-%d",
    )

    locator = ax.xaxis.get_major_locator()
    assert isinstance(locator, mdates.HourLocator)
    assert locator._get_interval() == 2  # noqa: PLR2004


def test_histo_methods_dont_crash(df_strong_and_weak_detections: DataFrame) -> None:
    data = DataAplose(df_strong_and_weak_detections)
    fig, ax = plt.subplots()
    bin_size = Timedelta("1h")
    tick_freq = frequencies.to_offset("2h")
    ax = data.set_ax(
        ax=ax,
        x_ticks_res=tick_freq,
        date_format="%y-%m-%d",
    )
    data.plot(
        mode="histogram",
        ax=ax,
        annotator="annotator1",
        label="label1",
        bin_size=bin_size,
    )


@pytest.mark.parametrize("mode", ["scatter", "heatmap", "timeline"])
def test_plot_scatter_heatmap_timeline(df_strong_and_weak_detections: DataFrame, mode: str) -> None:  # noqa: E501
    data = DataAplose(df_strong_and_weak_detections)
    data.lon = 0
    data.lat = 0
    fig, ax = plt.subplots()
    data.plot(mode=mode, ax=ax, annotator="annotator1", label="label1", color="red")


def test_plot_agreement(df_strong_and_weak_detections: DataFrame) -> None:
    data = DataAplose(df_strong_and_weak_detections)
    fig, ax = plt.subplots()
    data.plot(mode="agreement",
              ax=ax, annotator=["annotator1", "annotator2"],
              label=["label1", "label2"],
              bin_size=Timedelta("1h")
              )
