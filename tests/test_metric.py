from unittest.mock import patch

import pytest
from pandas import DataFrame, Series, date_range, Timestamp, Timedelta

from disclose.utils.metric import (
    detection_perf,
    _get_classification_timestamps,
    _compute_confusion_matrix,
    _prepare_annotator_vectors,
    get_detection_timestamps,
    _get_f_score,
    _get_recall,
    _get_precision,
    _compute_scores,
    _log_detection_results,
)
from disclose.utils.core import get_count
from disclose.utils.filtering import get_annotators, get_labels


# %% _get_classification_timestamps


def test_get_classification_timestamps():
    timestamps = date_range("2026-08-01", periods=10, freq="h")

    vec1 = Series([1, 0, 1, 0, 2, 0, 1, 1, 0, 1], index=timestamps)
    vec2 = Series([1, 1, 0, 0, 2, 1, 1, 1, 0, "not_a_bolean"], index=timestamps)

    result = _get_classification_timestamps(vec1, vec2)

    assert result == {
        "true_pos": [timestamps[0], timestamps[6], timestamps[7]],
        "false_pos": [timestamps[1], timestamps[5]],
        "false_neg": [timestamps[2]],
        "true_neg": [timestamps[3], timestamps[8]],
        "error": [timestamps[4], timestamps[-1]],
    }


# %% _compute_confusion_matrix


def test_compute_confusion_matrix():
    timestamps = {
        "true_pos": [
            Timestamp("2024-01-01"),
            Timestamp("2024-01-02"),
            Timestamp("2024-01-03"),
        ],
        "false_pos": [Timestamp("2024-01-04"), Timestamp("2024-01-05")],
        "false_neg": [Timestamp("2024-01-06")],
        "true_neg": [Timestamp("2024-01-07"), Timestamp("2024-01-08")],
        "error": [],
    }

    result = _compute_confusion_matrix(timestamps)

    assert result == {
        "true_pos": 3,
        "false_pos": 2,
        "false_neg": 1,
        "true_neg": 2,
        "error": 0,
    }


def test_compute_confusion_matrix_raises_on_errors():
    timestamps = {
        "true_pos": [Timestamp("2024-01-01")],
        "false_pos": [],
        "false_neg": [Timestamp("2024-01-02")],
        "true_neg": [],
        "error": [Timestamp("2024-01-03")],
    }

    with pytest.raises(ValueError, match="1 errors in metric computation."):
        _compute_confusion_matrix(timestamps)


def test_compute_confusion_matrix_raises_when_precision_impossible():
    timestamps = {
        "true_pos": [],
        "false_pos": [],
        "false_neg": [Timestamp("2024-01-01")],
        "true_neg": [Timestamp("2024-01-02")],
        "error": [],
    }

    with pytest.raises(ValueError, match="Precision/Recall computation impossible."):
        _compute_confusion_matrix(timestamps)


def test_compute_confusion_matrix_raises_when_recall_impossible():
    timestamps = {
        "true_pos": [],
        "false_pos": [Timestamp("2024-01-01")],
        "false_neg": [],
        "true_neg": [Timestamp("2024-01-02")],
        "error": [],
    }

    with pytest.raises(ValueError, match="Precision/Recall computation impossible."):
        _compute_confusion_matrix(timestamps)


# %% _prepare_annotator_vectors


def test_prepare_annotator_vectors(sample_df: DataFrame):
    sample_df_2_ann = sample_df[
        (sample_df["annotator"].isin(["ann1", "ann2"])) & (sample_df["label"] == "lbl1")
    ]
    df_count = get_count(df=sample_df_2_ann, bin_size=Timedelta("10min"))

    anns = get_annotators(sample_df_2_ann)
    lbl = get_labels(sample_df_2_ann)

    vec1, vec2, selection1, selection2 = _prepare_annotator_vectors(
        sample_df_2_ann,
        df_count,
        ref=(anns[0], lbl),
    )

    assert selection1 == (anns[0], lbl)
    assert selection2 == (anns[1], lbl)
    assert all(vec1 == df_count["lbl1-ann1"])
    assert all(vec2 == df_count["lbl1-ann2"])


def test_prepare_annotator_vectors_requires_two_annotators(sample_df: DataFrame):
    sample_df_1_ann = sample_df[sample_df["annotator"] == "ann1"]
    df_count = get_count(df=sample_df_1_ann, bin_size=Timedelta("10min"))

    with pytest.raises(
        ValueError,
        match="Two annotators needed, DataFrame contains 1 annotators",
    ):
        _prepare_annotator_vectors(
            sample_df_1_ann,
            df_count,
            ref=("ann1", "lbl1"),
        )


def test_prepare_annotator_vectors_reference_not_found(sample_df: DataFrame):
    sample_df_2_ann = sample_df[sample_df["annotator"].isin(["ann1", "ann2"])]
    df_count = get_count(df=sample_df_2_ann, bin_size=Timedelta("10min"))

    with pytest.raises(
        ValueError,
        match="No detection found for ann1/laymelli",
    ):
        _prepare_annotator_vectors(
            sample_df_2_ann,
            df_count,
            ref=("ann1", "laymelli"),
        )


# %% get_detection_timestamps


def test_get_detection_timestamps(sample_df: DataFrame):
    sample_df_2_ann = sample_df[
        (sample_df["annotator"].isin(["ann1", "ann2"])) & (sample_df["label"] == "lbl1")
    ]

    timestamps = get_detection_timestamps(
        sample_df_2_ann,
        ref=("ann1", "lbl1"),
    )

    assert set(timestamps) == {
        "true_pos",
        "false_pos",
        "false_neg",
        "true_neg",
        "error",
    }

    assert all(
        isinstance(timestamp, Timestamp)
        for values in timestamps.values()
        for timestamp in values
    )


# %% _get_precision / _get_recall / _get_f_score


def test_get_precision():
    confusion_matrix = {
        "true_pos": 8,
        "false_pos": 2,
        "false_neg": 4,
        "true_neg": 10,
        "error": 0,
    }

    assert _get_precision(confusion_matrix) == 0.8


def test_get_recall():
    confusion_matrix = {
        "true_pos": 8,
        "false_pos": 2,
        "false_neg": 4,
        "true_neg": 10,
        "error": 0,
    }

    assert _get_recall(confusion_matrix) == pytest.approx(8 / 12)


def test_get_f_score():
    confusion_matrix = {
        "true_pos": 8,
        "false_pos": 2,
        "false_neg": 4,
        "true_neg": 10,
        "error": 0,
    }

    precision = 8 / 10
    recall = 8 / 12
    expected = 2 * (precision * recall) / (precision + recall)

    assert _get_f_score(confusion_matrix) == pytest.approx(expected)


# %% compute_score


def test_compute_scores():
    confusion_matrix = {
        "true_pos": 8,
        "false_pos": 2,
        "false_neg": 4,
        "true_neg": 10,
        "error": 0,
    }

    precision, recall, f_score = _compute_scores(confusion_matrix)

    assert precision == 0.8
    assert recall == pytest.approx(8 / 12)
    assert f_score == pytest.approx(2 * (0.8 * (8 / 12)) / (0.8 + (8 / 12)))


# %% _log_detection_results


def test_log_detection_results(sample_df: DataFrame):
    matrix = {
        "true_pos": 8,
        "false_pos": 2,
        "false_neg": 4,
        "true_neg": 10,
        "error": 0,
    }

    with patch("logging.info") as mock_info:
        _log_detection_results(
            selection1=("ann1", "lbl1"),
            selection2=("ann2", "lbl1"),
            matrix=matrix,
            df=sample_df,
        )

    log = mock_info.call_args[0][0]

    assert "Detection results" in log
    assert "ann1/lbl1" in log
    assert "ann2/lbl1" in log
    assert "True positive:" in log
    assert "8" in log
    assert "True negative:" in log
    assert "10" in log
    assert "False positive:" in log
    assert "2" in log
    assert "False negative:" in log
    assert "4" in log
    assert "Precision:" in log
    assert "0.80" in log
    assert "Recall:" in log
    assert "0.67" in log
    assert "F-score:" in log


# %% detection_perf


def test_detection_perf(sample_df: DataFrame):
    sample_df_2_ann = sample_df[
        (sample_df["annotator"].isin(["ann1", "ann2"])) & (sample_df["label"] == "lbl1")
    ]

    score, confusion_matrix, classification = detection_perf(
        sample_df_2_ann,
        ref=("ann1", "lbl1"),
    )

    assert set(classification) == {
        "true_pos",
        "false_pos",
        "false_neg",
        "true_neg",
        "error",
    }

    assert confusion_matrix == {
        "true_pos": len(classification["true_pos"]),
        "false_pos": len(classification["false_pos"]),
        "false_neg": len(classification["false_neg"]),
        "true_neg": len(classification["true_neg"]),
        "error": len(classification["error"]),
    }

    assert score[0] == 1
    assert score[1] == 1
    assert score[2] == 1
