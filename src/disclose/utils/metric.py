"""Plot functions used for DataAplose objects."""

from __future__ import annotations

import logging
from typing import TypedDict

from pandas import DataFrame, DatetimeIndex, Series, Timestamp

from disclose.dataclass.recording_period import RecordingPeriod
from disclose.utils.core import get_count
from disclose.utils.filtering import (
    get_annotators,
    get_labels,
    get_max_time,
    intersection_or_union,
)


class Classification(TypedDict):
    true_positive: list[Timestamp]
    false_positive: list[Timestamp]
    true_negative: list[Timestamp]
    false_negative: list[Timestamp]
    error: list[Timestamp]


class ConfusionMatrix(TypedDict):
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    error: int


Scores = tuple[float, float, float]


def _get_classification_timestamps(
    vec1: Series,
    vec2: Series,
) -> dict[str, list[Timestamp]]:
    """Return timestamps classified as true/false positives/negatives or errors.

    Both vectors are expected to contain binary values (0 or 1), with other
    values classified as errors.
    """
    true_pos_mask = (vec1 == 1) & (vec2 == 1)
    false_pos_mask = (vec1 == 0) & (vec2 == 1)
    false_neg_mask = (vec1 == 1) & (vec2 == 0)
    true_neg_mask = (vec1 == 0) & (vec2 == 0)
    error_mask = (~vec1.isin([0, 1])) | (~vec2.isin([0, 1]))

    return {
        "true_pos": vec1.index[true_pos_mask].tolist(),
        "false_pos": vec1.index[false_pos_mask].tolist(),
        "false_neg": vec1.index[false_neg_mask].tolist(),
        "true_neg": vec1.index[true_neg_mask].tolist(),
        "error": vec1.index[error_mask].tolist(),
    }


def _compute_confusion_matrix(timestamps: dict[str, list[Timestamp]]) -> dict[str, int]:
    """Compute the confusion matrix counts from classified timestamps.

    Parameters
    ----------
    timestamps: dict[str, list[Timestamp]]
        Lists of timestamps for each category, true/false positives/negatives or errors

    Returns
    -------
    confusion_matrix: dict[str, int]
        Counts for true_pos, false_pos, false_neg, true_neg, error

    """
    confusion_matrix = {key: len(values) for key, values in timestamps.items()}

    if confusion_matrix["error"] != 0:
        msg = f"{confusion_matrix['error']} errors in metric computation."
        raise ValueError(msg)

    if (
        confusion_matrix["true_pos"] + confusion_matrix["false_pos"] == 0
        or confusion_matrix["false_neg"] + confusion_matrix["true_pos"] == 0
    ):
        msg = "Precision/Recall computation impossible."
        raise ValueError(msg)

    return confusion_matrix


def _prepare_annotator_vectors(
    df: DataFrame,
    df_count: DataFrame,
    ref: tuple[str, str],
) -> tuple[Series, Series, tuple[str, str], tuple[str, str]]:
    """Prepare the count vectors for the reference and second annotator/label pair.

    Parameters
    ----------
    df: DataFrame
        APLOSE formatted detection DataFrame
    df_count: DataFrame
        Detection counts per timebin, indexed by "{label}-{annotator}" columns
    ref: tuple[str, str]
        Reference annotator/label pair

    Returns
    -------
    vec1: Series
        Count vector for the reference annotator/label
    vec2: Series
        Count vector for the second annotator/label
    selection1: tuple[str, str]
        (annotator1, label1)
    selection2: tuple[str, str]
        (annotator2, label2)

    """
    annotators = get_annotators(df)
    annotators = [annotators] if isinstance(annotators, str) else annotators
    if len(annotators) != 2:  # noqa: PLR2004
        msg = f"Two annotators needed, DataFrame contains {len(annotators)} annotators"
        raise ValueError(msg)

    labels = get_labels(df)

    annotator1, label1 = ref
    detections1 = df[(df["annotator"] == annotator1) & (df["label"] == label1)]
    if detections1.empty:
        msg = f"No detection found for {annotator1}/{label1}"
        raise ValueError(msg)
    vec1 = df_count[f"{label1}-{annotator1}"]

    annotator2 = next(ant for ant in annotators if ant != annotator1)
    label2 = (
        next(lbl for lbl in labels if lbl != label1)
        if len(labels) == 2  # noqa: PLR2004
        else label1
    )
    vec2 = df_count[f"{label2}-{annotator2}"]

    return vec1, vec2, (annotator1, label1), (annotator2, label2)


def get_detection_timestamps(
    df: DataFrame,
    *,
    ref: tuple[str, str],
    time: DatetimeIndex | None = None,
    effort: RecordingPeriod | None = None,
) -> dict[str, list[Timestamp]]:
    """Retrieve the timestamps classified as true_pos/false_pos/false_neg/true_neg/error.

    Useful for verification purposes, e.g. inspecting the exact
    false-positive timebins for a given annotator/label pair.

    Parameters
    ----------
    df: DataFrame
        APLOSE formatted detection DataFrame
    ref: tuple[str, str]
        Reference annotator/label pair
    time: DatetimeIndex
        DatetimeIndex from a specified beginning to end
    effort: RecordingPeriod
        Recording effort period

    Returns
    -------
    timestamps: dict[str, list[Timestamp]]
        Lists of timestamps for each category: true_pos, false_pos,
        false_neg, true_neg, error

    """
    timebin = get_max_time(df)
    df_count = get_count(df, timebin, time, effort)

    vec1, vec2, _, _ = _prepare_annotator_vectors(df, df_count, ref)
    return _get_classification_timestamps(vec1, vec2)


def _compute_scores(
    confusion_matrix: dict[str, int],
) -> tuple[float | int, float | int, float | int]:
    """Compute precision, recall and f-score from a confusion matrix.

    Parameters
    ----------
    confusion_matrix: dict[str, int]
        Counts for true_pos, false_pos, false_neg, true_neg, error

    Returns
    -------
    precision: float
    recall: float
    f_score: float

    """
    return (
        _get_precision(confusion_matrix),
        _get_recall(confusion_matrix),
        _get_f_score(confusion_matrix),
    )


def _get_precision(confusion_matrix: dict) -> float:
    """Compute precision."""
    tp = confusion_matrix["true_pos"]
    fp = confusion_matrix["false_pos"]
    return tp / (tp + fp)


def _get_recall(confusion_matrix: dict) -> float:
    """Compute recall."""
    tp = confusion_matrix["true_pos"]
    fn = confusion_matrix["false_neg"]
    return tp / (tp + fn)


def _get_f_score(confusion_matrix: dict) -> float:
    """Compute F-score."""
    precision = _get_precision(confusion_matrix)
    recall = _get_recall(confusion_matrix)
    return 2 * (precision * recall) / (precision + recall)


def _log_detection_results(
    selection1: tuple[str, str],
    selection2: tuple[str, str],
    matrix: dict,
    df: DataFrame,
    effort: RecordingPeriod | None = None,
) -> None:
    """Log detection performance results."""
    annotator1, label1 = selection1
    annotator2, label2 = selection2
    precision = _get_precision(matrix)
    recall = _get_recall(matrix)
    f_score = _get_f_score(matrix)

    msg_result = (
        f"{' Detection results ':#^50}\n"
        f"{'Config 1:':<10}{f'{annotator1}/{label1}':>40}\n"
        f"{'Config 2:':<10}{f'{annotator2}/{label2}':>40}\n\n"
        f"{'True positive:':<25}{matrix['true_pos']:>25}\n"
        f"{'True negative:':<25}{matrix['true_neg']:>25}\n"
        f"{'False positive:':<25}{matrix['false_pos']:>25}\n"
        f"{'False negative:':<25}{matrix['false_neg']:>25}\n\n"
        f"{'Precision:':<25}{precision:>25.2f}\n"
        f"{'Recall:':<25}{recall:>25.2f}\n"
        f"{'F-score:':<25}{f_score:>25.2f}\n\n"
        f"{'Union:':<25}{len(intersection_or_union(df, 'union', effort)):>25.0f}\n"
        f"{'Intersection:':<25}{len(intersection_or_union(df, 'intersection', effort)):>25.0f}\n"
    )
    logging.info(msg_result)


def detection_perf(
    df: DataFrame,
    *,
    ref: tuple[str, str],
    time: DatetimeIndex | None = None,
    effort: RecordingPeriod | None = None,
) -> tuple[Scores, ConfusionMatrix, Classification]:
    """Compute the performance metrics for detection.

    Performances are computed with a reference annotator/label pair
    in comparison to a second annotator/label pair.

    Parameters
    ----------
    df: DataFrame
        APLOSE formatted detection DataFrame
    ref: tuple[str, str]
        Tuple of annotator/detector pairs.
    time: DatetimeIndex
        DatetimeIndex from a specified beginning to end
    effort: RecordingPeriod
        Recording effort period

    Returns
    -------
    score: Scores
        precision, recall and f-score
    confusion_matrix: ConfusionMatrix
        Count of each category: true positive, false positive, true negative, false negative, error
    classification: Classification
        Lists of timestamps for each category: true positive, false positive, true negative, false negative, error

    """
    timebin = get_max_time(df)
    df_count = get_count(df, timebin, time, effort)

    vec1, vec2, selection1, selection2 = _prepare_annotator_vectors(df, df_count, ref)

    classification = _get_classification_timestamps(vec1, vec2)
    confusion_matrix = _compute_confusion_matrix(classification)
    score = _compute_scores(confusion_matrix)

    _log_detection_results(
        selection1=selection1,
        selection2=selection2,
        matrix=confusion_matrix,
        df=df,
        effort=effort,
    )

    return score, confusion_matrix, classification
