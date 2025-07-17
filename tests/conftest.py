import pytest
from pandas import to_datetime, DataFrame, concat

@pytest.fixture
def df_weak_detections():
    data = {
        "dataset": ["dataset"] * 3,
        "filename": ["file1.wav", "file2.wav", "file3.wav"],
        "start_time": [0, 0, 0],
        "end_time": [3600, 3600, 3600],
        "start_frequency": [0, 0, 0],
        "end_frequency": [120, 120, 120],
        "annotation": ["label1", "label2", "label1"],
        "annotator": ["annotator1", "annotator2", "annotator1"],
        "start_datetime": to_datetime(
            ["2022-07-13 03:00:00+00:00", "2022-07-13 04:00:00+00:00", "2022-07-13 05:00:00+00:00"]
        ),
        "end_datetime": to_datetime(
            ["2022-07-13 04:00:00+00:00", "2022-07-13 05:00:00+00:00", "2022-07-13 06:00:00+00:00"]
        ),
        "is_box": [0, 0, 0],
    }

    return DataFrame(data)

@pytest.fixture
def df_strong_detections():
    data = {
        "dataset": ["dataset"] * 3,
        "filename": ["file1.wav", "file2.wav", "file3.wav"],
        "start_time": [1515.2, 123.456, 789.101],
        "end_time": [1515.25, 125.728, 789.999],
        "start_frequency": [20000, 26895, 9636],
        "end_frequency": [20500, 28456, 10579],
        "annotation": ["label1", "label2", "label1"],
        "annotator": ["annotator1", "annotator2", "annotator1"],
        "start_datetime": to_datetime(
            ["2022-07-13 03:15:01.123456+00:00", "2022-07-13 04:56:07.789168+00:00", "2022-07-13 05:13:42.183648+00:00"]
        ),
        "end_datetime": to_datetime(
            ["2022-07-13 03:15:01.173456+0000", "2022-07-13 04:56:10.061168+0000", "2022-07-13 05:13:43.081648+0000"]
        ),
        "is_box": [1, 1 ,1],
    }

    return DataFrame(data)

@pytest.fixture
def df_strong_and_weak_detections(df_weak_detections, df_strong_detections):
    """Fixture combining weak and strong detections, sorted by start_datetime."""
    df = concat([df_weak_detections, df_strong_detections], ignore_index=True)
    return df.sort_values("start_datetime").reset_index(drop=True)

