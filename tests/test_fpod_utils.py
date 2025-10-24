"""FPOD/ CPOD processing functions tests."""
import io
from datetime import datetime
from pathlib import Path

import datatest as dt
import pytest
from osekit.utils.timestamp_utils import strptime_from_text
from pandas import DataFrame, Timestamp, read_csv
from pandas.testing import assert_frame_equal

from post_processing.utils.fpod_utils import (
    csv_folder,
    deploy_period,
    extract_site,
    parse_timestamps,
    txt_folder,
    fpod2aplose,
    cpod2aplose,
    meta_cut_aplose,
    build_range,
    feeding_buzz,
    assign_daytime,
    is_dpm_col,
    build_aggregation_dict,
    resample_dpm)

SAMPLE_POD = """File,ChunkEnd,DPM,Nall,MinsOn
sample_dataset,2023/11/29 08:05,0,0,0

"""
SAMPLE_AP = """dataset,filename,start_time,end_time,start_frequency,end_frequency,
annotation,annotator,start_datetime,end_datetime,is_box
sample_dataset,,0,60,0,0,ann1,POD,2023-11-29T08:30:00.000+00:00,2023-11-29T08:31:00.000+00:00,0
sample_dataset,,0,60,0,0,ann1,POD,2023-11-29T08:31:00.000+00:00,2023-11-29T08:32:00.000+00:00,0
sample_dataset,,0,60,0,0,ann1,POD,2023-11-29T09:30:00.000+00:00,2023-11-29T09:31:00.000+00:00,0
sample_dataset,,0,60,0,0,ann1,POD,2023-11-30T08:30:00.000+00:00,2023-11-30T08:31:00.000+00:00,0
sample_dataset,,0,60,0,0,ann1,POD,2023-12-29T08:30:00.000+00:00,2023-12-29T08:31:00.000+00:00,0
sample_dataset,,0,60,0,0,ann1,POD,2024-11-29T08:30:00.000+00:00,2024-11-29T08:31:00.000+00:00,0
"""

@pytest.fixture
def pod_dataframe() -> DataFrame:
    data = DataFrame(
        {
            "File": [
                "sample_dataset",
                "sample_dataset",
                "sample_dataset",
                "sample_dataset",
                "sample_dataset",
                "sample_dataset",
            ],
            "ChunkEnd": [
                Timestamp("2023/11/29 08:30"),
                Timestamp("2023/11/29 08:31"),
                Timestamp("2023/11/29 08:32"),
                Timestamp("2023/11/29 08:33"),
                Timestamp("2023/11/29 08:34"),
                Timestamp("2023/11/29 08:35"),
            ],
            "deploy.name": [
                "site_deploy",
                "site_deploy",
                "site_deploy",
                "site_deploy",
                "site_deploy",
                "site_deploy",
            ],
            "DPM": [1, 1, 0, 0, 0, 0],
            "Nall": [44, 66, 0, 22, 0, 0],
            "MinsOn": [1, 1, 1, 1, 1, 0],
        },
    )

    return data.reset_index(drop=True)


@pytest.fixture
def aplose_dataframe() -> DataFrame:
    data = DataFrame(
        {
            "dataset": ["dataset_test", "dataset_test", "dataset_test", "dataset_test",
                        "dataset_test", "dataset_test"],
            "filename": ["", "", "", ""],
            "start_time": [0, 0, 0, 0, 0, 0],
            "end_time": [60, 60, 60, 60, 60, 60],
            "start_frequency": [0, 0, 0, 0, 0, 0],
            "end_frequency": [0, 0, 0, 0, 0, 0],
            "annotation": ["ann1", "ann1", "ann1", "ann1", "ann1", "ann1"],
            "annotator": ["POD", "POD", "POD", "POD", "POD", "POD"],
            "start_datetime": [
                Timestamp("2023-11-29T08:30:00.000+00:00"),
                Timestamp("2023-11-29T08:31:00.000+00:00"),
                Timestamp("2023-11-29T09:31:00.000+00:00"),
                Timestamp("2023-11-30T09:31:00.000+00:00"),
                Timestamp("2023-12-30T09:31:00.000+00:00"),
                Timestamp("2024-12-30T09:31:00.000+00:00"),
            ],
            "end_datetime": [
                Timestamp("2023-11-29T08:31:00.000+00:00"),
                Timestamp("2023-11-29T08:32:00.000+00:00"),
                Timestamp("2023-11-29T09:32:00.000+00:00"),
                Timestamp("2023-11-30T09:32:00.000+00:00"),
                Timestamp("2023-12-30T09:32:00.000+00:00"),
                Timestamp("2024-12-30T09:32:00.000+00:00"),
            ],
            "is_box": [0, 0, 0, 0, 0, 0],
            "deploy.name": ["site_campaign", "site_campaign", "site_campaign",
                            "site_campaign", "site_campaign", "site_campaign"],
        },
    )

    return data.reset_index(drop=True)

@pytest.fixture(scope="module")
@dt.working_directory(__file__)
def df_raw() -> DataFrame:
    return read_csv("pod_raw.csv")

@pytest.fixture(scope="module")
@dt.working_directory(__file__)
def df_ap() -> DataFrame:
    return read_csv("pod_aplose.csv")

@pytest.mark.mandatory
def test_columns(df_raw: DataFrame) -> None:
    dt.validate(
        df_raw.columns,
        {"File", "ChunkEnd", "DPM", "Nall", "MinsOn"},
    )

@pytest.mark.mandatory
def test_columns(df_ap: DataFrame) -> None:
    dt.validate(
        df_ap.columns,
        {"dataset","filename","start_time","end_time","start_frequency","end_frequency",
         "annotation","annotator","start_datetime","end_datetime","is_box"},
    )

def test_chunk_end(df_raw: DataFrame) -> None:
    dt.validate(df_raw["ChunkEnd"],
                strptime_from_text(df_raw["ChunkEnd"], "%Y/%m/%d %H:%M"))

def test_start_datetime(df_ap: DataFrame) -> None:
    dt.validate(df_ap["start_datetime"], strptime_from_text(df_ap["start_datetime"],
                                            "%Y-%m-%dT%H:%M:%S"))

@pytest.fixture
def sample_pod() -> DataFrame:
    df = read_csv(io.StringIO(SAMPLE_POD), parse_dates=["ChunkEnd"])
    return df.sort_values(["ChunkEnd"]).reset_index(drop=True)

# fpod2aplose


# cpod2aplose


# meta_cut_aplose


# build_range


# feeding_buzz


# assign_daytime


# fb_folder
def test_fb_folder_non_existent() -> None:
    with pytest.raises(FileNotFoundError):
        txt_folder(Path("/non/existent/folder"))

def test_fb_folder_no_files(tmp_path: pytest.fixture) -> None:
    with pytest.raises(ValueError, match="No .txt files found"):
        txt_folder(tmp_path)

# extract_site
def test_extract_site(self) -> None:
    input_data = [
        {"deploy.name":"Walde_Phase46"},
        {"deploy.name":"Site A Ile Haute_Phase8"},
        {"deploy.name":"Site B Ile Heugh_Phase9"},
        {"deploy.name":"Point E_Phase 4"},
    ]
    expected_site = [
        "Walde",
        "Site A Ile Haute",
        "Site B Ile Heugh",
        "Point E",
    ]
    expected_campaign = [
        "Phase46",
        "Phase8",
        "Phase9",
        "Phase 4",
    ]

    for variant, (input_row, site, campaign) in enumerate(
        zip(input_data, expected_site, expected_campaign, strict=False), start=1):
        with self.subTest(
            f"variation #{variant}",
            deploy_name=input_row["deploy.name"],
            expected_site=site,
            expected_campaign=campaign,
        ):
            df = DataFrame([input_row])
            result = extract_site(df)
            actual_site = result["site.name"].iloc[0]
            actual_campaign = result["campaign.name"].iloc[0]

            error_message_site = (
                f'Called extract_site() with deploy.name="{input_row["deploy.name"]}". '
                f'The function returned site.name="{actual_site}", but the test '
                f'expected "{expected_site}".'
            )

            error_message_campaign = (
                f'Called extract_site() with deploy.name="{input_row["deploy.name"]}". '
                f'The function returned campaign.name="{actual_campaign}", but the test'
                f'expected "{expected_campaign}".'
            )

            assert actual_site == expected_site, error_message_site
            assert actual_campaign == expected_campaign, error_message_campaign

            assert "deploy.name" in result.columns
            assert "value" in result.columns

# csv_folder
def test_csv_folder_non_existent() -> None:
    with pytest.raises(FileNotFoundError):
        csv_folder(Path("/non/existent/folder"))

def test_csv_folder_no_files(tmp_path: pytest.fixture) -> None:
    with pytest.raises(ValueError, match="No .csv files found"):
        csv_folder(tmp_path)

# is_dpm_col


# pf_datetime


# build_aggregation_dict


# resample_dpm


# parse_timestamps
def test_parse_timestamps() -> None:
    df = DataFrame({"date": ["2024-01-01T10:00:00", "06/01/2025 08:35"]})
    result = parse_timestamps(df, "date")
    expected = DataFrame({"date": ["2024-01-01 10:00:00",
                                   "2025-01-06 08:35:00"]}).astype("datetime64[ns]")
    assert_frame_equal(result, expected)

# deploy_period
def test_deploy_period() -> None:
    df = DataFrame(
        {
            "deploy.name": ["A", "A", "B"],
            "start_datetime": [
                datetime(2024, 1, 1, 10, 0, tzinfo=datetime.timezone.utc),
                datetime(2024, 1, 2, 15, 30, tzinfo=datetime.timezone.utc),
                datetime(2024, 1, 3, 8, 0, tzinfo=datetime.timezone.utc),
            ],
        })

    expected = DataFrame(
        {
            "deploy.name": ["A", "B"],
            "Début": [
                datetime(2024, 1, 1, 10, 0, tzinfo=datetime.timezone.utc),
                datetime(2024, 1, 3, 8, 0, tzinfo=datetime.timezone.utc),
            ],
            "Fin": [
                datetime(2024, 1, 2, 15, 30, tzinfo=datetime.timezone.utc),
                datetime(2024, 1, 3, 8, 0, tzinfo=datetime.timezone.utc),
            ],
        })
    result = deploy_period(df)
    assert_frame_equal(result, expected)

# actual_data