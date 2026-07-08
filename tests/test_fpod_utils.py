"""FPOD/ CPOD processing functions tests."""
from pathlib import Path

import pytest
import pytz
from pandas import DataFrame, Timedelta, Timestamp

from post_processing.utils.fpod_utils import (
    load_pod_folder,
    pod2aplose,
)

CLICKS_CPOD = """Minute,microsec,cycles,SPL_Pa,kHz,Bandwidth,end kHz,Qn,TrN
25/1/2019 11:45,55643215,7,38,130,0,121,2,38
25/1/2019 11:45,55707365,7,44,130,0,125,2,38
25/1/2019 11:45,55770865,7,36,132,0,131,2,38
25/1/2019 11:45,55830500,11,34,136,1,108,2,38
25/1/2019 11:45,55890495,10,33,135,1,131,2,38
"""

CLICKS_FPOD = """File,Minute,microsec,ICI,TrnAvPRF,Ncyc,ClkKHZ,IPIbefore,IPIatMax,IPIplus1,IPIplus2,EndIPI,ClkIPIrange,maxPk,maxPkE,Pkminus1%,Pkplus1%,PkAt,AmpReversals,tRateScore,Qn,TrnIDn,ClassID,Log(PRF)*10
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,64358756,40266515,10595,98,11,121,256,33,34,34,31,3,78,78,98,91,6,1,10,2,1,0,19
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,64358756,40276675,10160,98,11,121,256,33,33,33,33,3,79,79,98,91,5,1,10,2,1,0,19
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,64358756,40286600,9925,98,11,121,256,33,33,33,33,2,84,84,88,94,4,1,10,2,1,0,20
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,64358756,40296440,9840,98,10,121,256,33,33,34,33,3,79,79,91,100,4,1,10,2,1,0,20
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,64358756,40306520,10080,98,11,121,256,33,33,34,33,3,76,76,92,96,4,1,10,2,1,0,19
"""

TIMELOST = """File	podN,ChunkEnd,Minute,Temp,Angle,MinutesON,NBHF_DPM,DPM,Nfiltered/m,kHz_continuous_noise,NBHFclx,DOL_DPM,DOLclx,SONAR_DPM,SONARclx,Nall/m,%TimeLost,%m SonarRisk,%mSediment noise,LandmarkSeq_total,avOpThreshold
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,6669,05/05/2022 10:59,64348499,21.4,0,0m ON,0,108,14,0,0,0,0,0,0,,,0,0,0,0
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,6669,05/05/2022 11:59,64348559,21.4,0,0m ON,0,108,14,0,0,0,0,0,0,548.9,100,0,0,0,0
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,6669,05/05/2022 12:59,64348619,22.4,0,0,0,81.6,60,0,0,0,0,0,0,0.2,100,0,0,0,0
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,6669,05/05/2022 13:59,64348679,23,4,1.62,20,78,60,0,0,0,0,0,0,0,100,0,0,0,0
CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3,6669,05/05/2022 14:59,64348739,23,3,0.28,0,78,60,0,0,0,0,0,0,0,100,0,0,0,0
"""


@pytest.fixture
def pod_dataframe() -> DataFrame:
    return DataFrame({
        "File": [
            "Site A ile Haute 2019 01 25 POD3055 file01.CP3",
            "Site A ile Haute 2019 01 25 POD3055 file01.CP3",
            "Site A ile Haute 2019 01 25 POD3055 file01.CP3",
            "Site A ile Haute 2019 01 25 POD3055 file01.CP3",
            "Site A ile Haute 2019 01 25 POD3055 file01.CP3",
        ],
        "podN": [6669, 6669, 6669, 6669, 6669],
        "ChunkEnd": [
            "24/01/2019 06:17",
            "24/01/2019 06:18",
            "24/01/2019 06:19",
            "24/01/2019 06:20",
            "24/01/2019 06:21",
        ],
        "Minute": [64348546, 64348547, 64348548, 64348549, 64348550],
        "DPM": [0, 1, 1, 0, 0],
        "Nall": [0, 216, 75, 0, 28],
        "MinsOn": [0, 1, 1, 1, 1],
    })


@pytest.fixture
def click_dataframe() -> DataFrame:
    return DataFrame({
        "File": [
            "CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3",
            "CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3",
            "CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3",
            "CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3",
            "CETIROISEPHASE1POINTB 2022 05 05 FPOD_6669 file0.FP3",
        ],
        "microsec": [40255920, 40266515, 40276675, 40286600, 40296440],
        "Minute": [64348546, 64348547, 64348548, 64348549, 64348550],
    })


@pytest.fixture
def pod_aplose(sample_df: DataFrame) -> DataFrame:
    """Create a POD Dataframe for testing."""
    sample_df["type"] = "WEAK"
    return sample_df


# csv_folder
def test_folder_single_csv(pod_dataframe: DataFrame, tmp_path: Path) -> None:
    """Test processing a single CSV file."""
    csv_file = tmp_path / "pod_folder" / "pod_dataframe.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    pod_dataframe.to_csv(csv_file, index=False)
    result = load_pod_folder(csv_file.parent, ext="csv")

    assert isinstance(result, DataFrame)
    assert "Deploy" in result.columns
    assert all(result["Deploy"] == "pod_dataframe")
    assert list(result.columns) == ["File", "podN", "ChunkEnd", "Minute", "DPM",
                                    "Nall", "MinsOn", "Deploy", "Datetime"]


def test_folder_single_txt(
        monkeypatch: pytest.MonkeyPatch,
        click_dataframe: DataFrame,
        tmp_path: Path) -> None:
    """Test processing a single CSV file."""
    monkeypatch.setattr("post_processing.utils.fpod_utils.process_feeding_buzz",
                        lambda df, species: df)
    txt_file = tmp_path / "click_folder" / "click_dataframe.txt"
    txt_file.parent.mkdir(parents=True, exist_ok=True)
    click_dataframe.to_csv(txt_file, index=False)
    result = load_pod_folder(txt_file.parent, ext="txt")

    assert isinstance(result, DataFrame)
    assert "Deploy" in result.columns
    assert all(result["Deploy"] == "click_dataframe")
    assert list(result.columns) == [
        "File",
        "microsec",
        "Minute",
        "Deploy",
        "Datetime",
    ]


def test_folder_multiple(pod_dataframe: DataFrame, tmp_path: Path) -> None:
    """Test processing multiple CSV files."""
    csv_file = tmp_path / "pod_folder" / "pod_dataframe1.csv", "pod_dataframe2.csv"


@pytest.mark.parametrize(
    ("mocked_df", "should_raise"),
    [
        pytest.param(
            DataFrame({
                "ChunkEnd": ["01/01/2024 12:00"],
                "DPM": [1],
                "MinsOn": [30.0],
                "microsec": [100],
            }),
            False,
            id="valid-dpm-columns",
        ),
        pytest.param(
            DataFrame({
                "ChunkEnd": ["01/01/2024 12:00"],
                "%TimeLost": [0.1],
                "Nall/m": [1.0],
                "File": ["f1"],
                "microsec": [100],
            }),
            False,
            id="valid-timelost-columns",
        ),
        pytest.param(
            DataFrame({
                "ChunkEnd": ["01/01/2024 12:00"],
                "col1": [0.1],
                "Nall/m": [1.0],
                "File": ["f1"],
                "microsec": [100],
            }),
            True,
            id="invalid-missing-timelost",
        ),
        pytest.param(
            DataFrame({
                "ChunkEnd": ["01/01/2024 12:00"],
                "%TimeLost": [0.1],
                "col1": [1.0],
                "File": ["f1"],
                "microsec": [100],
            }),
            True,
            id="invalid-missing-nall",
        ),
        pytest.param(
            DataFrame({
                "ChunkEnd": ["01/01/2024 12:00"],
                "File": ["f1"],
                "col1": [1],
                "MinsOn": ["x"],
                "microsec": [100],
            }),
            True,
            id="invalid-missing-dpm",
        ),
        pytest.param(
            DataFrame({
                "ChunkEnd": ["01/01/2024 12:00"],
                "File": ["f1"],
                "DPM": [1],
                "col3": ["x"],
                "microsec": [100],
            }),
            True,
            id="invalid-missing-minson",
        ),
        pytest.param(
            DataFrame({"col1": [1], "col2": [2], "col3": [3]}),
            True,
            id="invalid-no-required-columns",
        ),
    ],
)
def test_right_csv_format(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        mocked_df: DataFrame,
        should_raise: bool,
    ) -> None:
    """Mocked read_csv to test load_pod_folder column validation."""
    fake_path = Path("fake/deploy_01.csv")

    monkeypatch.setattr(Path, "rglob", lambda self, pattern: [fake_path])
    monkeypatch.setattr("post_processing.utils.fpod_utils.find_delimiter", lambda f: ";")
    monkeypatch.setattr("post_processing.utils.fpod_utils.read_csv", lambda *args, **kwargs: mocked_df)

    if should_raise:
        with pytest.raises((ValueError, KeyError)):
            load_pod_folder(Path("fake/folder"), "csv")
    else:
        result = load_pod_folder(Path("fake/folder"), "csv")
        assert isinstance(result, DataFrame)


# pod2aplose
@pytest.fixture
def sample_df() -> DataFrame:
    """Create a sample POD DataFrame for testing. Mocked load_pod_folder output."""
    return DataFrame({
        "Datetime": [
            Timestamp("15-01-2024 10:30:00"),
            Timestamp("15-01-2024 11:00:00"),
            Timestamp("15-01-2024 09:45:00"),
        ],
        "Deploy": ["deploy1", "deploy2", "deploy1"],
    })


@pytest.fixture
def empty_df() -> DataFrame:
    """Create a sample POD DataFrame for testing. Mimic load_pod_folder output."""
    return DataFrame({
        "Datetime": [],
        "Deploy": [],
    })

@pytest.fixture
def timezone():
    """Return UTC timezone for testing."""
    return pytz.UTC


def test_pod2aplose_basic_structure(sample_df: DataFrame, timezone) -> None:
    """Test that basic structure and required columns are present."""
    result = pod2aplose(
        df=sample_df,
        tz=pytz.UTC,
        dataset_name="dataset",
        annotation="porpoise",
        annotator="fpod",
        bin_size=Timedelta(seconds=60),
    )

    expected_columns = [
        "dataset",
        "filename",
        "start_time",
        "end_time",
        "min_frequency",
        "max_frequency",
        "annotation",
        "annotator",
        "start_datetime",
        "end_datetime",
        "type",
        "deploy",
    ]

    assert isinstance(result, DataFrame)
    assert list(result.columns) == expected_columns
    assert len(result) == len(sample_df)
    assert result["dataset"].iloc[0] == "dataset"
    assert all(result["dataset"] == "dataset")
    assert result["filename"].iloc[0] != 0
    assert all(result["start_time"] == 0)
    assert all(result["end_time"] == 60)
    assert all(result["min_frequency"] == 0)
    assert all(result["max_frequency"] == 0)
    assert all(result["annotation"] == "porpoise")
    assert all(result["annotator"] == "fpod")
    assert len(result["start_datetime"].iloc[0]) > 0
    assert len(result["end_datetime"].iloc[0]) > 0
    assert "deploy" in result.columns
    assert len(result["deploy"]) == len(sample_df)
    assert set(result["deploy"]) == {"deploy1", "deploy2"}


def test_pod2aplose_empty_dataframe(empty_df: DataFrame, timezone) -> None:
    """Test handling of empty DataFrame."""
    result = pod2aplose(
        df=empty_df,
        tz=pytz.UTC,
        dataset_name="dataset",
        annotation="porpoise",
        annotator="fpod",
        bin_size=Timedelta(seconds=60),
    )

    assert len(result) == 0
    assert list(result.columns) == [
        "dataset",
        "filename",
        "start_time",
        "end_time",
        "min_frequency",
        "max_frequency",
        "annotation",
        "annotator",
        "start_datetime",
        "end_datetime",
        "type",
        "deploy",
    ]

# meta_cut_aplose


# build_range


# feeding_buzz


# assign_daytime


# fb_folder
# def test_fb_folder_non_existent() -> None:
#     with pytest.raises(FileNotFoundError):
#         txt_folder(Path("/non/existent/folder"))
#
# def test_fb_folder_no_files(tmp_path: pytest.fixture) -> None:
#     with pytest.raises(ValueError, match="No .txt files found"):
#         txt_folder(tmp_path)

# extract_site
# def test_extract_site(self) -> None:
#     input_data = [
#         {"deploy.name":"Walde_Phase46"},
#         {"deploy.name":"Site A Ile Haute_Phase8"},
#         {"deploy.name":"Site B Ile Heugh_Phase9"},
#         {"deploy.name":"Point E_Phase 4"},
#     ]
#     expected_site = [
#         "Walde",
#         "Site A Ile Haute",
#         "Site B Ile Heugh",
#         "Point E",
#     ]
#     expected_campaign = [
#         "Phase46",
#         "Phase8",
#         "Phase9",
#         "Phase 4",
#     ]
#
#     for variant, (input_row, site, campaign) in enumerate(
#         zip(input_data, expected_site, expected_campaign, strict=False), start=1):
#         with self.subTest(
#             f"variation #{variant}",
#             deploy_name=input_row["deploy.name"],
#             expected_site=site,
#             expected_campaign=campaign,
#         ):
#             df = DataFrame([input_row])
#             result = extract_site(df)
#             actual_site = result["site.name"].iloc[0]
#             actual_campaign = result["campaign.name"].iloc[0]
#
#             error_message_site = (
#                 f'Called extract_site() with deploy.name="{input_row["deploy.name"]}". '
#                 f'The function returned site.name="{actual_site}", but the test '
#                 f'expected "{expected_site}".'
#             )
#
#             error_message_campaign = (
#                 f'Called extract_site() with deploy.name="{input_row["deploy.name"]}". '
#                 f'The function returned campaign.name="{actual_campaign}", but the test'
#                 f'expected "{expected_campaign}".'
#             )
#
#             assert actual_site == expected_site, error_message_site
#             assert actual_campaign == expected_campaign, error_message_campaign
#
#             assert "deploy.name" in result.columns
#             assert "value" in result.columns

# csv_folder
# def test_csv_folder_non_existent() -> None:
#     with pytest.raises(FileNotFoundError):
#         csv_folder(Path("/non/existent/folder"))
#
# def test_csv_folder_no_files(tmp_path: pytest.fixture) -> None:
#     with pytest.raises(ValueError, match="No .csv files found"):
#         csv_folder(tmp_path)

# is_dpm_col


# pf_datetime


# build_aggregation_dict


# resample_dpm


# parse_timestamps
# def test_parse_timestamps() -> None:
#     df = DataFrame({"date": ["2024-01-01T10:00:00", "06/01/2025 08:35"]})
#     result = parse_timestamps(df, "date")
#     expected = DataFrame({"date": ["2024-01-01 10:00:00",
#                                    "2025-01-06 08:35:00"]}).astype("datetime64[ns]")
#     assert_frame_equal(result, expected)

# deploy_period
# def test_deploy_period() -> None:
#     df = DataFrame(
#         {
#             "deploy.name": ["A", "A", "B"],
#             "start_datetime": [
#                 datetime(2024, 1, 1, 10, 0, tzinfo=datetime.timezone.utc),
#                 datetime(2024, 1, 2, 15, 30, tzinfo=datetime.timezone.utc),
#                 datetime(2024, 1, 3, 8, 0, tzinfo=datetime.timezone.utc),
#             ],
#         })
#
#     expected = DataFrame(
#         {
#             "deploy.name": ["A", "B"],
#             "Début": [
#                 datetime(2024, 1, 1, 10, 0, tzinfo=datetime.timezone.utc),
#                 datetime(2024, 1, 3, 8, 0, tzinfo=datetime.timezone.utc),
#             ],
#             "Fin": [
#                 datetime(2024, 1, 2, 15, 30, tzinfo=datetime.timezone.utc),
#                 datetime(2024, 1, 3, 8, 0, tzinfo=datetime.timezone.utc),
#             ],
#         })
#     result = deploy_period(df)
#     assert_frame_equal(result, expected)

# actual_data