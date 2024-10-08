import json
from pathlib import Path
from typing import Union
import pandas as pd
from dataclasses import dataclass

@dataclass
class Deployment:
    """
    A class to represent and manage metadata for a deployment.

    Attributes:
        path_json (Path, str): Path to the JSON file containing metadata.
        campaign (str, int): Campaign identifier.
        deployment (str, int): Deployment identifier.
        recorder (str, int): Recorder identifier.
        path_metadata (Path, str): Path to the file containing file metadata.
        path_origin_metadata (Path, str): Path to the file containing origin metadata.
        path_origin_timestamp (Path, str): Path to the file containing origin timestamps.
        path_segment_metadata (Path, str): Path to the file containing segment metadata.
        path_segment_timestamp (Path, str): Path to the file containing segment timestamps.
        path_pamguard (Path, str): Path to the PAMGuard annotations file.
        path_thalassa (Path, str): Path to the Thalassa annotations file.
        path_aplose (Path, str): Path to the APLOSE annotations file.
    """

    def __init__(
        self,
        path_json: Path | str = None,
        campaign: str | int = None,
        deployment: str | int = None,
        recorder: str | int = None,
        path_metadata: Union[Path | str] = None,
        path_origin_metadata: Path | str = None,
        path_origin_timestamp: Path | str = None,
        path_segment_metadata: Path | str = None,
        path_segment_timestamp: Path | str = None,
        path_pamguard: Path | str = None,
        path_thalassa: Path | str = None,
        path_aplose: Path | str = None,
    ) -> None:

        argument = [
            campaign,
            deployment,
            recorder,
            path_metadata,
            path_origin_metadata,
            path_origin_timestamp,
            path_segment_metadata,
            path_segment_timestamp,
        ]
        assert (
            all(arg is not None for arg in argument) or path_json is not None
        ), "Argument(s) missing"
        assert all(
            isinstance(arg, Union[str, int]) or arg is None
            for arg in [campaign, deployment, recorder]
        ), "Input arguments error"

        if path_json:
            self.load_json(path_json)
            self.datetime_deployment = pd.to_datetime(self.datetime_deployment)
            self.datetime_recovery = pd.to_datetime(self.datetime_recovery)

        else:
            self.load_metadata(
                campaign,
                deployment,
                recorder,
                path_metadata,
                path_origin_metadata,
                path_origin_timestamp,
                path_segment_metadata,
                path_segment_timestamp,
                path_pamguard,
                path_thalassa,
                path_aplose,
            )
            self.campaign = campaign
            self.deployment = deployment
            self.recorder = recorder
            self.datetime_deployment = self.origin_timestamp[0]
            self.datetime_recovery = self.origin_timestamp[-1]

        self.duration = self.datetime_recovery - self.datetime_deployment

    def load_json(self, path_json):
        """Load metadata from a JSON file and initialize attributes."""
        try:
            with open(path_json, "r") as file:
                metadata_json = json.load(file)

            for key, value in metadata_json.items():
                setattr(self, key.replace(" ", "_"), value)

            # assign None to self.path{detector} if it does not exist
            for attr in ["pamguard", "thalassa", "aplose"]:
                setattr(self, f"path_{attr}", getattr(self, f"path_{attr}", None))

            self.load_metadata(
                self.campaign,
                self.deployment,
                self.recorder,
                self.path_metadata,
                self.path_origin_metadata,
                self.path_origin_timestamp,
                self.path_segment_metadata,
                self.path_segment_timestamp,
                self.path_pamguard,
                self.path_thalassa,
                self.path_aplose,
            )

        except FileNotFoundError:
            raise ValueError(f"JSON file not found at {self.path_json}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON file at {self.path_json}")
        except Exception as e:
            raise ValueError(f"Error reading JSON file from {self.path_json}: {e}")

    def load_metadata(
        self,
        campaign,
        deployment,
        recorder,
        path_metadata,
        path_origin_metadata,
        path_origin_timestamp,
        path_segment_metadata,
        path_segment_timestamp,
        path_pamguard,
        path_thalassa,
        path_aplose,
    ):
        """Load and process various metadata"""
        try:
            attributes = []

            # name
            self.project = getattr(self, "project", None)
            if self.project == "APOCADO":
                self.name = f"C{campaign}D{deployment} ST{recorder}"
            else:
                self.name = f"{campaign}-{deployment}-{recorder}"
            attributes.append("name")

            # file metadata
            self.metadata = self.load_csv(path_metadata)
            metadata = self.metadata
            self.origin_sr = metadata["origin_sr"][0]
            self.audio_file_count = metadata["audio_file_count"][0]
            self.start_date = pd.Timestamp(metadata["start_date"][0])
            self.end_date = pd.Timestamp(metadata["end_date"][0])
            self.gps_coordinates = (metadata["lat"][0], metadata["lon"][0])
            self.depth = metadata["depth"][0]
            attributes.append(
                [
                    "metadata",
                    "origin_sr",
                    "audio_file_count",
                    "start_date",
                    "end_date",
                    "gps_coordinates",
                    "depth",
                ]
            )

            # origin  metadata
            self.origin_metadata = self.load_csv(
                path_origin_metadata, parse_dates=["timestamp"]
            )
            self.origin_duration = self.origin_metadata["duration"].to_list()
            attributes.append(["origin_metadata", "origin_duration"])

            # originin timestamp
            self.origin_timestamp_df = self.load_csv(
                path_origin_timestamp, parse_dates=["timestamp"]
            )
            self.origin_timestamp = self.origin_timestamp_df["timestamp"].to_list()
            self.origin_filename = self.origin_timestamp_df["filename"].to_list()
            attributes.append(["origin_timestamp", "origin_filename"])

            # segment metadata
            self.segment_metadata = self.load_csv(path_segment_metadata)
            attributes.append("segment_metadata")

            # segment_timestamp
            self.segment_timestamp_df = self.load_csv(
                path_segment_timestamp, parse_dates=["timestamp"]
            )
            self.segment_timestamp = self.segment_timestamp_df["timestamp"].to_list()
            self.segment_filename = self.segment_timestamp_df["filename"].to_list()
            attributes.append(["segment_timestamp", "segment_filename"])

            self.path_pamguard = path_pamguard
            self.path_thalassa = path_thalassa
            self.path_aplose = path_aplose
            for name in ["pamguard", "thalassa", "aplose"]:
                path = getattr(self, f"path_{name}", None)
                if path:
                    att = self.get_annotation_file(name)
                    attributes.extend(att)

            flat_attributes = [
                item
                for sublist in attributes
                for item in (sublist if isinstance(sublist, list) else [sublist])
            ]

            self.generate_dynamic_properties(flat_attributes)

        except FileNotFoundError as e:
            raise ValueError(f"File not found: {e.filename}")
        except pd.errors.ParserError:
            raise ValueError("Error parsing CSV file.")
        except Exception as e:
            raise ValueError(f"Error loading metadata: {e}")

    @staticmethod
    def load_csv(path: Union[Path, str], **kwargs) -> pd.DataFrame:
        """Helper method to load a CSV file and handle errors."""
        try:
            return pd.read_csv(path, **kwargs)
        except FileNotFoundError:
            raise ValueError(f"CSV file not found at {path}")
        except pd.errors.ParserError:
            raise ValueError(f"Error parsing CSV file at {path}")
        except Exception as e:
            raise ValueError(f"Error reading CSV file from {path}: {e}")

    def get_annotation_file(self, name: str):
        """Load and process manual annotation or automatic detection file in APLOSE format"""
        attributes = []
        path = getattr(self, f"path_{name}")
        df = (
            self.load_csv(path, parse_dates=["start_datetime", "end_datetime"])
            .sort_values("start_datetime")
            .reset_index(drop=True)
        )
        annotators = list(df["annotator"].unique())
        annotators = [item for item in annotators if not pd.isna(item)]
        annotations = list(df["annotation"].unique())
        annotations = [item for item in annotations if not pd.isna(item)]

        setattr(self, f"df_{name}", df)
        setattr(
            self,
            f"{name}_annotator",
            annotators[0] if len(annotators) == 1 else annotators,
        )
        setattr(
            self,
            f"{name}_annotation",
            annotations[0] if len(annotations) == 1 else annotations,
        )

        attributes.append(f"df_{name}")
        attributes.append(f"{name}_annotator")
        attributes.append(f"{name}_annotation")

        return attributes

    def generate_dynamic_properties(self, attributes):
        """Generate properties and their setters dynamically."""
        for attr in attributes:
            if hasattr(self, f"_{attr}"):
                setattr(
                    self.__class__,
                    attr,
                    property(self.getter_function(attr), self.setter_function(attr)),
                )

    def getter_function(self, attribute):
        """Create a getter function for an attribute."""

        def getter(self):
            return getattr(self, f"_{attribute}")

        return getter

    def setter_function(self, attribute):
        """Create a setter function for an attribute."""

        def setter(self, value):
            raise AttributeError(f"can't set attribute '{attribute}'")

        return

    def __str__(self):
        """Return a string representation of the deployment metadata."""
        result = f"Metadata of deployment {self.name}:\n"

        if all(self.metadata):
            list_display_metadata = [
                "origin_sr",
                "audio_file_count",
                "start_date",
                "end_date",
                "lat",
                "lon",
            ]

            ending_charac = [
                "(Hz)",
                "",
                "",
                "",
                "(DD)",
                "(DD)",
            ]
            for var, ct in zip(list_display_metadata, ending_charac):
                result += f"- {var} : {self.metadata[var][0]} {ct}\n"

            file = ""
            for name in ["pamguard", "thalassa", "aplose"]:
                path = getattr(self, f"path_{name}", None)
                if path:
                    file += f"{name}, "
            if file != "":
                result += f"- result file : {file[:-2]}"

        return result
