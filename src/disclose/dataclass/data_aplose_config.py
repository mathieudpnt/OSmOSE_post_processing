from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pandas import Timedelta, Timestamp

from disclose.utils.filtering import read_dataframe, get_max_time


@dataclass(slots=True)
class DataAploseConfig:
    """Configuration object for loading and filtering APLOSE-formatted detection data.

    Parameters
    ----------
    detection_file : Path
        Path to the detection file to be loaded.
    timebin_new : Timedelta | None
        Optional resampling or re-binning time resolution.
    start_datetime : Timestamp | None
        Start datetime used to filter detections.
    end_datetime : Timestamp | None
        End datetime used to filter detections.
    annotator : str | list[str] | None
        Filter for one or multiple annotators.
    label : str | list[str] | None
        Filter for one or multiple annotation labels.
    type : str | None
        Optional detection type filter.
    recording_file : Path | None
        Optional external recording period file.
    min_frequency : float | None
        Minimum frequency threshold for filtering detections.
    max_frequency : float | None
        Maximum frequency threshold for filtering detections.
    confidence : float | None
        Minimum confidence threshold for detections.
    filename_format : str | None
        Optional filename formatting rule.
    timebin_origin : Timedelta | None
        Automatically computed base time bin derived from the detection file.
        This field is set internally and should not be provided manually.

    """

    detection_file: Path | None = None
    recording_file: Path | None = None
    filename_format: str | None = None
    start_datetime: Timestamp | None = None
    end_datetime: Timestamp | None = None
    annotator: str | list[str] | None = None
    label: str | list[str] | None = None
    type: str | None = None
    min_frequency: float | None = None
    max_frequency: float | None = None
    confidence: float | None = None
    timebin_new: Timedelta | None = None
    timebin_origin: Timedelta | None = None

    def __post_init__(self) -> None:
        """Compute derived configuration fields after initialization."""
        if self.detection_file is not None and self.timebin_origin is None:
            df = read_dataframe(self.detection_file)
            object.__setattr__(self, "timebin_origin", get_max_time(df))

    @classmethod
    def empty(cls) -> DataAploseConfig:
        return cls()

    @classmethod
    def from_dict(
        cls,
        config: dict | list[dict],
    ) -> DataAploseConfig | list[DataAploseConfig]:
        """Build one or more configuration objects from dictionaries."""

        if isinstance(config, dict):
            return cls(**config)

        return [cls(**c) for c in config]

    @classmethod
    def merge(cls, configs: list[DataAploseConfig]) -> DataAploseConfig:
        """Merge several configurations."""

        if not configs:
            return cls.empty()

        merged = {}

        for field in cls.__dataclass_fields__:
            values = [
                getattr(conf, field)
                for conf in configs
                if getattr(conf, field) is not None
            ]

            if not values:
                merged[field] = None
                continue

            unique = sorted(set(values))

            if len(unique) == 1:
                merged[field] = unique[0]

            elif field == "start_datetime":
                merged[field] = min(unique)

            elif field == "end_datetime":
                merged[field] = max(unique)

            else:
                merged[field] = unique

        return cls(**merged)
