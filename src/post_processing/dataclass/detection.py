from dataclasses import dataclass

from matplotlib import patches
from matplotlib.axes import Axes
from pandas import DataFrame, Series, Timedelta, Timestamp, to_datetime


@dataclass
class Detection:
    """A class to handle annotations from an APLOSE formatted DataFrame."""

    dataset: str
    filename: str
    start_datetime: Timestamp
    end_datetime: Timestamp
    start_time: float
    end_time: float
    duration: Timedelta
    start_frequency: float
    end_frequency: float
    annotation: str
    annotator: str
    annotator_expertise: str | None = None
    type: str | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        """Sanity checks."""
        if self.start_datetime >= self.end_datetime:
            msg = (
                f"start_datetime ({self.start_datetime}) must be strictly"
                f" less than end_datetime ({self.end_datetime})"
            )
            raise ValueError(msg)
        if self.start_time >= self.end_time:
            msg = f"start_time ({self.start_time}) must be strictly less than "
            f"end_time ({self.end_time})"
            raise ValueError(msg)
        if self.start_frequency >= self.end_frequency:
            msg = (
                f"start_frequency ({self.start_frequency}) must be strictly"
                f" less than end_frequency ({self.end_frequency})"
            )
            raise ValueError(msg)

    @classmethod
    def from_series(cls, row: Series) -> "Detection":
        """Create a Detection object from a pandas Series."""
        return cls(
            dataset=row["dataset"],
            filename=row["filename"],
            start_datetime=to_datetime(row["start_datetime"]),
            end_datetime=to_datetime(row["end_datetime"]),
            start_time=float(row["start_time"]),
            end_time=float(row["end_time"]),
            duration=to_datetime(row["end_datetime"])
            - to_datetime(row["start_datetime"]),
            start_frequency=float(row["start_frequency"]),
            end_frequency=float(row["end_frequency"]),
            annotation=row["annotation"],
            annotator=row["annotator"],
            annotator_expertise=row.get("annotator_expertise", None),
            type=row["type"],
            score=row.get("score", None),
        )

    @classmethod
    def from_df(cls, df: DataFrame) -> list["Detection"]:
        """Create a Detection object from a pandas DataFrame."""
        df = df[df["type"].str.lower().isin({"box", "weak"})]
        det_list = []
        for _, row in df.iterrows():
            det_list.append(cls.from_series(row))
        return det_list

    def draw_box(
        self,
        ax: Axes,
        color: str = "lime",
        label: str | None = None,
    ) -> Axes:
        """Draw annotation boxes for a given annotator on a spectrogram image."""
        x1 = self.start_datetime
        x2 = self.end_datetime

        y1 = self.start_frequency
        y2 = self.end_frequency

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        if label is not None:
            ax.text(
                x1,
                y2,
                label,
                color="white",
                verticalalignment="bottom",
                horizontalalignment="left",
                fontsize=6,
                bbox={
                    "facecolor": color,
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 1,
                },
            )

        return ax
