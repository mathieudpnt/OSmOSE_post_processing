from dataclasses import dataclass

from matplotlib import patches
from matplotlib.axes import Axes
from pandas import DataFrame, Series, Timestamp, to_datetime


@dataclass
class Detection:
    """A class to handle annotations from an APLOSE formatted DataFrame."""

    dataset: str
    filename: str
    start_datetime: Timestamp
    end_datetime: Timestamp
    start_time: float
    end_time: float
    start_frequency: float
    end_frequency: float
    annotation: str
    annotator: str
    annotator_expertise: str | None = None
    type: str | None = None

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
            start_frequency=float(row["start_frequency"]),
            end_frequency=float(row["end_frequency"]),
            annotation=row["annotation"],
            annotator=row["annotator"],
            annotator_expertise=row["annotator_expertise"],
            type=row["type"],
        )

    @classmethod
    def from_df(cls, df: DataFrame) -> list["Detection"]:
        """Create a Detection object from a pandas DataFrame."""
        det_list = []
        for _, row in df.iterrows():
            det_list.append(cls.from_series(row))
        return det_list

    def draw_box(
        self,
        ax: Axes,
        color: str = "lime",
    ) -> Axes:
        """Draw annotation boxes for a given annotator on a spectrogram image."""
        x1 = self.start_datetime
        x2 = self.end_datetime

        y1 = self.start_frequency
        y2 = self.end_frequency

        _, labels = ax.get_legend_handles_labels()
        label = self.annotation if self.annotation not in labels else "_nolegend_"

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            label=label,
        )
        ax.add_patch(rect)

        return ax
