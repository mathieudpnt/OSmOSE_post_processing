"""Utils for first analysis on annotation/detection DataFrame."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from pandas import Timedelta


# TODO: define if this is useful with users
def set_y_axis_to_percentage(
    ax: plt.Axes,
    max_annotation_number: int,
    res: int = 10,
) -> None:
    """Change ax properties.

    Whether the plot is visualized in percentage or in raw values.

    Parameters
    ----------
    res: int
        The resolution of y-ticks.
    ax: Axes
        Plot axes
    max_annotation_number: int
        Maximum number of annotations possible

    """
    y_max = int(max([patch.get_height() for patch in ax.patches]))

    bars = np.arange(0, 101, res)
    y_pos = [max_annotation_number * pos / 100 for pos in bars]

    ax.set_yticks(y_pos, bars)
    current_label = ax.get_ylabel().split("\n")
    current_label[0] = current_label[0] + " (%)"
    new_label = "\n".join(current_label)
    ax.set_ylabel(new_label)

    # set y-axis limit
    ax.set_ylim([0, min(int(1.4 * y_max), max_annotation_number)])


def timedelta_to_str(td: Timedelta) -> str:
    """From a Timedelta to corresponding string."""
    seconds = int(td.total_seconds())

    if seconds % 86400 == 0:
        return f"{seconds // 86400}D"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"