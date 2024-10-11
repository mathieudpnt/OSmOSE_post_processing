import pandas as pd
import gzip
from pathlib import Path
from tqdm import tqdm
import numpy as np
from numpy.lib.npyio import NpzFile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils.glider_config import NAV_STATE

def load_glider_nav(directory: Path):
    """Load the navigation data from glider output files in a specified directory.
    This function searches the given directory for compressed `.gz` files
    containing the substring 'gli' in their filenames.

    Parameters
    ----------
    directory : Path
        The path to the directory containing the glider output files. This can
        be provided as a string or a Path object.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing the combined navigation data from all
        matching glider output files. The DataFrame includes:
        - Columns from the original CSV files.
        - A 'yo' column representing the file number extracted from the filename.
        - A 'file' column indicating the source file for each row.
        - A 'Lat DD' column for latitude in decimal degrees.
        - A 'Lon DD' column for longitude in decimal degrees.
        - A 'Depth' column with depth values adjusted to be positive.
    """

    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    file = [f for f in directory.glob("*.gz") if "gli" in f.name]

    if not len(file) > 0:
        raise FileNotFoundError(f"Directory '{directory}' does not contain '.gz' files.")

    all_rows = []  # Initialize an empty list to store the contents of all CSV files
    yo = []  # List to store the file numbers
    data = []

    first_file = True
    file_number = 1  # Initialize the file number

    for f in tqdm(file, "Reading data..."):
        with gzip.open(f, "rt") as gz_file:
            delimiter = ";"  # Specify the desired delimiter
            gz_reader = pd.read_csv(gz_file, delimiter=delimiter)
            # If it's the first file, append the header row
            if first_file:
                all_rows.append(gz_reader.columns.tolist())
                first_file = False
            # Add the rows from the current CSV file to the all_rows list
            all_rows.extend(gz_reader.values.tolist())
            # Add yo number to the yo list
            yo.extend([str(f).split(".")[-2]] * len(gz_reader))
            data.extend([f.name] * len(gz_reader))
            file_number += 1  # Increment the file number for the next file

    # Create a DataFrame from the combined data
    df = pd.DataFrame(all_rows)
    df.columns = df.iloc[0]  # set 1st row as headers
    df = df.iloc[1:, 0:-1]  # delete last column and 1st row

    # Add the yo number to the DataFrame
    df["yo"] = [int(x) for x in yo]

    df["file"] = data
    df = df.drop(df[(df["Lat"] == 0) & (df["Lon"] == 0)].index).reset_index(drop=True)
    df["Lat"] = [
        int(x) + (((x - int(x)) / 60) * 100) if not np.isnan(x) else np.nan
        for x in df["Lat"] / 100
    ]
    df["Lon"] = [
        int(x) + (((x - int(x)) / 60) * 100) if not np.isnan(x) else np.nan
        for x in df["Lon"] / 100
    ]
    df["Depth"] = -df["Depth"]
    df["NavState"] = df["NavState"].replace(NAV_STATE)
    df["Timestamp"] = [pd.to_datetime(ts, format="%d/%m/%Y %H:%M:%S").tz_localize('UTC') for ts in df['Timestamp']]
    df = df.sort_values(by=["Timestamp"]).reset_index(drop=True)

    return df


def plot_nav_state(df: pd.DataFrame, npz: NpzFile):
    """Plot the LTAS from a npz file and the associated glider state of navigation

    Parameters
    ----------
    df: pd.DataFrame, glider navigation data
    npz: npz file containing the LTAS matrix, its associated frequency and time vectors
    """
    f = npz["Freq"]
    sxx = npz["LTAS"]
    t = npz["time"]

    fig, (ax, cax) = plt.subplots(2, gridspec_kw={"height_ratios": [1, 0.05]})
    im = ax.imshow(
        sxx[1:-1],
        aspect="auto",
        extent=[t[0], t[-1], f[1], f[-1]],
        origin="lower",
        vmin=40,
        vmax=100,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt="%H:%M"))
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=3600 * 4))
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Date")

    cbar = fig.colorbar(im, orientation="horizontal", cax=cax)
    cbar.ax.set_xlabel("dB ref 1µPa/Hz")

    ax2 = ax.twinx()
    ax2.plot(
        df[(df["Timestamp"] >= t[0]) & (df["Timestamp"] <= t[-1])]["Timestamp"],
        df[(df["Timestamp"] >= t[0]) & (df["Timestamp"] <= t[-1])]["NavState"],
        color="white",
        linewidth=1,
        alpha=0.7,
        zorder=3,
    )
    ax2.set_ylabel("Navigation state")

    plt.tight_layout()
    plt.show()
