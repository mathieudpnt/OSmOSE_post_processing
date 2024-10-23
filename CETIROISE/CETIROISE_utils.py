import pandas as pd
from OSmOSE.utils.timestamp_utils import strptime_from_text


def clean_pamguard_false_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans PAMGuard whistle and moan detector first detection of each audio file (might be very specific to Sylence data).
    This is because the first detection on each audio file corresponds to the detection of an electronic buzz made by the recorder

    Parameters
    ----------
    df: pd.DataFrame
        An APLOSE formatted result file (presumably from PAMGuard)

    Returns
    -------
    df_clean: pd.DataFrame
        The result cleaned of the false detections
    """
    # read the filename date of the detection
    filenames = df["filename"]
    tz_data = df["start_datetime"][0].tz
    filename_datetimes = [
        strptime_from_text(fn, "%Y_%m_%d_%H_%M_%S").tz_localize(tz_data)
        for fn in filenames
    ]

    # read detection date in start_datetime
    start_datetimes = df["start_datetime"]

    # compare date of filename detection and date of detection
    # and delete all lines for which the detection happens in the 5 first seconds of the file
    idx_false_detections = []
    for i in range(0, len(start_datetimes)):
        d = (start_datetimes[i] - filename_datetimes[i]).total_seconds()
        if d < 5:
            idx_false_detections.append(i)

    df_clean = df.drop(labels=idx_false_detections)

    return df_clean


def fpod2aplose(
    df: pd.DataFrame, dataset_name: str, annotation: str, bin_size: int
) -> pd.DataFrame:
    # TODO : reformat
    """transforms a FPOD.xls result files to a APLOSE csv results file
    dataset_name = easygui.enterbox("Dataset name (enter a string): ")
    species = easygui.enterbox("Label (enter a species and a call type): ")
    det_bin_size = int(easygui.enterbox("Size of the detection bin (in sec)"))
    """
    # Read detection begin time
    # Number of detections
    nb_det = len(df)

    # Transform start detection format from string to absolute datatime (with time zone info)
    df_FPOD_start_dt = sorted(
        [
            pytz.timezone(tz_data).localize(pd.to_datetime(x, format="%d/%m/%Y %H:%M"))
            for x in df["Date heure"]
        ]
    )
    # Compute the absolute end date time of detection
    df_FPOD_end_dt = sorted(
        [x + timedelta(seconds=det_bin_size) for x in df_FPOD_start_dt]
    )

    # Change datetime format to match with APLOSE format
    df_FPOD_start_AP = [
        datetime.strftime(dt, ("%Y-%m-%dT%H:%M:%S.%f"))[:-3]
        + datetime.strftime(dt, ("%Y-%m-%dT%H:%M:%S.%f%z"))[26:29]
        + ":"
        + datetime.strftime(dt, ("%Y-%m-%dT%H:%M:%S.%f%z"))[29:]
        for dt in df_FPOD_start_dt
    ]
    df_FPOD_end_AP = [
        datetime.strftime(dt, ("%Y-%m-%dT%H:%M:%S.%f"))[:-3]
        + datetime.strftime(dt, ("%Y-%m-%dT%H:%M:%S.%f%z"))[26:29]
        + ":"
        + datetime.strftime(dt, ("%Y-%m-%dT%H:%M:%S.%f%z"))[29:]
        for dt in df_FPOD_end_dt
    ]

    # Build the dataframe
    data = {
        "dataset": [dataset_name] * nb_det,
        "filename": [""] * nb_det,
        "start_time": [0] * nb_det,
        "end_time": [det_bin_size] * nb_det,
        "start_frequency": [0] * nb_det,
        "end_frequency": [0] * nb_det,
        "annotation": [species] * nb_det,
        "annotator": ["FPOD"] * nb_det,
        "start_datetime": df_FPOD_start_AP,
        "end_datetime": df_FPOD_end_AP,
    }
    df_APLOSE = pd.DataFrame(data)
    # ▒ Save the csv file in the same folder as the FPOD results csv file
    df_APLOSE.to_csv(FPOD_file_path + "_APLOSE.csv", index=False)

    return


# TODO : reformatter et trouver un nom plus explicite -> @maëlle
from post_processing_detections.utilities.def_func import (
    get_detection_files,
    sorting_detections,
    input_date,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

file = "path/to/file"

delimiter = ","
df = pd.read_csv(file[0], sep=delimiter)
list_annotators = list(df["annotator"].drop_duplicates())
list_labels = list(df["annotation"].drop_duplicates())
max_freq = int(max(df["end_frequency"]))


# %%
startF = df["start_frequency"]
endF = df["end_frequency"]
# datetime det
dt_det = df["start_datetime"]
dt_det = pd.to_datetime(df["start_datetime"], format="%Y-%m-%dT%H:%M:%S.%f%z")
ts_det = [datetime.timestamp(d) for d in dt_det]

fig, ax = plt.subplots()
plt.hist(startF, 100)
plt.hist(endF, 100)


# %% Look for positions of detections with a small bandwidth
bandwidth = [end - start for end, start in zip(endF, startF)]
pos = [x for x in range(len(bandwidth)) if bandwidth[x] < 200]
small_bw = [bandwidth[x] for x in pos]

startF_sb = [startF[x] for x in pos]
ts_det_sb = [ts_det[x] for x in pos]

fig, ax = plt.subplots()
plt.scatter(ts_det_sb, startF_sb)

# %% Look for positions of detections with a small startf

pos = [x for x in range(len(startF)) if startF[x] < 10000]

bandwidth_sf = [bandwidth[x] for x in pos]
startF_sf = [startF[x] for x in pos]
ts_det_sf = [ts_det[x] for x in pos]

fig, ax = plt.subplots(figsize=(20, 9))
plt.scatter(ts_det_sf[5000:6000], startF_sf[5000:6000])


# %% All freq

fig, ax = plt.subplots(figsize=(20, 9))
plt.scatter(ts_det[4000:5000], startF[4000:5000])
plt.ylim(5000, 10000)
