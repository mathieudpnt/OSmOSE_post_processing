from pathlib import Path
import pandas as pd
import yaml
from hydra_zen import instantiate


def read_yaml(file_path: Path):
    with open(file_path, "r") as stream:
        return yaml.safe_load(stream)


def load_parameters_from_yaml():
    """Loads parameters from scriptsscripts\premier_resultats_parameters.yaml"""

    # Read YAML file
    parameters = read_yaml(Path(r".\results\premiers_resultats_parameters.yaml"))

    # Initialize an empty dataframe
    df = pd.DataFrame()

    # Iterate over the files and their associated parameters in YAML
    for file, param_values in parameters.items():
        # Create a hydra-zen config from the YAML data
        Config = make_config(**param_values)

        # Instantiate the config as an object and pass it to your function
        config_instance = instantiate(Config)

        # Pass parameters to `sort_detections` function
        df = pd.concat([df, sort_detections(**config_instance)], ignore_index=True)

    time_bin = list(set(df["end_time"]))
    fmax = list(set(df["end_frequency"]))
    annotators = sorted(list(set(df["annotator"])))
    labels = sorted(list(set(df["annotation"])))
    tz_data = [df["start_datetime"].iloc[0].tz]

    datetime_begin = df["start_datetime"].iloc[0]
    datetime_end = df["end_datetime"].iloc[-1]

    return df, time_bin, annotators, labels, fmax, datetime_begin, datetime_end, tz_data


# %% import matplotlib as mpl

from utils.def_func import get_coordinates

from premiers_resultats_utils import (
    load_parameters_from_yaml,
    scatter_detections,
    plot_hourly_detection_rate,
    single_plot,
    multilabel_plot,
    multiuser_plot,
    overview_plot,
)

mpl.rcdefaults()
mpl.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["figure.figsize"] = [10, 8]

# %% load parameters from the YAML file
df_detections, time_bin, annotators, labels, fmax, datetime_begin, datetime_end, _ = (
    load_parameters_from_yaml()
)
print(
    f"\ntime_bin: {time_bin}\nfmax: {fmax}\nannotators: {annotators}\nlabels: {labels}"
)

# %% Overview plots
overview_plot(df_detections)

# %% Single seasonality plot
single_plot(df_detections)

# %% Single diel pattern plot (scatter raw detections)
lat, lon = get_coordinates()
scatter_detections(df=df_detections, lat=lat, lon=lon, date_format="%d/%m")

# %% Single diel pattern plot (Hourly detection rate)
plot_hourly_detection_rate(df=df_detections, lat=lat, lon=lon, date_format="%d/%m")

# %% Multilabel plot
multilabel_plot(df_detections)

# %% Multi-user plot
multiuser_plot(df_detections)
