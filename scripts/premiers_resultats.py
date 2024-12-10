from pathlib import Path
import matplotlib as mpl

from utils.def_func import get_coordinates, reshape_timebin

from premiers_resultats_utils import (
    load_parameters_from_yaml,
    scatter_detections,
    plot_hourly_detection_rate,
    single_plot,
    multilabel_plot,
    multiuser_plot,
    overview_plot,
    get_detection_perf,
)

mpl.rcdefaults()
mpl.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["figure.figsize"] = [10, 6]

# %% load parameters from the YAML file
yaml_file = Path(r".\scripts\yaml_example.yaml")
df_detections, time_bin, annotators, labels, fmax, datetime_begin, datetime_end, _ = (
    load_parameters_from_yaml(file=yaml_file)
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
get_detection_perf(df=df_detections, annotators=annotators, annotations=labels)
multiuser_plot(df_detections[df_detections["annotator"] != "PAMGuard"])
