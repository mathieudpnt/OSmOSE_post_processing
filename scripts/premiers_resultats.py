from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

from def_func import (
    get_coordinates,
    json2df,
    add_season_period,
    add_recording_period
)

from premiers_resultats_utils import (
    load_parameters_from_yaml,
    scatter_detections,
    plot_hourly_detection_rate,
    single_plot,
    multilabel_plot,
    multiuser_plot,
    overview_plot,
    get_detection_perf
)

mpl.rcdefaults()
mpl.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["figure.figsize"] = [10, 6]

# %% load parameters from the YAML file
yaml_file = Path(r".\scripts\yaml_example.yaml")

df_detections, time_bin, annotators, labels, fmax, datetime_begin, datetime_end, _ = (
    load_parameters_from_yaml(file=yaml_file)
)

json = Path(r"\path\to\json")
metadatax = json2df(json_path=json)

# %% Overview plots
overview_plot(df=df_detections)
plt.show()

# %% Single seasonality plot
single_plot(df=df_detections, metadata=metadatax, season_bar=True)
add_season_period()  # add seasons to plot
add_recording_period(metadatax)  # add recording periods
plt.show()

# %% Single diel pattern plot (scatter raw detections)
lat, lon = get_coordinates()
scatter_detections(df=df_detections, lat=lat, lon=lon)
plt.show()

# %% Single diel pattern plot (Hourly detection rate)
plot_hourly_detection_rate(df=df_detections, lat=lat, lon=lon)
plt.show()

# %% Multilabel plot
multilabel_plot(df=df_detections)
plt.show()

# %% Multi-user plot
get_detection_perf(df=df_detections)
multiuser_plot(df=df_detections)
plt.show()