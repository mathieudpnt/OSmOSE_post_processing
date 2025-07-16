from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import Timedelta, Timestamp
from pandas.tseries import frequencies

from post_processing.premiers_resultats_utils import load_parameters_from_yaml
from post_processing.dataclass.data_aplose import DataAplose

mpl.rcdefaults()
plt.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["figure.figsize"] = [14, 6]

#%% Load data

yaml_file = Path(r"C:\Users\dupontma2\AppData\Roaming\JetBrains\PyCharm2024.3\scratches\post_processing\yaml.yml")
df = load_parameters_from_yaml(file=yaml_file)
data = DataAplose(df)

#%% Overview

data.overview()
plt.tight_layout()
plt.show()

#%% Single plot

fig, ax = plt.subplots(1,1)

# bins = frequencies.to_offset('1d')
bins = Timedelta('1D')
ticks = frequencies.to_offset('1W')
fmt = "%d %b"

ax = data.set_ax(ax=ax, bin_size=bins, xticks_res=ticks, date_format=fmt)
data.plot(annotator=['eledu', 'lleboul',], label='Antarctic blue whale song', ax=ax)
ax.set_xlim([Timestamp('July 2022'), Timestamp('September 2022')])

plt.tight_layout()
plt.show()

#%% Multi plot

fig, (ax0, ax1, ax2) = plt.subplots(3,1)

bins = Timedelta('1d')
ticks = frequencies.to_offset('BMS')
fmt = "%b %y"

ax0 = data.set_ax(ax=ax0, bin_size=bins, xticks_res=ticks, date_format=fmt)
data.plot(annotator=['eledu', 'lleboul'], label='Antarctic blue whale song', ax=ax0)

data.copy_ax(source_ax=ax0, target_ax=ax1)
data.plot(annotator=['eledu', 'lleboul'], label='Ind 42 Hz', ax=ax1)

data.copy_ax(source_ax=ax0, target_ax=ax2)
data.plot(annotator=['eledu', 'lleboul'], label='LF 8 sec pulse', ax=ax2)

lim = []
for ax in (ax0, ax1, ax2):
    lim.append(ax.get_ylim()[-1])

for ax in (ax0, ax1, ax2):
    ax.set_xlim([Timestamp('July 2022'), Timestamp('September 2022')])
    ax.set_ylim([0, int(max(lim))])
    ax.set_yticks(range(0, int(max(lim)), 4))

plt.tight_layout()
plt.show()

#%% Single diel pattern plot (scatter raw detections)

fig, ax = plt.subplots(1,1)

ticks = frequencies.to_offset('1W')
fmt = "%d %b"

ax = data.set_ax(ax=ax, xticks_res=ticks, date_format=fmt)
data.scatter_detections(ax=ax, annotator='lleboul', label='Antarctic blue whale song')
ax.set_xlim([Timestamp('July 2022'), Timestamp('September 2022')])

plt.tight_layout()
plt.show()