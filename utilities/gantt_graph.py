import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
import datetime as dt

# Load and format data
data = pd.read_excel('L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/APOCADO - Suivi déploiements.xlsm', skiprows=[0])
data = data[data['check heure Raven'] == 1].reset_index(drop=True)

data['datetime deployment'] = [pd.Timestamp.combine(data['date deployment'][i].date(), data['time deployment'][i]) for i in range(len(data))]
data['datetime recovery'] = [pd.Timestamp.combine(data['date recovery'][i].date(), data['time recovery'][i]) for i in range(len(data))]

data_gant = pd.DataFrame(columns=['campaign', 'recorder', 'dt_deployment', 'dt_recovery'])
campaign_list = list(set(data['campaign']))
for C in campaign_list:
    recorder_list = list(set(data[data['campaign'] == C]['ID recorder']))
    for R in recorder_list:
        date_beg = data[(data['campaign'] == C) & (data['ID recorder'] == R)]['datetime deployment'].min()
        date_end = data[(data['campaign'] == C) & (data['ID recorder'] == R)]['datetime recovery'].max()
        data_gant.loc[len(data_gant)] = [C, str(R), date_beg, date_end]

# create a column with the color for each element of the arg variable
arg = 'campaign'
list_arg = sorted(list(data_gant[arg].unique()))
colors = [mpl.colors.rgb2hex(i) for i in mpl.cm.tab20(range(len(list_arg)))]
c_dict = dict(zip(list_arg, colors))
data_gant['color'] = data_gant[arg].map(c_dict)

arg2 = 'recorder'
data_gant = data_gant.sort_values(arg2, ascending=False).reset_index(drop=True)

# %% Plot
plt.rcParams['font.size'] = 16
# fig, (ax, ax1) = plt.subplots(2, figsize=(32, 12), gridspec_kw={'height_ratios': [6, 1]}, facecolor='#36454F')
fig, (ax, ax1) = plt.subplots(2, figsize=(20, 8), gridspec_kw={'height_ratios': [6, 1]})
# ax.set_facecolor('#36454F')
# ax1.set_facecolor('#36454F')

# data to plot
for i in range(len(data_gant)):
    ax.barh(y=data_gant[arg2][i], width=(data_gant['dt_recovery'][i] - data_gant['dt_deployment'][i]), left=data_gant['dt_deployment'][i], color=data_gant.color[i], alpha=0.8)  # ticks
ax.set_xlim(data_gant['dt_deployment'].min() - dt.timedelta(days=7), data_gant['dt_recovery'].max() + dt.timedelta(days=7))

# labels
# ylab_dict = dict(zip(list(data_gant[arg2].unique()), range(len(data_gant[arg2].unique()))))
# for idx, row in data_gant.iterrows():
#     row_lab = ylab_dict[row[arg2]]
#     ax.text(row['dt_deployment']- dt.timedelta(hours=45), row_lab, row[arg], va='center', ha='right', alpha=0.8, color='w')

# grid lines
ax.set_axisbelow(True)
ax.xaxis.grid(color='k', linestyle='dashed', alpha=0.4, which='both')

# ticks
xticks = pd.date_range(start=data_gant['dt_deployment'].min(), end=data_gant['dt_recovery'].max(), freq='MS')
xticks_labels = [date.strftime("%m/%y") for date in xticks]
ax.set_xticks(xticks)
# ax.set_xticklabels(xticks_labels, color='w')
ax.set_xticklabels(xticks_labels)
ax.tick_params(axis='x', rotation=0)
# ax.tick_params(axis='both', colors='w', rotation=0)

# spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_color('w')
# ax.spines['left'].set_color('w')

plt.suptitle('APOCADO')
# plt.suptitle('APOCADO', color='w')

acqui_ST = pd.Timestamp('2022-09-01')
# ax.axvline(x=acqui_ST, ymin=0, ymax=0.15, color='w')
ax.axvline(x=acqui_ST, ymin=0, ymax=0.15, color='black')
# ax.text(acqui_ST - dt.timedelta(days=5), ax.get_ylim()[1] * 0, 'Acquisition\n ST400HF', color='w', ha='right', va='bottom')
ax.text(acqui_ST - dt.timedelta(days=5), ax.get_ylim()[1] * 0, 'Acquisition\n ST400HF', color='black', ha='right', va='bottom')

# seasons
# printemps : 21/03 -> jour 79
# été : 21/06 -> jour 171
# automne : 23/09 -> jour 266
# hiver : 21/12 -> jour 355

# Add vertical lines for the first day of each season
season_dates = pd.date_range(start=data['date deployment'].min(), end=data['date recovery'].max(), freq='D')
for date in season_dates:
    yday = date.timetuple().tm_yday
    year = date.timetuple().tm_year
    season_yday = [79, 171, 266, 355]
    if yday in season_yday:
        # ax.axvline(date, color='white', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(date, color='black', linestyle='--', linewidth=1, alpha=0.5)
        if yday == season_yday[0]:
            ax.text(date, ax.get_ylim()[1] * 1.02, f'Printemps {year}', color='black', ha='left', va='bottom')
        if yday == season_yday[1]:
            ax.text(date, ax.get_ylim()[1] * 1.02, f'Été {year}', color='black', ha='left', va='bottom')
        if yday == season_yday[2]:
            ax.text(date, ax.get_ylim()[1] * 1.02, f'Automne {year}', color='black', ha='left', va='bottom')
        if yday == season_yday[3]:
            ax.text(date, ax.get_ylim()[1] * 1.02, f'Hiver {year}', color='black', ha='left', va='bottom')

# legend
legend_elements = [Patch(facecolor=c_dict[dep], label=dep) for dep in sorted(list(data_gant[arg].unique()))]
legend = ax1.legend(handles=legend_elements, loc='upper center', ncols=len(legend_elements)//2, frameon=False, title=arg)
# plt.setp(legend.get_texts(), color='w')
plt.setp(legend.get_texts(), color='black')
# legend.get_title().set_color('w')
legend.get_title().set_color('black')

# clean second axis
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])

# font size
plt.rcParams['font.size'] = 18
