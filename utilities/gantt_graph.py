import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
import datetime as dt

##### DATA #####
data = pd.read_excel('L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/APOCADO - Suivi déploiements.xlsx', skiprows=[0])

data = data.rename(columns={'N° campagne': 'Campagne', 'ID Soundtrap': 'Instrument', 'Date début campagne': 'Start', 'Date fin campagne': 'End'})
data['Instrument'] = data['Instrument'].astype(str)
data = data[~((data['Campagne'] == 7) & (data['N° déploiement'] == 1))]
data = data[~((data['Campagne'] == 4) & (data['N° déploiement'] == 9) & (data['Instrument'] == '336363566'))]
data = data.reset_index(drop=True)


data['Start2'] = [pd.Timestamp.combine(data['Date début déploiement'][i].date(), data['Heure début déploiement'][i]) for i in range(len(data))]
data['End2'] = [pd.Timestamp.combine(data['Date fin déploiement'][i].date(), data['Heure fin déploiement'][i]) for i in range(len(data))]
#%%
# project start date
proj_start = data['Start2'].min()

# number of days from project start to task start
data['start_num'] = (data['Start2']-proj_start).dt.days

# number of days from project start to end of tasks
data['end_num'] = (data['End2']-proj_start).dt.days

# days between start and end of each task
data['days_start_to_end'] = data['end_num'] - data['start_num']

# create a column with the color for each element of the arg variable
arg='Campagne'
list_arg = sorted(list(data[arg].unique()))
colors = [mpl.colors.rgb2hex(i) for i in mpl.cm.tab10(range(len(list_arg)))]
c_dict = dict(zip(list_arg, colors))
data['color'] = data[arg].map(c_dict)

arg2 = 'Instrument'
data = data.sort_values(arg2, ascending=False).reset_index(drop=True)


##### PLOT #####
fig, (ax, ax1) = plt.subplots(2, figsize=(16,6), gridspec_kw={'height_ratios':[6, 1]}, facecolor='#36454F')
ax.set_facecolor('#36454F')
ax1.set_facecolor('#36454F')


# bars
ax.barh(data[arg2], data.days_start_to_end, left=data.start_num, color=data.color, alpha=0.8)# ticks
ax.set_xlim(-10, data.end_num.max())

# #labels
# ylab_dict = dict(zip(list(data[arg2].unique()), range(len(data[arg2].unique()))))
# for idx, row in data.iterrows():
#     row_lab = ylab_dict[row[arg2]]
#     ax.text(row.start_num-0.1, row_lab, row[arg], va='center', ha='right', alpha=0.8, color='w')

# grid lines
ax.set_axisbelow(True)
ax.xaxis.grid(color='k', linestyle='dashed', alpha=0.4, which='both')

# ticks
import locale
locale.setlocale(locale.LC_TIME,'')
first_of_month = pd.date_range(start=data.Start.min(), end=data.End.max(), freq='MS')
xticks = [(date - proj_start).days+1 for date in first_of_month]
xticks_labels = [date.strftime("%B %y") for date in first_of_month]

ax.set_xticks(xticks)
ax.set_xticklabels(xticks_labels, color='w')
ax.tick_params(axis='both', colors='w')

# spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_color('w')
ax.spines['left'].set_color('w')

plt.suptitle('APOCADO', color='w')

#seasons
    #printemps : 21/03 -> jour 79
    #été : 21/06 -> jour 171
    #automne : 23/09 -> jour 266
    #hiver : 21/12 -> jour 355

# Add vertical lines for the first day of each season
season_dates = pd.date_range(start=data.Start.min(), end=data.End.max(), freq='D')
season_days = [abs(date - proj_start).days for date in season_dates]
for day in season_days:
    if season_dates[day].timetuple().tm_yday in [80, 172, 264, 355]:
        ax.axvline(day, color='white', linestyle='--', linewidth=1, alpha=0.5)
        if season_dates[day].timetuple().tm_yday == 80:
            ax.text(day, ax.get_ylim()[1] * 1.02, 'Printemps', color='white', ha='left', va='bottom')
        if season_dates[day].timetuple().tm_yday == 172:
            ax.text(day, ax.get_ylim()[1] * 1.02, 'Été', color='white', ha='left', va='bottom')
        if season_dates[day].timetuple().tm_yday == 264:
            ax.text(day, ax.get_ylim()[1] * 1.02, 'Automne', color='white', ha='left', va='bottom')
        if season_dates[day].timetuple().tm_yday == 355:
            ax.text(day, ax.get_ylim()[1] * 1.02, 'Hiver', color='white', ha='left', va='bottom')

pd.Timestamp('2022-09-01').timetuple().tm_year
acqui_ST = pd.Timestamp('2022-09-01').timetuple().tm_yday - proj_start.timetuple().tm_yday
ax.axvline(x=acqui_ST, ymin=0, ymax=0.15, color='w')
ax.text(acqui_ST-2, ax.get_ylim()[1] * 0, 'Acquisition\ndes ST400HF', color='w', ha='right', va='bottom')

# test_date = (pd.Timestamp('2023-01-01').timetuple().tm_yday+365) - proj_start.timetuple().tm_yday
# ax.axvline(x=test_date, ymin=0, ymax=0.15, color='r')
# ax.text(test_date-2, ax.get_ylim()[1] * 0, 'test', color='r', ha='right', va='bottom')


##### LEGENDS #####
legend_elements = [Patch(facecolor=c_dict[dep], label=dep) for dep in sorted(list(data[arg].unique()))]
legend = ax1.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), frameon=False, title = arg)
plt.setp(legend.get_texts(), color='w')
legend.get_title().set_color('w')

# clean second axis
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])



