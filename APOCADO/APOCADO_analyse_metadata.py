import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from tqdm import tqdm
import numpy as np
from collections import Counter
import glob
import json
from scipy.stats import norm, shapiro, mannwhitneyu, ttest_ind
import seaborn as sns
import pickle
import matplotlib as mpl
from cycler import cycler
import pytz
import re

os.chdir(r"U:/Documents_U/Git/post_processing_detections")
from utilities.def_func import sorting_detections, get_season
from utilities.APOCADO_stat import stats_diel_pattern

mpl.rcdefaults()
mpl.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["figure.figsize"] = [10, 4]

# %%
data_load = "pickle"  # manual or pickle
detector = ["pamguard", "thalassa"]
arg = ["season", "net", "all"]
timebin = 60  # seconds
light_regime = ["Night", "Dawn", "Day", "Dusk"]

# %% Import csv deployments

deploy_path = r"L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/APOCADO - Suivi déploiements.xlsm"
deploy = pd.read_excel(deploy_path, skiprows=1)
deploy = deploy[deploy["final selection"] == 1].reset_index(drop=True)

deploy["datetime deployment"] = [
    pd.Timestamp.combine(deploy["date deployment"][i], deploy["time deployment"][i])
    for i in range(len(deploy))
]
deploy["datetime recovery"] = [
    pd.Timestamp.combine(deploy["date recovery"][i], deploy["time recovery"][i])
    for i in range(len(deploy))
]
deploy["duration"] = [
    deploy["datetime recovery"][i] - deploy["datetime deployment"][i]
    for i in range(len(deploy))
]
deploy["season_y"] = [get_season(i) for i in deploy["datetime deployment"]]
deploy["season"] = [i.split(" ")[0] for i in deploy["season_y"]]
deploy["year"] = [int(s[-4:]) for s in deploy["season_y"]]
deploy["species"] = deploy["species"].apply(
    lambda x: x.lower() if isinstance(x, str) else x
)

# result deployments
print("\n# Observation effort #")
t_tot = (deploy["duration"] / deploy["recorder number"]).sum().total_seconds() / 3600
print(f"-data collected: {t_tot:.0f} h")
t_tot2 = round(deploy["duration"].sum().total_seconds() / 3600)
print(f"-data collected (with ST appaired): {t_tot2:.0f} h")

print("\n# Observation effort per season #")
list_season = ["spring", "summer", "autumn", "winter"]
for s in list_season:
    t = (
        deploy[deploy["season"] == s]["duration"]
        / deploy[deploy["season"] == s]["recorder number"]
    ).sum().total_seconds() / 3600
    print(f"-{s}: {100 * (t / t_tot):2.0f}% - {t:4.0f} h")

print("\n# Observation effort per season & per year #")
list_year = sorted(list(set(deploy["year"])))
for y in list_year:
    list_season_y = list(dict.fromkeys(deploy[(deploy["year"] == y)]["season_y"]))
    for sy in list_season_y:
        t = (
            deploy[deploy["season_y"] == sy]["duration"]
            / deploy[deploy["season_y"] == sy]["recorder number"]
        ).sum().total_seconds() / 3600
        print(f"-{sy}: {100 * (t / t_tot):2.0f}% - {t:4.0f} h")

print("\n# Observation effort per net type #")
net_type = sorted(list(set(deploy["net"])))
for net in net_type:
    duration_net = (
        deploy[deploy["net"] == net]["duration"]
        / deploy[deploy["net"] == net]["recorder number"]
    ).sum().total_seconds() / 3600
    print(f"-{net}: {duration_net:2.0f}h")

print("\n# Observation effort per net type & per season #")
net_type = sorted(list(set(deploy["net"])))
for net in net_type:
    list_season = list(dict.fromkeys(deploy[(deploy["net"] == net)]["season"]))
    for s in list_season:
        duration_net = (
            deploy[(deploy["net"] == net) & (deploy["season"] == s)]["duration"]
            / deploy[(deploy["net"] == net) & (deploy["season"] == s)][
                "recorder number"
            ]
        ).sum().total_seconds() / 3600
        print(f"-{s}/{net}: {duration_net:2.0f}h")

print("\n# Observation effort per net type & per season #")
list_season = ["spring", "summer", "autumn", "winter"]
net_type = ["Droit", "Trémail"]
for season in list_season:
    for net in net_type:
        duration_net = (
            deploy[(deploy["net"] == net) & (deploy["season"] == season)]["duration"]
            / deploy[(deploy["net"] == net) & (deploy["season"] == season)][
                "recorder number"
            ]
        ).sum().total_seconds() / 3600
        print(f"-{season}/{net}: {duration_net:2.0f}h")


print("\n# Mean duration of a deployment per net type #")
for net in net_type:
    duration_net_mean = (
        deploy[deploy["net"] == net]["duration"]
        / deploy[deploy["net"] == net]["recorder number"]
    ).mean().total_seconds() / 3600
    duration_net_std = (
        deploy[deploy["net"] == net]["duration"]
        / deploy[deploy["net"] == net]["recorder number"]
    ).std().total_seconds() / 3600
    print(f"{net}: {duration_net_mean:.0f}h +/- {duration_net_std:.0f}h")

# distribution of net lengths
# if 2 ST are present on the net, the duration is divided by 2
net_length = sorted(list(set(deploy["net length"])))
net_length_str = [str(elem) for elem in net_length]
net_length_distrib = [
    (
        deploy[deploy["net length"] == L]["duration"]
        / deploy[deploy["net length"] == L]["recorder number"]
    )
    .sum()
    .total_seconds()
    / 3600
    for L in net_length
]
print("\n# Duration per net length #")
for length, yi in zip(net_length, net_length_distrib):
    print(f"{length}m: {round(yi)}h")

# distribution of deployment durations
duration_distrib = [t.total_seconds() / 3600 for t in deploy["duration"]]

fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [4, 3]})
axs[0].hist(
    duration_distrib,
    bins=range(0, int(1.05 * max(duration_distrib)), 2),
    edgecolor="black",
    linewidth=0.5,
    zorder=2,
)
axs[0].set_xlabel("Hours")
axs[0].set_ylabel("Deployment number")
axs[0].set_title("Distribution of deployment durations")
axs[0].grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
axs[1].set_title("Distribution of net lengths")
axs[1].bar(net_length_str, net_length_distrib, edgecolor="black", linewidth=1, zorder=2)
axs[1].set_ylabel("Data collected [hours]")
axs[1].set_xlabel("Net length [m]")
axs[1].grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
plt.tight_layout()
plt.show()

# distribution of deployments per species
species_set = set(deploy["species"])
unique_species = set()
for entry in species_set:
    if entry and isinstance(entry, str):
        species = re.split(r"[\n ]", entry)
        unique_species.update([sp.lower() for sp in species])
unique_species_list = list(unique_species)

species_result = []
for i in range(len(deploy)):
    line = []
    if not pd.isna(deploy["species"][i]):
        [
            line.append(1) if u in deploy["species"][i] else line.append(0)
            for u in unique_species_list
        ]
        species_result.append(line)
    else:
        species_result.append([0] * len(unique_species_list))

species_result = pd.DataFrame(species_result, columns=unique_species_list)

species_result.sum().plot(
    kind="bar", zorder=2, edgecolor="black", linewidth=1, figsize=(6, 2)
)
plt.xlabel("Species")
plt.ylabel("Deployment number")
plt.title("Distribution of deployment species")
plt.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
plt.show()

# %% Load all metadata

match data_load:

    case "pickle":
        with open(
            r"L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO\data_timebin60s.pkl",
            "rb",
        ) as f:
            data = pickle.load(f)

    case "manual":
        data = []
        path_acoustock = [
            r"L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO",
            r"Y:\Bioacoustique\APOCADO2",
            r"Z:\Bioacoustique\DATASETS\APOCADO3",
        ]

        list_json = [
            file_path
            for path in path_acoustock
            for file_path in glob.glob(
                os.path.join(path, "**/metadata.json"), recursive=True
            )
        ]

        for i in tqdm(range(len(list_json)), desc="Scanning metadata files"):
            r_file = open(list_json[i], "r")
            data.append(json.load(r_file))
            r_file.close()
        data = pd.DataFrame.from_dict(data)

        df_detections_pamguard, df_detections_thalassa = [], []
        for i in tqdm(range(len(data)), desc="DataFrame creation"):
            f1 = data["path pamguard"][i]
            f2 = data["path thalassa"][i]
            tz = (
                data["datetime deployment"][i][-5:-2]
                + ":"
                + data["datetime deployment"][i][-2:]
            )
            # tz = data['datetime deployment'][i].tz
            ts = (
                pd.read_csv(
                    data["path segment timestamp"][i], parse_dates=["timestamp"]
                )
                .drop_duplicates()
                .reset_index(drop=True)["timestamp"]
            )

            df_detections_file1, _ = sorting_detections(
                file=f1,
                tz=tz,
                timebin_new=timebin,
                timestamp=ts,
                date_begin=data["datetime deployment"][i],
                date_end=data["datetime recovery"][i],
            )
            df_detections_file2, _ = sorting_detections(
                file=f2,
                tz=tz,
                timebin_new=timebin,
                timestamp=ts,
                date_begin=data["datetime deployment"][i],
                date_end=data["datetime recovery"][i],
            )
            df_detections_pamguard.append(df_detections_file1)
            df_detections_thalassa.append(df_detections_file2)

        data["df pamguard"] = df_detections_pamguard
        data["df thalassa"] = df_detections_thalassa

        data["platform"] = [
            "C" + str(c) + "D" + str(d)
            for c, d in zip(data["campaign"], data["deployment"])
        ]
        data["datetime deployment"] = [
            pd.to_datetime(d) for d in data["datetime deployment"]
        ]
        data["datetime recovery"] = [
            pd.to_datetime(d) for d in data["datetime recovery"]
        ]
        data["season_y"] = [get_season(i) for i in data["datetime deployment"]]
        data["season"] = [i.split(" ")[0] for i in data["season_y"]]
        data["year"] = [int(s[-4:]) for s in data["season_y"]]
        data["duration"] = [pd.Timedelta(d) for d in data["duration"]]
        data["deployment"] = [
            data["platform"][i] + " ST" + str(data["recorder"][i])
            for i in range(len(data))
        ]
        data["species"] = data["species"].apply(
            lambda x: x.lower() if isinstance(x, str) else x
        )

        for d in detector:
            data[f"detection rate {d}"] = [
                (
                    len(data[f"df {d}"][i])
                    / (data["duration"][i].total_seconds() / 86400)
                )
                / (86400 / timebin)
                for i in range(len(data))
            ]

# exclude data rows if deployment not selected in csv
data = data.loc[data["deployment"].isin(deploy["ID deployment"])].reset_index(drop=True)

# %% Save data

# save the DataFrame to a pickle file
with open(
    os.path.join(
        r"L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO",
        f"data_timebin{timebin}s.pkl",
    ),
    "wb",
) as f:
    pickle.dump(data, f)

# %% hourly detection rate and double check selection

hourly_detection_rate = []
for i in tqdm(range(len(data))):

    timestamp_range_begin = data["datetime deployment"][i].replace(
        minute=0, second=0, microsecond=0
    )
    if (
        (data["datetime recovery"][i].minute != 0)
        or (data["datetime recovery"][i].second != 0)
        or (data["datetime recovery"][i].microsecond != 0)
    ):
        timestamp_range_end = data["datetime recovery"][i].replace(
            hour=data["datetime recovery"][i].hour + 1,
            minute=0,
            second=0,
            microsecond=0,
        )
    else:
        timestamp_range_end = data["datetime recovery"][i]

    timestamp_range = pd.date_range(
        start=timestamp_range_begin, end=timestamp_range_end, freq="1h"
    ).tolist()

    test = pd.read_csv(data["path segment timestamp"][i], parse_dates=["timestamp"])
    timestamp_range2 = test["timestamp"][::360]

    df_hourly = pd.DataFrame()
    for d in detector:
        start_series = pd.Series(data[f"df {d}"][i]["start_datetime"])
        count_per_interval = start_series.groupby(
            pd.cut(start_series, timestamp_range, include_lowest=True, right=False),
            observed=True,
        ).count()
        coef_norm_begin = (
            1
            - ((data["datetime deployment"][i] - timestamp_range_begin).seconds // 60)
            / 60
        )
        coef_norm_end = (
            1
            - ((timestamp_range_end - data["datetime recovery"][i]).seconds // 60) / 60
        )
        count_per_interval.iloc[0] = (
            round(count_per_interval.iloc[0] / coef_norm_begin)
            if count_per_interval.iloc[0] / coef_norm_begin < 60
            else 60
        )
        count_per_interval.iloc[-1] = (
            round(count_per_interval.iloc[-1] / coef_norm_end)
            if count_per_interval.iloc[-1] / coef_norm_end < 60
            else 60
        )
        count_per_interval = count_per_interval.reindex(
            timestamp_range[:-1], fill_value=0
        ).to_frame()
        count_per_interval.rename(
            columns={"start_datetime": f"{d} positive minute"}, inplace=True
        )
        count_per_interval[f"{d} coverage"] = (
            [coef_norm_begin] + [1] * (len(count_per_interval) - 2) + [coef_norm_end]
        )
        count_per_interval[f"{d} hourly detection rate"] = list(
            count_per_interval[f"{d} positive minute"].values / 60
        )
        df_hourly = pd.concat([df_hourly, count_per_interval], axis=1)

    df_hourly["ID"] = data["platform"][i] + "_ST" + data["recorder"][i]
    hourly_detection_rate.append(df_hourly)
data["hourly detection rate"] = hourly_detection_rate

df_hourly_all = pd.DataFrame()
for i in range(len(data)):
    df_hourly_all = pd.concat([df_hourly_all, data["hourly detection rate"][i]])

# bins = range(10, 91, 5)
bins = range(0, 101, 5)
for d in detector:
    sns.histplot(
        100 * df_hourly_all[f"{d} hourly detection rate"],
        bins=bins,
        kde=False,
        label=f"{d}",
    )
plt.xlabel("Percentage")
plt.ylabel("Density")
plt.title("Hourly detection rate distribution")
plt.legend()
plt.show()

# double_check_selection = df_hourly_all[(df_hourly_all['pamguard hourly detection rate'] == 1) & (df_hourly_all['pamguard coverage'] == 1) & (df_hourly_all['thalassa hourly detection rate'] == 1) & (df_hourly_all['thalassa coverage'] == 1)].sample(n=10).sort_values(by='start_datetime')
# lim_low = 0.2
# lim_high = 0.8
# double_check_selection = df_hourly_all[(df_hourly_all['pamguard hourly detection rate'] <= lim_high) &
#                                        (df_hourly_all['pamguard hourly detection rate'] >= lim_low ) &
#                                        (df_hourly_all['pamguard coverage'] == 1) &
#                                        (df_hourly_all['thalassa hourly detection rate'] <= lim_high) &
#                                        (df_hourly_all['thalassa hourly detection rate'] >= lim_low ) &
#                                        (df_hourly_all['thalassa coverage'] == 1)].sample(n=10).sort_values(by='start_datetime')
# print(f"\nSelection for double check : \n\n{double_check_selection['ID']}")

# %% filtering data for following analysis
"""
here the data is treated differently according to which detector is used i.e. which signal are studied
for whistles, appaired recorders are trated separately if the length on the net  is >500m
for clicks, appaired recorders are trated separately if the length on the net  is >200m
In the other case, only one of the appaired recorder is considered
"""

detector = ["pamguard", "thalassa"]

filtered_df_1 = data[(data["recorder number"] == 1)]  # data with 1 ST
# filtered_df_1 = data[(data['recorder number'] == 1) & (data['duration'] >= pd.Timedelta(hours=17)) & (data['duration'] <= pd.Timedelta(hours=30))].reset_index(drop=True)
filtered_data = {}

for d in detector:

    if d == "thalassa":
        colorplot = "teal"
        filtered_df_2 = data[
            (data["recorder number"] == 2) & (data["net length"] <= 200)
        ]
        sub_df_2 = (
            filtered_df_2.groupby("platform").first().reset_index()
        )  # only the first ST
        filtered_df_3 = data[
            (data["recorder number"] == 2) & (data["net length"] > 200)
        ]
        filtered_data[f"{d}"] = pd.concat(
            [filtered_df_1, sub_df_2, filtered_df_3]
        ).reset_index(drop=True)

    elif d == "pamguard":
        colorplot = "darkorange"
        filtered_df_2 = data[
            (data["recorder number"] == 2) & (data["net length"] <= 500)
        ]
        sub_df_2 = (
            filtered_df_2.groupby("platform").first().reset_index()
        )  # only the first ST
        filtered_df_3 = data[
            (data["recorder number"] == 2) & (data["net length"] > 500)
        ]
        filtered_data[f"{d}"] = pd.concat(
            [filtered_df_1, sub_df_2, filtered_df_3]
        ).reset_index(drop=True)
# %% distribution of deployement detection rates

for d in detector:

    mean_filter_df = filtered_data[f"{d}"][f"detection rate {d}"].mean()
    std_filter_df = filtered_data[f"{d}"][f"detection rate {d}"].std()
    max_filter_df = filtered_data[f"{d}"][f"detection rate {d}"].max()
    min_filter_df = filtered_data[f"{d}"][f"detection rate {d}"].min()
    print(
        f"\n{d} / all season: {mean_filter_df * 100:.0f}% +/- {100 * std_filter_df:.0f}%, max={100 * max_filter_df:.0f}%, min={100 * min_filter_df:.0f}%, N={len(filtered_data[f'{d}'])}"
    )

    fig, ax = plt.subplots()
    ax.set_axisbelow(True)  # Set grid behind plot
    plt.grid(alpha=0.5)
    hist, bins, _ = plt.hist(
        100 * filtered_data[f"{d}"][f"detection rate {d}"],
        bins=np.arange(0, 101, 5),
        density=False,
        edgecolor="black",
    )
    # Fit a normal distribution to the data
    mu, std = norm.fit(100 * filtered_data[f"{d}"][f"detection rate {d}"])
    # Create range for the smoothed line
    x_range = np.linspace(0, 100, 1000)
    # Calculate the Gaussian curve
    pdf = norm.pdf(x_range, mu, std)
    # Plot the Gaussian curve
    # plt.plot(x_range, (pdf / max(pdf)) * max(hist), color='teal')
    plt.xticks(np.arange(0, 100 + 1, 10))
    plt.xlabel("Positive minute detection rate / day")
    plt.ylabel("Deployment number")
    plt.title(f"Distribution of deployment detection rate\ndetector: {d}")
    plt.show()

    arg = "season"
    for i in list(dict.fromkeys(filtered_data[f"{d}"][f"{arg}"])):

        # for i in unique_species_list:

        if i == "spring":
            colorplot = "#59D955"
        elif i == "summer":
            colorplot = "#F2E205"
        elif i == "autumn":
            colorplot = "#F28B66"
        elif i == "winter":
            colorplot = "#4694A6"

        # species
        # N = filtered_data[f'{d}'][f'{arg}'].str.contains(i).sum()
        # mean = filtered_data[f'{d}'][filtered_data[f'{d}'][f'{arg}'].str.contains(i, na=False)][f'detection rate {d}'].mean()
        # std = filtered_data[f'{d}'][filtered_data[f'{d}'][f'{arg}'].str.contains(i, na=False)][f'detection rate {d}'].std()
        # max_stat = filtered_data[f'{d}'][filtered_data[f'{d}'][f'{arg}'].str.contains(i, na=False)][f'detection rate {d}'].max()
        # min_stat = filtered_data[f'{d}'][filtered_data[f'{d}'][f'{arg}'].str.contains(i, na=False)][f'detection rate {d}'].min()

        N = len(filtered_data[f"{d}"][filtered_data[f"{d}"][f"{arg}"] == i])
        mean = filtered_data[f"{d}"][filtered_data[f"{d}"][f"{arg}"] == i][
            f"detection rate {d}"
        ].mean()
        std = filtered_data[f"{d}"][filtered_data[f"{d}"][f"{arg}"] == i][
            f"detection rate {d}"
        ].std()
        max_stat = filtered_data[f"{d}"][filtered_data[f"{d}"][f"{arg}"] == i][
            f"detection rate {d}"
        ].max()
        min_stat = filtered_data[f"{d}"][filtered_data[f"{d}"][f"{arg}"] == i][
            f"detection rate {d}"
        ].min()

        print(
            f"\t{d} / {i}: {mean * 100:.0f}% +/- {100 * std:.0f}%, max={100 * max_stat:.0f}%, min={100 * min_stat:.0f}%, N={N}"
        )

        fig, ax = plt.subplots()
        ax.set_axisbelow(True)  # Set grid behind plot
        plt.grid(alpha=0.5)
        # hist, bins, _ = plt.hist(100 * filtered_data[f'{d}'][filtered_data[f'{d}'][f'{arg}'] == i][f'detection rate {d}'], bins=np.arange(0, 101, 5), density=False, edgecolor='black', color=colorplot)
        hist, bins, _ = plt.hist(
            100
            * filtered_data[f"{d}"][
                filtered_data[f"{d}"][f"{arg}"].str.contains(i, na=False)
            ][f"detection rate {d}"],
            bins=np.arange(0, 101, 5),
            density=False,
            edgecolor="black",
            color=colorplot,
        )
        # Fit a normal distribution to the data
        mu, std = norm.fit(
            100
            * filtered_data[f"{d}"][filtered_data[f"{d}"][f"{arg}"] == i][
                f"detection rate {d}"
            ]
        )
        # Create range for the smoothed line
        x_range = np.linspace(0, 100, 1000)
        # Calculate the Gaussian curve
        pdf = norm.pdf(x_range, mu, std)
        # Plot the Gaussian curve
        # plt.plot(x_range, (pdf / max(pdf)) * max(hist), color='teal')
        plt.xticks(np.arange(0, 100 + 1, 10))
        plt.xlabel("Positive minute detection rate / day")
        plt.ylabel("Deployment number")
        plt.title(
            f"Distribution of deployment detection rate\n{arg}: {i} - detector: {d} - N={N}"
        )
        plt.show()

# %% distribution of the period with most detections

"""
Distribution of all the detection dataframes most populated time period,
first, each detection dataframe is divided into n periods of equal durations
Then the distribution of the period with the most detections for each df is plotted
"""
n_periods = 10
detector = ["thalassa"]
# arg = ['season', 'net', 'all']
arg = ["all"]

for d in detector:

    # data2 = filtered_data[f'{d}'][(filtered_data[f'{d}']['duration'] >= pd.Timedelta(hours=17)) & (filtered_data[f'{d}']['duration'] <= pd.Timedelta(hours=28))].reset_index(drop=True)
    data2 = filtered_data[f"{d}"]
    print(
        f"{d}: {100 * len(data2) / len(filtered_data[f'{d}']):.0f}% of deployments selected ({len(data2)}/{len(filtered_data[f'{d}'])} deployments)"
    )

    # # distribution of deployment durations
    # test = [t.total_seconds() / 3600 for t in data2['duration']]
    # fig, ax = plt.subplots(figsize=(18, 9), facecolor='#36454F')
    # ax.set_facecolor('#36454F')
    # ax.spines['bottom'].set_color('w')
    # ax.spines['left'].set_color('w')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.tick_params(axis='both', colors='w', labelsize=14)
    # # ax.hist(test, bins=range(0, int(1.05 * max(test)), 2))
    # ax.hist(test, bins=range(0, 151, 2))
    # plt.xlabel('Hours', fontsize=16, color='w')
    # plt.ylabel('Deployment number', fontsize=16, color='w')
    # plt.title('Distribution of deployement durations', color='w', fontsize=18)
    # ax.grid(axis='y', linestyle='--', color='w', alpha=0.5)
    # plt.show()

    for a in arg:
        if a != "all":
            filter_list = list(dict.fromkeys(data2[f"{a}"]))

            for f in filter_list:
                data3 = data2[data2[f"{a}"] == f]

                N = len(data3)
                rank_max, rank_min = [], []
                for i in range(len(data3)):
                    data_test = data3.iloc[i]
                    df_detections = data_test[f"df {d}"]
                    data_histo = df_detections["start_datetime"]
                    periods = pd.date_range(
                        start=data_test["datetime deployment"],
                        end=data_test["datetime recovery"],
                        freq=str(int(data_test["duration"].total_seconds() / n_periods))
                        + "s",
                    )
                    detect_periods, _ = np.histogram(data_histo, periods)
                    rank_max.append(np.argmax(detect_periods) + 1)
                    rank_min.append(np.argmin(detect_periods) + 1)

                total = len(data3)
                value_max_counts = Counter(rank_max)
                value_min_counts = Counter(rank_min)

                # Fill in the counts for null values
                for value in range(1, n_periods + 1, 1):
                    if value not in value_max_counts:
                        value_max_counts[value] = 0
                    if value not in value_min_counts:
                        value_min_counts[value] = 0

                value_max_counts = dict(sorted(value_max_counts.items()))
                value_min_counts = dict(sorted(value_min_counts.items()))

                count_max = [max_c for max_c in list(value_max_counts.values())]
                count_min = [min_c for min_c in list(value_min_counts.values())]
                values = list(value_max_counts.keys())

                fig, ax = plt.subplots()
                # bar plots
                ax.bar(values, count_max, align="center", edgecolor="black")
                # grid
                ax.yaxis.grid(color="gray", linestyle="--")
                # x-label
                ax.set_xlabel("Periods")
                # title
                ax.set_title(
                    f"Distribution of the period with most detections\n{a}: {f} - detector: {d} - N={N}"
                )
                # ticks
                ax.set_xticks(values)
                # ax.set_yticks(range(0, max(count_max) + 1, 3))
                ax.set_yticks(range(0, 20, 2))
                plt.ylim(0, max(count_max) + 2)
                # plt.ylim(0, 20)
                plt.show()

        else:
            data3 = data2

            N = len(data3)
            rank_max, rank_min = [], []
            for i in range(len(data3)):
                data_test = data3.iloc[i]
                df_detections = data_test[f"df {d}"]
                data_histo = df_detections["start_datetime"]
                periods = pd.date_range(
                    start=data_test["datetime deployment"],
                    end=data_test["datetime recovery"],
                    freq=str(int(data_test["duration"].total_seconds() / n_periods))
                    + "s",
                )
                detect_periods, _ = np.histogram(data_histo, periods)
                rank_max.append(np.argmax(detect_periods) + 1)
                rank_min.append(np.argmin(detect_periods) + 1)

            total = len(data3)
            value_max_counts = Counter(rank_max)
            value_min_counts = Counter(rank_min)

            # Fill in the counts for null values
            for value in range(1, n_periods + 1, 1):
                if value not in value_max_counts:
                    value_max_counts[value] = 0
                if value not in value_min_counts:
                    value_min_counts[value] = 0

            value_max_counts = dict(sorted(value_max_counts.items()))
            value_min_counts = dict(sorted(value_min_counts.items()))

            count_max = [max_c for max_c in list(value_max_counts.values())]
            count_min = [min_c for min_c in list(value_min_counts.values())]
            values = list(value_max_counts.keys())

            fig, ax = plt.subplots()
            # bar plots
            ax.bar(values, count_max, align="center", edgecolor="black")
            # grid
            ax.yaxis.grid(color="gray", linestyle="--")
            ax.set_axisbelow(True)  # Set grid behind plot
            # x-label
            ax.set_xlabel("Periods")
            # title
            ax.set_title(
                f"Distribution of the period with most detections\ndetector: {d} - N={N}"
            )
            # ticks
            ax.set_xticks(values)
            ax.set_yticks(range(0, max(count_max) + 1, 2))
            # ax.set_yticks(range(0, 25, 2))
            ax.tick_params(axis="both")
            plt.ylim(0, max(count_max) + 1)
            # plt.ylim(0, 19)
            plt.show()


# %% statistics on diel plots
"""
Distribution of all the detection over night/day periods
"""

detector = ["pamguard"]
# arg = ['season', 'net', 'all']
arg = ["all", "season"]
lr_result = {}
pie_data = pd.DataFrame()

for d in detector:
    # data2 = filtered_data[f'{d}'][(filtered_data[f'{d}']['duration'] >= pd.Timedelta(hours=17)) & (filtered_data[f'{d}']['duration'] <= pd.Timedelta(hours=28))].reset_index(drop=True)
    data2 = filtered_data[f"{d}"]

    for a in arg:
        if a != "all":
            filter_list = list(dict.fromkeys(data2[f"{a}"]))

            for f in filter_list:
                data3 = data2[data2[f"{a}"] == f]
                N = len(data3)

                # Pie plot data
                pie_row = pd.DataFrame(
                    [
                        {
                            "detector": f"{d}",
                            f"{a}": f"{f}",
                            "duration": data3["duration"].sum().total_seconds() / 3600,
                            "detection": sum([len(df) for df in data3[f"df {d}"]]),
                        }
                    ]
                )
                pie_data = pd.concat([pie_data, pie_row], ignore_index=True)

                # Light regime
                lr = pd.DataFrame()
                for i in range(len(data3)):
                    lr = pd.concat(
                        [stats_diel_pattern(deployment=data3.iloc[i], detector=d), lr],
                        ignore_index=True,
                    )
                lr_result[f"{d}/{a}/{f}"] = lr

                ## Remove NaN values from each column for boxplot
                lr_clean = []
                for col in lr:
                    col_clean = lr[col][~np.isnan(lr[col])]
                    lr_clean.append(col_clean)

                # Light Regime boxplot
                fig, ax = plt.subplots(figsize=(2, 4))
                ax.grid(visible=False)
                ax.boxplot(
                    x=lr_clean,
                    positions=[1, 1.25],
                    patch_artist=True,
                    notch=False,
                    showfliers=False,
                    medianprops=dict(color="black"),
                    widths=0.15,
                    whis=0.75,
                )
                plt.xlim(0.8, 1.4)
                plt.ylim(-15, 10)
                plt.xticks([1, 1.25], ["Night", "Day"])
                plt.ylabel(
                    f"Detection distribution\n{a}: {f} - detector: {d} - N_deployment={N}",
                    fontsize=8,
                )
                plt.tight_layout()

        else:
            data3 = data2
            N = len(data3)

            lr = pd.DataFrame()
            for i in range(len(data3)):
                lr = pd.concat(
                    [stats_diel_pattern(deployment=data3.iloc[i], detector=d), lr],
                    ignore_index=True,
                )
            lr_result[f"{d}/{a}"] = lr

            # Remove NaN values from each column for boxplot
            lr_clean = []
            for col in lr:
                col_clean = lr[col][~np.isnan(lr[col])]
                lr_clean.append(col_clean)

            fig, ax = plt.subplots(figsize=(2, 4))
            ax.grid(visible=False)
            ax.boxplot(
                x=lr_clean,
                positions=[1, 1.25],
                patch_artist=True,
                notch=False,
                showfliers=False,
                medianprops=dict(color="black"),
                widths=0.15,
                whis=0.75,
            )
            plt.xticks([1, 1.25], ["Night", "Day"])
            plt.xlim(0.8, 1.4)
            plt.ylim(-15, 10)
            plt.ylabel(
                f"Detection distribution\n{a} - detector: {d} - N_deployment={N}"
            )
            plt.tight_layout()

        result_stat = []
        for i, regime in enumerate(lr_result):
            for period in ["Night", "Day"]:
                n = len(lr_result[regime][period])
                mean = lr_result[regime][period].mean()
                std = lr_result[regime][period].std()
                shapiro_stat, shapiro_pvalue = shapiro(
                    lr_result[regime][period], nan_policy="omit"
                )

                if shapiro_pvalue < 1e-7:
                    mann_stat, mann_pvalue = mannwhitneyu(
                        lr_result[regime]["Night"],
                        lr_result[regime]["Day"],
                        method="exact",
                        nan_policy="omit",
                    )
                    student_stat = float("NaN")
                    student_pvalue = float("NaN")
                else:
                    mann_stat = float("NaN")
                    mann_pvalue = float("NaN")
                    student_stat, student_pvalue = ttest_ind(
                        lr_result[regime]["Night"],
                        lr_result[regime]["Day"],
                        equal_var=False,
                        nan_policy="omit",
                    )

                result_stat.append(
                    [
                        regime,
                        period,
                        n,
                        mean,
                        std,
                        shapiro_stat,
                        shapiro_pvalue,
                        mann_stat,
                        mann_pvalue,
                        student_stat,
                        student_pvalue,
                    ]
                )

        result_stat2 = pd.DataFrame(
            result_stat,
            columns=[
                "regime",
                "period",
                "n",
                "mean",
                "std",
                "shapiro_stat",
                "shapiro_pvalue",
                "mann_stat",
                "mann_pvalue",
                "student_stat",
                "student_pvalue",
            ],
        )


# Pie plot
for d in detector:
    pie_data2 = pie_data[pie_data["detector"] == d]
    pie_total = (pie_data2["detection"] / pie_data2["duration"]).sum()
    perc = []
    for i in range(len(pie_data2)):
        perc.append(
            (pie_data2["detection"].iloc[i] / pie_data2["duration"].iloc[i]) / pie_total
        )
    pie_data2["percentage"] = perc

    plt.figure()
    plt.pie(
        pie_data2["percentage"],
        labels=pie_data2["season"],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#59D955", "#F2E205", "#F28B66", "#4694A6"],
    )
    plt.title(f"{d}")
    plt.show()

# %% Pie plots - distribution of detection over seasons

detector = ["pamguard", "thalassa"]

for d in detector:
    pie_data = pd.DataFrame()

    # data2 = filtered_data[f'{d}'][(filtered_data[f'{d}']['duration'] >= pd.Timedelta(hours=17)) & (filtered_data[f'{d}']['duration'] <= pd.Timedelta(hours=28))].reset_index(drop=True)
    data2 = filtered_data[f"{d}"]

    filter_list = list(dict.fromkeys(data2["season"]))

    for f in filter_list:
        data3 = data2[data2[f"{a}"] == f]

        pie_row = pd.DataFrame(
            [
                {
                    "detector": f"{d}",
                    f"{a}": f"{f}",
                    "duration": data3["duration"].sum().total_seconds() / 3600,
                    "detection": sum([len(df) for df in data3[f"df {d}"]]),
                }
            ]
        )
        pie_data = pd.concat([pie_data, pie_row], ignore_index=True)

    pie_total = (pie_data["detection"] / pie_data["duration"]).sum()

    perc = []
    for i in range(len(pie_data)):
        perc.append(
            (pie_data["detection"].iloc[i] / pie_data["duration"].iloc[i]) / pie_total
        )
    pie_data["percentage"] = perc

    plt.figure()
    plt.pie(
        pie_data["percentage"],
        labels=pie_data["season"],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#59D955", "#F2E205", "#F28B66", "#4694A6"],
    )
    plt.title(f"{d}")
    plt.show()

# %% cumulated histogram of detections for a single detection file

i = 40
detector = ["pamguard", "thalassa"]

for d in detector:
    data2 = filtered_data[f"{d}"]
    df_detections = data2[f"df {d}"][i]
    threshold = 0.75
    ID_test = data2["platform"][i] + " ST" + data2["recorder"][i]

    data_histo = df_detections["start_datetime"]
    tb = timebin
    deploy_dt = [
        data2["datetime deployment"][i],
        data2["datetime recovery"][i],
    ]  # beginning and end of deployment

    res_min = 10  # minute
    delta, start_vec, end_vec = (
        dt.timedelta(seconds=60 * res_min),
        deploy_dt[0],
        deploy_dt[1],
    )
    bins = pd.date_range(start=deploy_dt[0], end=deploy_dt[1], freq="1min")
    n_annot_max = (
        res_min * 60
    ) / tb  # max nb of annoted time_bin max per res_min slice

    total_detections = len(data_histo)
    threshold_detect = round(total_detections * threshold)

    hist, _ = np.histogram(
        [dt.timestamp() for dt in data_histo], bins=[b.timestamp() for b in bins]
    )
    cumul_count = np.cumsum(hist)
    bin_index = int(np.argmax(cumul_count >= threshold_detect))
    dt_thr = bins[bin_index]  # datetime of the threshold
    min_elapsed = bin_index  # elapsed time in minutes to achieve threshold_detect, freq bins is 1min
    perc_elapsed = round(
        (dt_thr - bins[0]) / (bins[-1] - bins[0]), 2
    )  # % of elapsed time to achieve threshold_detect

    bin_height = cumul_count[bin_index] / total_detections

    min_elapsed = (
        bin_index * res_min
    )  # elapsed time in minutes to achieve threshold_detect
    perc_elapsed = (dt_thr - bins[0]) / (
        bins[-1] - bins[0]
    )  # % of elapsed time to achieve threshold_detect

    fig, ax = plt.subplots(figsize=(20, 9))
    ax.hist(data_histo, bins, cumulative=True)  # histo
    # vline = ax.axvline(x=dt_thr, ymax = bin_height*0.92 ,color='r', linestyle='-', label='75% of Detections')
    vline = ax.axvline(
        x=dt_thr, ymax=1, color="r", linestyle="-", label="75% of Detections"
    )
    # hline = ax.axhline(y=threshold_detect, xmax = (dt_thr-bins[0])/(bins[-1]-bins[0]), color='r', linestyle='-', label='75% of Detections')
    hline = ax.axhline(
        y=threshold_detect, xmax=1, color="r", linestyle="-", label="75% of Detections"
    )
    ax.set_ylabel("Detections", fontsize=20)
    ax.text(
        bins[bin_index + 10],
        50,
        "{0} min - {1:.0f}%".format(min_elapsed, 100 * perc_elapsed),
        color="r",
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        bins[0],
        threshold * total_detections * 1.02 + 2,
        "{0} detections".format(threshold_detect),
        color="r",
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    tz_data = deploy_dt[0].tz
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz_data))
    fig.suptitle(
        f"Cumulated detections - {ID_test}\n{deploy_dt[0].strftime('%d/%m/%y')} - {deploy_dt[1].strftime('%d/%m/%y')} - detector: {d}",
        fontsize=24,
        y=0.98,
    )
    ax.grid(color="k", linestyle="-", linewidth=0.2, axis="both")

# %% analyze metadata - histogram of the detections for a single detection file
"""
The histogram is divided into n periods of equal durations
the periods number with the most detections are printed
"""

i = 50
n_periods = 12
detector = ["pamguard", "thalassa"]

for d in detector:
    data2 = filtered_data[f"{d}"]
    data_test = data2.iloc[i]
    name = data_test["platform"] + " ST" + data_test["recorder"]

    df_detections = data_test[f"df {d}"]

    data_histo = df_detections["start_datetime"]
    periods = pd.date_range(
        start=data_test["datetime deployment"],
        end=data_test["datetime recovery"],
        freq=str(int(data_test["duration"].total_seconds() / n_periods)) + "s",
    )

    fig, ax = plt.subplots()
    detect_periods, _, bars = ax.hist(data_histo, periods, edgecolor="black")

    # grids
    ax.yaxis.grid(color="gray", linestyle="--")

    # x-labels
    ax.set_xlabel("Periods")

    [ax.axvline(x=period, color="lime") for period in periods]
    ax.grid(color="k", linestyle="-", linewidth=0.1, axis="both")

    max_value = np.max(detect_periods)
    rank_max = np.argmax(detect_periods) + 1
    bars[rank_max - 1].set_color("darkorange")

    ax.set_ylabel("Detections")

    perc_max = detect_periods[rank_max - 1] / len(data_test[f"df {d}"])
    fig.suptitle(f"{name}\ndetector: {d}")

    print(
        f"\n{d}: most detections at period {rank_max}/{n_periods}, {detect_periods[rank_max-1]:.0f}/{len(data_test[f'df {d}'])} detections ({100*perc_max:.0f}%)",
        end="",
    )

# %% export csv for QGIS

detector = "thalassa"

filtered_df_1 = deploy[(deploy["recorder number"] == 1)]  # data with 1 ST

if detector == "thalassa":
    filtered_df_2 = deploy[
        (deploy["recorder number"] == 2) & (deploy["net length"] <= 200)
    ]
    filtered_df_3 = deploy[
        (deploy["recorder number"] == 2) & (deploy["net length"] > 200)
    ]
elif detector == "pamguard":
    filtered_df_2 = deploy[
        (deploy["recorder number"] == 2) & (deploy["net length"] <= 500)
    ]
    filtered_df_3 = deploy[
        (deploy["recorder number"] == 2) & (deploy["net length"] > 500)
    ]

sub_df_2 = (
    filtered_df_2.groupby("ID platform").first().reset_index()
)  # only the first ST
deploy2 = (
    pd.concat([filtered_df_1, sub_df_2, filtered_df_3])
    .sort_values(by=["campaign", "deployment", "ID recorder"])
    .reset_index(drop=True)
)

df_deploy = []
for c, d, r in zip(deploy2["campaign"], deploy2["deployment"], deploy2["ID recorder"]):
    df_deploy.append(
        data[
            (data["campaign"] == c)
            & (data["deployment"] == d)
            & (data["recorder"] == str(r))
        ][f"df {detector}"].iloc[-1]
    )

deploy2[f"detection rate {detector}"] = [
    (
        len(df_deploy[i])
        / (pd.to_timedelta(deploy2["duration"][i]).total_seconds() / 86400)
        / 1440
    )
    * 100
    for i in range(len(deploy2))
]

deploy_out = deploy2.drop(
    columns=[
        "lat D",
        "lat DM",
        "lat DD",
        "long D",
        "long DM",
        "long DD",
        "weather deployment",
        "weather recovery",
        "comment",
        "species",
    ]
)
date_today = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
out_filename = f"APOCADO - Suivi déploiements {detector}_{date_today}.csv"
out_folder = r"L:\acoustock\Bioacoustique\DATASETS\APOCADO\Code\carto"
deploy_out.to_csv(
    os.path.join(out_folder, out_filename), index=False, encoding="latin1"
)
