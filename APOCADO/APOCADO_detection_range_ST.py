import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from scipy import stats
import json
from tqdm import tqdm
from scipy.stats import linregress
import pickle
import matplotlib as mpl

from def_func import t_rounder, get_season, sort_detections

mpl.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 200
# %%
detector = ["pamguard", "thalassa"]
arg = ["season", "net"]
timebin1 = 60  # en seconde
timebin2 = 10  # en minute

# %% Load data

data_load = "pickle"
match data_load:
    case "pickle":
        # with open(r'L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO\data.pkl', 'rb') as f:
        with open(
            r"C:\Users\dupontma2\Desktop\data_local\data_timebin60s.pkl", "rb"
        ) as f:
            data = pickle.load(f)

    case "manual":
        data = []
        path_json = [
            r"L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO",
            r"Y:\Bioacoustique\APOCADO2",
            r"Z:\Bioacoustique\DATASETS\APOCADO3",
        ]

        list_json = [
            file_path
            for path in path_json
            for file_path in glob.glob(
                os.path.join(path, "**/metadata.json"), recursive=True
            )
        ]

        for i in tqdm(range(len(list_json)), desc="Scanning metadata files"):
            r_file = open(list_json[i], "r")
            data.append(json.load(r_file))
            r_file.close()
        data = (
            pd.DataFrame.from_dict(data)
            .sort_values(["campaign", "deployment"])
            .reset_index(drop=True)
        )

        df_detections_pamguard, df_detections_thalassa = [], []
        for i in tqdm(range(len(data)), desc="DataFrame creation"):
            data["campaign"][i]

            f1 = data["pamguard detection file"][i]
            f2 = data["thalassa detection file"][i]
            tz = (
                data["datetime deployment"][i][-5:-2]
                + ":"
                + data["datetime deployment"][i][-2:]
            )
            ts = data["segment timestamp file"][i]
            df_detections_file1, _ = sort_detections(
                file=f1,
                tz=tz,
                timebin_new=timebin1,
                timestamp_file=ts,
                date_begin=data["datetime deployment"][i],
                date_end=data["datetime recovery"][i],
            )
            df_detections_file2, _ = sorting_detections(
                file=f2,
                tz=tz,
                timebin_new=timebin1,
                timestamp_file=ts,
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

        for d in detector:
            data[f"detection rate {d}"] = [
                (
                    len(data[f"df {d}"][i])
                    / (pd.to_timedelta(data["duration"][i]).total_seconds() / 86400)
                )
                / (86400 / timebin1)
                for i in range(len(data))
            ]

data["deployment ID"] = data["platform"] + " ST" + data["recorder"]
# %% Plot regressions

data1 = data[(data["recorder number"] == 2)].reset_index(drop=True)

for d in detector:

    if d == "thalassa":
        net_category = []
        for i in range(len(data1)):
            if data1["net length"].iloc[i] <= 200:
                net_category.append("1: [0 - 200] m")
            elif (data1["net length"].iloc[i] > 200) & (
                data1["net length"].iloc[i] <= 300
            ):
                net_category.append("2: ]200 - 300] m")
            elif (data1["net length"].iloc[i] > 300) & (
                data1["net length"].iloc[i] <= 400
            ):
                net_category.append("3: ]300 - 400] m")
            elif data1["net length"].iloc[i] > 400:
                net_category.append("4: +400m")
            else:
                raise ValueError(f"{i}")
        data1["net category"] = net_category

    elif d == "pamguard":
        net_category = []
        for i in range(len(data1)):
            if data1["net length"].iloc[i] <= 300:
                net_category.append("1: [0 - 300] m")
            elif (data1["net length"].iloc[i] > 300) & (
                data1["net length"].iloc[i] <= 500
            ):
                net_category.append("2: ]300 - 500] m")
            elif data1["net length"].iloc[i] > 500:
                net_category.append("3: +500m")
            else:
                raise ValueError(f"{i}")
        data1["net category"] = net_category

    data_corr = pd.DataFrame()

    list_length = sorted(list(set(data1["net category"])))
    n_samples = pd.Series.sort_index(0.5 * data1["net category"].value_counts()).astype(
        int
    )

    for length in list_length:
        sub_data = data1[data1["net category"] == length].sort_values(
            by=["deployment ID"]
        )
        list_sub_ID = sorted(
            list(set([i.split(" ")[0] for i in sub_data["deployment ID"]]))
        )
        for ID in list_sub_ID:
            sub_data2 = sub_data[
                sub_data["deployment ID"].str.contains(ID)
            ].reset_index(drop=False)
            if len(sub_data2) == 2:
                df1_detections = sub_data2[f"df {d}"][0]
                df2_detections = sub_data2[f"df {d}"][1]

                tz_data = df1_detections["start_datetime"][0].tz

                # time_bin = list(set(sub_data2[f'{d} timebin']))[0]
                time_bin = timebin1

                label_legend = [
                    sub_data2["deployment ID"][0],
                    sub_data2["deployment ID"][1],
                ]

                res_min = timebin2

                delta = dt.timedelta(seconds=60 * res_min)
                start_vec = t_rounder(
                    pd.Timestamp(min(sub_data2["datetime deployment"])), res=60
                )
                end_vec = t_rounder(
                    pd.Timestamp(max(sub_data2["datetime recovery"])), res=60
                )

                time_vector = [
                    start_vec + i * delta
                    for i in range(int((end_vec - start_vec) / delta) + 1)
                ]
                duration_h = int(
                    (time_vector[-1] - time_vector[0]).total_seconds() // 3600
                )

                n_annot_max = (
                    res_min * 60
                ) / time_bin  # max nb of annoted time_bin max per res_min slice

                hist1, _ = np.histogram(
                    [dt.timestamp() for dt in df1_detections["start_datetime"]],
                    bins=[dt.timestamp() for dt in time_vector],
                )
                hist2, _ = np.histogram(
                    [dt.timestamp() for dt in df2_detections["start_datetime"]],
                    bins=[dt.timestamp() for dt in time_vector],
                )
                counts1 = 100 * (hist1 / n_annot_max)
                counts2 = 100 * (hist2 / n_annot_max)

                df_corr = pd.DataFrame(
                    {
                        "deployment": label_legend[0].split(" ")[0]
                        + " "
                        + label_legend[0].split(" ")[-1]
                        + "/"
                        + label_legend[1].split(" ")[-1],
                        "ST1": list(counts1),
                        "ST2": list(counts2),
                        "net_len": length,
                    }
                )

                if len(data_corr) == 0:
                    data_corr = df_corr
                else:
                    data_corr = pd.concat([data_corr, df_corr])

    print("Done")

    sns.set(font_scale=2)

    # ST correlation
    g = sns.lmplot(
        data=data_corr,
        x="ST1",
        y="ST2",
        hue="net_len",
        scatter_kws={"s": 1},
        line_kws={"lw": 1},
        ci=90,
        legend=False,
        height=12,
        markers="o",
        palette="tab10",
    )

    color_leg = [g.ax.lines[i].get_color() for i in range(len(g.ax.lines))]
    for i, n in enumerate(sorted(list(set(data_corr["net_len"])))):
        r, p = stats.pearsonr(
            data_corr[data_corr["net_len"] == n]["ST1"],
            data_corr[data_corr["net_len"] == n]["ST2"],
        )

        slope, intercept, r_value, p_value, std_err = linregress(
            data_corr[data_corr["net_len"] == n]["ST1"],
            data_corr[data_corr["net_len"] == n]["ST2"],
        )

        ax = plt.gca()
        ax.text(
            0.45,
            0.20 - (i * 0.035),
            f"{n} - R²={r*r:.2f} - N={n_samples[i]}",
            transform=ax.transAxes,
            color=color_leg[i],
        )

    g.ax.set(
        xlabel="ST1 - 1min positive detection rate",
        ylabel="ST2 - 1min positive detection rate",
    )

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title(
        f"Appaired SoundTraps correlation\n detector: {d} - {timebin1}s window per {res_min}min time bin",
        y=1,
    )

    plt.show()
