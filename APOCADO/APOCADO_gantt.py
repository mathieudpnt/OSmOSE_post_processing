from pandas import Timedelta, Timestamp, DataFrame, read_excel, date_range
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
from def_func import add_season_period

mpl.rcdefaults()
mpl.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["figure.figsize"] = [10, 4]

def main():
    # Load and format data
    data = read_excel(
        r"L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/APOCADO - Suivi déploiements.xlsm",
        skiprows=[0]
    )
    data = data[data["check heure Raven"] == 1].reset_index(drop=True)

    data["datetime deployment"] = [
        Timestamp.combine(data["date deployment"][i].date(), data["time deployment"][i])
        for i in range(len(data))
    ]
    data["datetime recovery"] = [
        Timestamp.combine(data["date recovery"][i].date(), data["time recovery"][i])
        for i in range(len(data))
    ]

    data_gant = DataFrame(
        columns=["campaign", "recorder", "dt_deployment", "dt_recovery"]
    )
    campaign_list = list(set(data["campaign"]))
    for C in campaign_list:
        recorder_list = list(set(data[data["campaign"] == C]["ID recorder"]))
        for R in recorder_list:
            date_beg = data[(data["campaign"] == C) & (data["ID recorder"] == R)][
                "datetime deployment"
            ].min()
            date_end = data[(data["campaign"] == C) & (data["ID recorder"] == R)][
                "datetime recovery"
            ].max()
            data_gant.loc[len(data_gant)] = [C, str(R), date_beg, date_end]

    # create a column with the color for each element of the arg variable
    list_arg = sorted(list(data_gant["campaign"].unique()))
    colors = [mpl.colors.rgb2hex(i) for i in mpl.cm.tab20(range(len(list_arg)))]
    c_dict = dict(zip(list_arg, colors))
    data_gant["color"] = data_gant["campaign"].map(c_dict)

    data_gant = data_gant.sort_values("recorder", ascending=False).reset_index(drop=True)

    # Plot
    fig, (ax0, ax1) = plt.subplots(2, gridspec_kw={"height_ratios": [6, 1]})

    # data to plot
    for i in range(len(data_gant)):
        ax0.barh(
            y=data_gant["recorder"][i],
            width=(data_gant["dt_recovery"][i] - data_gant["dt_deployment"][i]),
            left=data_gant["dt_deployment"][i],
            color=data_gant.color[i],
            alpha=0.8,
        )
    ax0.set_xlim(
        data_gant["dt_deployment"].min() - Timedelta(days=7),
        data_gant["dt_recovery"].max() + Timedelta(days=7),
    )

    # ticks
    xticks = date_range(
        start=data_gant["dt_deployment"].min(),
        end=data_gant["dt_recovery"].max(),
        freq="MS",
    )
    xticks_labels = [date.strftime("%m/%y") for date in xticks]
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xticks_labels)
    ax0.tick_params(axis="x")

    # spines
    ax0.spines["right"].set_visible(False)
    ax0.spines["top"].set_visible(False)

    # title
    plt.suptitle("APOCADO")

    acqui_ST = Timestamp("2022-09-01")
    ax0.axvline(x=acqui_ST, ymin=0, ymax=0.15, color="black")
    ax0.text(
        acqui_ST - Timedelta(days=5),
        ax0.get_ylim()[1] * 0,
        "Acquisition\n ST400HF",
        color="black",
        ha="right",
        va="bottom",
    )

    # seasons
    add_season_period(ax=ax0)

    # legend
    legend_elements = [
        Patch(facecolor=c_dict[dep], label=dep)
        for dep in sorted(list(data_gant["campaign"].unique()))
    ]
    legend = ax1.legend(
        handles=legend_elements,
        loc="upper center",
        ncols=len(legend_elements) // 2,
        frameon=False,
        title="campaign",
    )
    plt.setp(legend.get_texts(), color="black")
    legend.get_title().set_color("black")

    # clean second axis
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

if __name__ == '__main__':
    main()
    plt.show()