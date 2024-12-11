import numpy as np
from dataclasses import dataclass
from typing import List, Union
import datetime as dt
from pathlib import Path
import matplotlib.pyplot as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import scipy.signal as signal
from def_func import suntime_hour
import csv
import pytz
import os
from tqdm import tqdm

mpl.rcdefaults()
mpl.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["figure.figsize"] = [10, 4]


@dataclass
class LTAS:
    def __init__(
        self,
        matrix: np.ndarray = None,
        time: List[pd.Timestamp] = None,
        freq: List[Union[int, float]] = None,
        path: Union[Path, str] = None,
        begin_datetime: pd.Timestamp = None,
        datetime_min: pd.Timestamp = None,
        datetime_max: pd.Timestamp = None,
        t_res: int = None,
        f_res: int = None,
        sensitivity: float = None,
        duty_cycle: int = 100,
    ):
        if matrix is not None:
            assert np.issubdtype(
                matrix.dtype, np.number
            ), "Matrix must be an array of numbers"
            assert isinstance(freq, list) and all(
                isinstance(item, (int, float)) for item in freq
            ), "freq arg must be a list of numerical values"
            assert all(
                isinstance(item, pd.Timestamp) for item in time
            ), "time arg must be a list of datetimes"
            assert (
                len(time) == matrix.shape[1]
            ), f"time vector is of length {len(time)} while matrix is of length {matrix.shape[1]}"
            assert (
                len(freq) == matrix.shape[0]
            ), f"frequency vector is of length {len(time)} while matrix is of height {matrix.shape[0]}"

            matrix[:, -1] = (
                0  # set last welch to zero so that spectrograms appears blue in case of big gap in the data
            )
            self.welch = matrix
            self.time = time
            self.freq = freq
            self.f_max = freq[-1]

        else:
            assert path.endswith("npz") or path.endswith(
                "csv"
            ), "path arg must be a .npz or .csv"

            if path.endswith("npz"):
                pkl = np.load(path, allow_pickle=True)
                self.welch = pkl["welch"]
                self.time = np.sort(pkl["time"])
                self.freq = pkl["freq"]

                """
                #  On trie les welch, car ils ne sont pas forcement rangés dans l'odre dans le fichier pkl
                sorted_welch_idx = np.argsort(self.time)
                welch_sorted = welch[..., sorted_welch_idx]
                welch_sorted[:, -1] = 0
                self.welch = welch_sorted
                """
            elif path.endswith("csv"):
                self.load_from_csv(
                    path,
                    begin_datetime,
                    datetime_min,
                    datetime_max,
                    t_res,
                    f_res,
                    sensitivity,
                    duty_cycle,
                )

    def load_from_csv(
        self,
        path,
        begin_datetime,
        date_min,
        date_max,
        t_res,
        f_res,
        sensitivity=None,
        duty_cycle=100,
    ):
        assert isinstance(
            begin_datetime, pd.Timestamp
        ), "A datetime must be passed to begin_datetime arg"
        assert isinstance(
            t_res, (int, np.integer)
        ), "An integer must be passed to t_res arg"
        assert isinstance(
            f_res, (int, np.integer)
        ), "An integer must be passed to f_res arg"
        assert sensitivity is None or isinstance(
            sensitivity, Union[float, int]
        ), "If provided, sensitivity must be a numerical value > 0"
        assert (
            isinstance(duty_cycle, int) and duty_cycle > 0 and duty_cycle <= 100
        ), "If provided, duty_cycle must be an integer between 0 and 100"

        # get matrix shape
        with open(path, mode="r") as f:
            reader = csv.reader(f, delimiter=",")
            shape_t = int(next(reader)[1]) - 1
            shape_f = int(next(reader)[1]) - 1

        freq = list(np.linspace(0, shape_f * f_res, shape_f).astype("int32"))
        end_datetime = begin_datetime + pd.Timedelta(
            int(t_res // (0.01 * duty_cycle)) * (shape_t), "second"
        )
        time = pd.date_range(
            start=begin_datetime,
            end=end_datetime,
            freq=str(int(t_res // (0.01 * duty_cycle))) + "s",
        ).to_list()

        index = [0, len(time)]
        if date_min is not None and date_max is not None:
            time2 = [t for t in time if date_min <= t <= date_max]
            index = [time.index(t) for t in [time2[0], time2[-1]]]
            time = time2

        # matrix_raw = pd.read_csv(path, delimiter=',', skiprows=2, header=None, nrows=index[-1], low_memory=False)

        chunks = []
        chunk_size = 1000

        try:
            for i, chunk in tqdm(
                enumerate(
                    pd.read_csv(
                        path,
                        delimiter=",",
                        skiprows=2,
                        header=None,
                        chunksize=chunk_size,
                        low_memory=False,
                        on_bad_lines="skip",
                    )
                ),
                desc="Reading csv...",
                total=(shape_t // chunk_size) + 1,
                unit="chunk",
                colour="green",
            ):
                chunk.replace(-np.inf, np.nan, inplace=True)
                chunk.dropna(inplace=True)
                chunks.append(chunk)
        except pd.errors.ParserError as e:
            print(f"Parser error: {e}")
        print("Done!")

        matrix_raw = pd.concat(chunks, ignore_index=True)

        matrix = matrix_raw.iloc[index[0] : index[-1]]
        matrix = np.array(matrix, dtype=np.float64).transpose()

        if sensitivity:
            matrix += np.abs(sensitivity)

        self.time = time
        self.welch = matrix[:-1]
        self.freq = freq
        self.f_max = freq[-1]
        return

    def concatenate_LTAS(self, ltas):
        # concatenate_LTAS returns a LTAS object from two ltas objects
        welch_concat = np.concatenate((self.welch, ltas.welch), axis=1)
        time_concat = np.concatenate([self.time, ltas.time])
        freq_concat = self.freq

        ltas_concat = LTAS(matrix=welch_concat, time=time_concat, freq=freq_concat)

        return ltas_concat

    def plot_LTAS(self, output_path=None, output_name=None, dyn_min=-120, dyn_max=0):
        # fig, ax = plt.subplots()
        # X, Y = np.meshgrid(self.time, self.freq)
        # p = plt.pcolormesh(
        #     X,
        #     Y,
        #     self.welch[:-1, :-1],
        #     vmin=40,
        #     vmax=80,
        #     rasterized=True,
        #     shading="flat",
        # )
        # cbar = fig.colorbar(p)
        # cbar.set_label("dB ref 1µPa² @ 1m", rotation=270, labelpad=20)

        fig, ax = plt.subplots()
        plt.imshow(
            self.welch[:-1, :-1],
            aspect="auto",
            vmin=dyn_min,
            vmax=dyn_max,
            origin="lower",
            extent=[self.time[0], self.time[-1], self.freq[0], self.freq[-1]],
        )
        plt.colorbar(label="dB ref 1µPa² @ 1m")

        duration = self.time[-1] - self.time[0]

        if duration < pd.Timedelta(12, "hour"):
            t_inter = 1
            date_fmt = "%H:%M"
            locator = mdates.HourLocator(interval=t_inter)
        elif duration < pd.Timedelta(24, "hour"):
            t_inter = 2
            date_fmt = "%H:%M"
            locator = mdates.HourLocator(interval=t_inter)
        elif duration < pd.Timedelta(7, "day"):
            t_inter = 4
            date_fmt = "%m/%d\n%H:%M"
            locator = mdates.HourLocator(interval=t_inter)
        elif duration > pd.Timedelta(7, "day"):
            t_inter = 1
            date_fmt = "%Y%n%b"
            locator = mdates.MonthLocator(interval=t_inter)

        tz = pytz.FixedOffset(self.time[0].utcoffset().total_seconds() // 60)

        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt, tz=tz))
        ax.xaxis.set_major_locator(locator)

        if output_name is not None:
            plt.title(
                f"LTAS {output_name}\nfrom {self.time[0].strftime('%Y/%m/%d %H:%M')} to {self.time[-1].strftime('%Y/%m/%d %H:%M')}"
            )
        else:
            plt.title(
                f"LTAS from {self.time[0].strftime('%Y/%m/%d %H:%M')} to {self.time[-1].strftime('%Y/%m/%d %H:%M')}"
            )
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time")
        plt.tight_layout()

        if output_path is not None and output_name is not None:
            out_name = os.path.join(output_path, "LTAS - " + output_name)
            plt.savefig(fname=out_name, bbox_inches="tight")
            np.savez(
                f"{out_name}.npz",
                welch=self.welch[:-1, :-1],
                time=self.time,
                freq=self.freq,
            )
            print(f"Figure and npz saved to {output_path}\n")
        plt.show()

    def plot_PSD(self, output_path, output_name):
        med = np.median(self.welch, axis=1)
        q5 = np.quantile(self.welch, 0.05, axis=1)
        q25 = np.quantile(self.welch, 0.25, axis=1)
        q75 = np.quantile(self.welch, 0.75, axis=1)
        q95 = np.quantile(self.welch, 0.95, axis=1)

        fig, ax = plt.subplots()
        plt.plot(self.freq, med, color="lightcoral", label="Median")
        plt.plot(
            self.freq,
            q5,
            color="mediumaquamarine",
            linestyle="--",
            label="Quantiles 5% and 95%",
        )
        plt.plot(self.freq, q25, color="firebrick", label="Quantiles 25% and 75%")
        plt.plot(self.freq, q75, color="firebrick")
        plt.plot(self.freq, q95, color="mediumaquamarine", linestyle="--")

        plt.grid(True, which="both", color="gainsboro")

        ax.set_ylabel("Amplitude (dB ref 1µPa²/Hz)")
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_xscale("log")
        ax.legend(loc="upper right")

        plt.ylim(20, 140)

        plt.title(
            f"PSD of signal from {self.time[0].strftime('%Y/%m/%d %H:%M')} to {self.time[-1].strftime('%Y/%m/%d %H:%M')}"
        )

        if output_name is not None:
            plt.title(
                f"PSD {output_name}\nfrom {self.time[0].strftime('%Y/%m/%d %H:%M')} to {self.time[-1].strftime('%Y/%m/%d %H:%M')}"
            )
        else:
            plt.title(
                f"PSD from {self.time[0].strftime('%Y/%m/%d %H:%M')} to {self.time[-1].strftime('%Y/%m/%d %H:%M')}"
            )
        plt.tight_layout()

        if output_path is not None and output_name is not None:
            out_name = os.path.join(output_path, "PSD - " + output_name)
            plt.savefig(fname=out_name, bbox_inches="tight")
            np.savez(
                f"{out_name}.npz",
                freq=self.freq,
                med=med,
                q5=q5,
                q25=q25,
                q75=q75,
                q95=q95,
            )
            print(f"Figure and npz saved to {output_path}\n")
        plt.show()

    def compute_periodicty(self):
        med = np.median(self.welch, axis=0)
        q5 = np.quantile(self.welch, 0.05, axis=0)
        q25 = np.quantile(self.welch, 0.25, axis=0)
        q75 = np.quantile(self.welch, 0.75, axis=0)
        q95 = np.quantile(self.welch, 0.95, axis=0)

        # Plot temporal mediane and quartiles
        fig, ax = plt.subplots(2)
        ax[0].plot(self.time, med, label="median")
        ax[0].plot(self.time, q5, label="q5")
        ax[0].plot(self.time, q25, label="q25")
        ax[0].plot(self.time, q75, label="q75")
        ax[0].plot(self.time, q95, label="q95")

        ax[0].legend(loc="upper right")
        ax[0].grid(True, which="both", color="gainsboro")

        tz = self.time[0].tz

        file_maree = r"C:\Users\torterma\Documents\Projets_Osmose\Sciences\2_FirstSoundscapeAnalysis\maregraphe_conquet_152_2022.csv"
        # plot_tides(data_path, date_begin, date_end, tz):
        df_maree = pd.read_csv(file_maree, skiprows=13, delimiter=";")
        df_maree["# Date"] = pd.to_datetime(
            df_maree["# Date"], format="%d/%m/%Y %H:%M:%S"
        )
        df_maree = df_maree[df_maree["Source"] == 1]  # Only data from Source==1
        df_maree = df_maree.drop(["Source"], axis=1)  # 'Source' column useless
        df_maree["# Date"] = df_maree["# Date"].dt.tz_localize(
            "UTC"
        )  # Timezone of source data
        df_maree["# Date"] = df_maree["# Date"].dt.tz_convert(tz)  # Change of TZ
        df2_maree = df_maree[
            (df_maree["# Date"] <= self.time[-1]) & (df_maree["# Date"] >= self.time[0])
        ]  # sorting
        df2_maree = df2_maree.reset_index(
            drop=True
        )  # reset the indexes of row after sorting the df

        dt_maree = df2_maree["# Date"]
        hauteur = df2_maree["Valeur"]
        # hauteur2 = pd.Series(savgol_filter(hauteur, 301, 4)) #filtering
        # return (dt_maree, hauteur2, h_max)

        ax[1].plot(dt_maree, hauteur)
        ax[1].grid(True, which="both", color="gainsboro")

        plt.show()

        fig, ax = plt.subplots()
        # compute FTT signal q5
        Fs = 1 / ((self.time[1] - self.time[0]).seconds)

        psd_med = signal.welch(med, fs=Fs, scaling="spectrum")
        psd_q5 = signal.welch(q5, fs=Fs, scaling="spectrum")
        # psd_q25 = signal.welch(q25, fs=Fs, scaling="spectrum")
        # psd_q75 = signal.welch(q75, fs=Fs, scaling="spectrum")
        # psd_q95 = signal.welch(q95, fs=Fs, scaling="spectrum")
        # plt.plot(psd_med[0], psd_med[1])
        # plt.plot(psd_q5[0], psd_q5[1])
        # # plt.legend(loc='upper right')
        # # ax.set_xscale('log')
        # plt.show()

        # Find peaks in PSD

        psd_med_dB = 10 * np.log10(psd_med[1] / min(psd_med[1]))
        psd_q5_dB = 10 * np.log10(psd_q5[1] / min(psd_q5[1]))
        # psd_q25_dB = 10*np.log10(psd_q25[1]/min(psd_q25[1]))
        # psd_q75_dB = 10*np.log10(psd_q75[1]/min(psd_q75[1]))
        # psd_q95_dB = 10*np.log10(psd_q95[1]/min(psd_q95[1]))

        f_psd = psd_med[0]
        # # Find peaks with 5 dB prominence compared to the rest
        idx_peaks_med = signal.find_peaks(psd_med_dB, prominence=5)
        idx_peaks_q5 = signal.find_peaks(psd_q5_dB, prominence=5)
        fig, ax = plt.subplots()
        plt.plot(psd_med[0], psd_med_dB, label="median")
        plt.plot(f_psd[idx_peaks_med[0]], psd_med_dB[idx_peaks_med[0]], "x")
        for i, j in zip(f_psd[idx_peaks_med[0]], psd_med_dB[idx_peaks_med[0]]):
            ax.annotate(str(round((1 / i) / 3600, 1)), xy=(i, j))
            # ax.set_xscale('log')

        plt.plot(psd_q5[0], psd_q5_dB, label="q5")
        plt.plot(f_psd[idx_peaks_q5[0]], psd_q5_dB[idx_peaks_q5[0]], "x")
        for i, j in zip(f_psd[idx_peaks_q5[0]], psd_q5_dB[idx_peaks_q5[0]]):
            ax.annotate(str(round((1 / i) / 3600, 1)), xy=(i, j))
            # ax.set_xscale('log')

        # If x axis in hours
        # ax.set_xlim(0, max(f_psd))
        # xlabels = ax.get_xticklabels()
        # xlabels_int = [float(l.get_text()) for l in xlabels]
        # xlabels_int[1:] = [str(round((1/x)/3600,1)) for x in xlabels_int[1:]]
        # ax.set_xticklabels(xlabels_int)

        plt.grid(True, which="both", color="gainsboro")
        plt.legend(loc="upper right")
        ax.set_ylabel("Amplitude (arbitraire)")
        ax.set_xlabel("Fréquence (Hz)")
        plt.show()

        # Compute median noise in LF (< 1000Hz)
        index_f = np.argmin(np.abs(np.array(self.freq) - 1000))

        med = np.median(self.welch[0:index_f, :], axis=0)

        # Plot temporal mediane and quartiles
        # fig, ax = plt.subplots()
        # plt.plot(self.time, med)
        # plt.show()

    def plot_monthly_PSD(self):
        # We need the welch to be split by month (1 w per month) and then we compute PSD on new welch

        # Define colors and names of month
        list_color = {
            "label": [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ],
            "number": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "color": [
                "powderblue",
                "lightskyblue",
                "navy",
                "plum",
                "orchid",
                "purple",
                "lightgreen",
                "lime",
                "darkgreen",
                "mistyrose",
                "lightcoral",
                "chocolate",
            ],
        }
        df_color = pd.DataFrame(data=list_color)

        # List of month with data
        months = [t.month for t in self.time]
        unique_month = np.unique(months)
        # Set figure parameters
        welch_month = np.empty(
            [
                len(self.freq),
            ]
        )
        fig, ax = plt.subplots()
        ax.set_ylabel("Amplitude (dB ref 1µPa²/Hz)")
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_xscale("log")
        plt.grid(True, which="both", color="gainsboro")
        # Browse months and compute median noise for each month
        for m in unique_month:
            for j, n in enumerate(months):
                if n == m:
                    welch_month = np.column_stack([welch_month, self.welch[:, j]])
            c = df_color.loc[df_color["number"] == m, "color"].values[0]
            ml = df_color.loc[df_color["number"] == m, "label"].values[0]
            med = np.median(welch_month, axis=1)
            # Plot the PSD for month m
            plt.plot(self.freq, med, color=c, label=ml)

        ax.legend(loc="upper right", fontsize=14)
        plt.show()

    def plot_seasonal_PSD(self):
        # Define colors and names of seasons
        list_color = {
            "label": ["Winter", "Spring", "Summer", "Autumn"],
            "number": [1, 2, 3, 4],
            "color": ["powderblue", "plum", "lightgreen", "chocolate"],
        }
        df_color = pd.DataFrame(data=list_color)

        # Define season of each timestamp
        season = []
        for i in self.time:
            m = i.month
            if m >= 3 and m <= 5:
                season.append(2)
            elif m >= 6 and m <= 8:
                season.append(3)
            elif m >= 9 and m <= 11:
                season.append(4)
            else:
                season.append(1)

        # Set figure parameters
        welch_season = np.empty(
            [
                len(self.freq),
            ]
        )
        fig, ax = plt.subplots()
        ax.set_ylabel("Amplitude (dB ref 1µPa²/Hz)")
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_xscale("log")
        plt.grid(True, which="both", color="gainsboro")

        # Browse months and compute median noise for each month
        for s in np.unique(season):
            for j, n in enumerate(season):
                if n == s:
                    welch_season = np.column_stack([welch_season, self.welch[:, j]])
            c = df_color.loc[df_color["number"] == s, "color"].values[0]
            ml = df_color.loc[df_color["number"] == s, "label"].values[0]
            med = np.median(welch_season, axis=1)
            # Plot the PSD for month m
            plt.plot(self.freq, med, color=c, label=ml)

        ax.legend(loc="upper right", fontsize=14)
        plt.show()

    def diel_PSD(self, lat, lon):
        # Define colors and names of light regimes
        list_color = {
            "label": ["Night", "Dawn", "Day", "Dusk"],
            "number": [1, 2, 3, 4],
            "color": ["powderblue", "plum", "lightgreen", "chocolate"],
        }
        df_color = pd.DataFrame(data=list_color)

        # Find sun times for the days with data
        [_, _, dt_dusk, dt_dawn, dt_day, dt_night] = suntime_hour(
            self.time[0], self.time[-1], self.time[0].tz, lat, lon
        )

        # List of days in the dataset
        list_days = [dt.date(d.year, d.month, d.day) for d in dt_day]

        # Assign a light regime to each detection
        # : 1 = night ; 2 = dawn ; 3 = day ; 4 = dusk
        day_det = [start_datetime.date() for start_datetime in self.time]
        light_regime = []
        for idx_day, day in enumerate(list_days):
            for idx_det, d in enumerate(day_det):
                # If the detection occured during 'day'
                if d == day:
                    if (
                        self.time[idx_det] > dt_dawn[idx_day]
                        and self.time[idx_det] < dt_day[idx_day]
                    ):
                        lr = 2
                        light_regime.append(lr)
                    elif (
                        self.time[idx_det] > dt_day[idx_day]
                        and self.time[idx_det] < dt_night[idx_day]
                    ):
                        lr = 3
                        light_regime.append(lr)
                    elif (
                        self.time[idx_det] > dt_night[idx_day]
                        and self.time[idx_det] < dt_dusk[idx_day]
                    ):
                        lr = 4
                        light_regime.append(lr)
                    else:
                        lr = 1
                        light_regime.append(lr)

        # Set figure parameters
        welch_diel = np.empty(
            [
                len(self.freq),
            ]
        )
        fig, ax = plt.subplots()
        ax.set_ylabel("Amplitude (dB ref 1µPa²/Hz)")
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_xscale("log")
        plt.grid(True, which="both", color="gainsboro")

        # Browse months and compute median noise for each month
        for s in np.unique(light_regime):
            for j, n in enumerate(light_regime):
                if n == s:
                    welch_diel = np.column_stack([welch_diel, self.welch[:, j]])
            c = df_color.loc[df_color["number"] == s, "color"].values[0]
            ml = df_color.loc[df_color["number"] == s, "label"].values[0]
            med = np.median(welch_diel, axis=1)
            # Plot the PSD for month m
            plt.plot(self.freq, med, color=c, label=ml)

        ax.legend(loc="upper right", fontsize=14)
        plt.show()

    def __str__(self):
        """Return a string representation of the LTAS"""
        print("LTAS:")
        result = ""
        result += f"matrix shape: {self.welch.shape}\n"
        result += f"fmax: {self.f_max} Hz\n"
        result += f"begin_datetime: {self.time[0]}\n"
        result += f"end_datetime: {self.time[-1]}\n"

        return result
