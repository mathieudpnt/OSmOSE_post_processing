from typing import Tuple
import pytz
import pandas as pd
import datetime as dt
import random
import numpy as np
from tqdm import tqdm


def oneday_per_month(
    time_vector_ts, time_vector_str, vec
) -> Tuple[list, list, list, list]:
    # select a random day for each months in input datetimes list and returns all the datetimes of those randomly selected days

    time_vector = [
        dt.datetime.fromtimestamp(time_vector_ts[i]) for i in range(len(time_vector_ts))
    ]

    datetimes_by_month = {}
    for i in range(len(time_vector)):
        key = (time_vector[i].year, time_vector[i].month)
        if key not in datetimes_by_month:
            datetimes_by_month[key] = []
        datetimes_by_month[key].append((time_vector[i], vec[i], time_vector_str[i]))

    # randomly select one day for each month
    selected_datetimes = []
    selected_vec = []
    selected_str = []
    for dt_by_month in datetimes_by_month.values():
        if len(dt_by_month) > 0:
            month_days = list(
                set(list_dt[0].day for list_dt in dt_by_month)
            )  # get all unique days in the month
            selected_day = random.choice(month_days)  # randomly select one day
            for i, PG, time_str in dt_by_month:
                if i.day == selected_day:
                    selected_datetimes.append(i)
                    selected_vec.append(PG)
                    selected_str.append(time_str)

    unique_dates = sorted(
        list(set(i.strftime("%d/%m/%Y") for i in selected_datetimes)),
        key=lambda x: dt.datetime.strptime(x, "%d/%m/%Y"),
    )
    return (
        [selected_datetimes[i].timestamp() for i in range(len(selected_datetimes))],
        [selected_vec[i] for i in range(len(selected_vec))],
        [selected_str[i] for i in range(len(selected_str))],
        unique_dates,
    )


def n_random_hour(
    time_vector_ts, time_vector_str, vec, n_hour: int, tz, time_step: int
) -> Tuple[list, list, list, list]:
    """Randomly select n non-overlapping hours from the time vector
    Parameter :
        time_vector_ts : vector of timestamps
        time_vector_str : vector of strings corresponding to the timestamps
        vec: vector of 0/1 representing the absense/presence of a detection at the corresponding timestamp of the time_vector
        n_hour: number of hours to select
        tz : timezone object
        time_step: time bin of the time vector
    Returns :
        t: rounded Timestamp"""

    if type(tz) is not pytz._FixedOffset and tz is not pytz.utc:
        tz = pytz.timezone(tz)

    if not isinstance(n_hour, int):
        print("n_hour is not an integer")
        return

    selected_time_vector_ts, selected_dates = [], []
    while len(selected_dates) < n_hour:
        # choose a random datetime from the time vector
        rand_idx = random.randrange(len(time_vector_ts))
        rand_datetime = time_vector_ts[rand_idx]

        rand_datetime = np.round(rand_datetime / 3600) * 3600

        selected_dates.append(rand_datetime)

        # select all datetimes that fall within the hour following this datetime
        possible_datetimes = time_vector_ts[
            rand_idx : rand_idx + round(3600 / time_step) + 1
        ]

        # check if any of the selected datetimes overlap with the previously selected datetimes
        overlap = False
        for i in selected_time_vector_ts:
            if any(i <= time < i + 3600 for time in possible_datetimes):
                overlap = True
                break

        if overlap:
            continue

        # add the selected datetimes to the list
        selected_time_vector_ts.extend(possible_datetimes)

        # sort the selected datetimes in chronological order
        selected_time_vector_ts.sort()
        selected_dates.sort()

    # extract the corresponding vectors and time strings
    selected_vec = [
        vec[time_vector_ts.index(i)]
        for i in tqdm(selected_time_vector_ts, position=0, leave=True)
    ]
    selected_time_vector_str = [
        time_vector_str[time_vector_ts.index(i)]
        for i in tqdm(selected_time_vector_ts, position=0, leave=True)
    ]
    selected_dates = [
        dt.datetime.fromtimestamp(i, tz).strftime("%d/%m/%Y %H:%M:%S")
        for i in selected_dates
    ]

    return (
        selected_time_vector_ts,
        selected_time_vector_str,
        selected_vec,
        selected_dates,
    )


def pick_datetimes(
    time_vector_ts, time_vector_str, vec, selected_dates, selected_durations, TZ
) -> Tuple[list, list, list, list]:
    # user-selected datetimes from the time vector

    selected_df_out = pd.DataFrame(
        {"datetimes": selected_dates, "durations": selected_durations}
    )

    # format the datetimes and durations from strings to datetimes/timedeltas
    # selected_dates = [dt.datetime.strptime(i, '%d/%m/%Y %H:%M:%S').timestamp() for i in selected_dates]
    selected_dates = [
        pd.to_datetime(i, format="%d/%m/%Y %H:%M:%S").timestamp()
        for i in selected_dates
    ]
    timedeltas = []
    for i in selected_durations:
        if i.endswith("h"):
            timedeltas.append(dt.timedelta(hours=int(i[:-1])).total_seconds())
        elif i.endswith("m"):
            timedeltas.append(dt.timedelta(minutes=int(i[:-1])).total_seconds())
        elif i.endswith("s"):
            timedeltas.append(dt.timedelta(seconds=int(i[:-1])).total_seconds())
        elif i.endswith("d"):
            timedeltas.append(dt.timedelta(days=int(i[:-1])).total_seconds())
        else:
            print("incorrect duration format")
            return
    selected_durations = timedeltas

    selected_time_vector_ts = []

    for i in range(len(selected_dates)):
        # select all datetimes that fall within the durations following this datetime
        possible_datetimes = [
            time
            for time in time_vector_ts
            if selected_dates[i] <= time <= selected_dates[i] + selected_durations[i]
        ]

        # add the selected datetimes to the list
        selected_time_vector_ts.extend(possible_datetimes)

    # sort the selected datetimes in chronological order
    selected_time_vector_ts.sort()

    # extract the corresponding vectors and time strings
    selected_vec = [
        vec[time_vector_ts.index(i)]
        for i in tqdm(selected_time_vector_ts, position=0, leave=True)
    ]
    selected_time_vector_str = [
        time_vector_str[time_vector_ts.index(i)]
        for i in tqdm(selected_time_vector_ts, position=0, leave=True)
    ]
    selected_dates = [
        dt.datetime.fromtimestamp(i, pytz.timezone(TZ)).strftime("%d/%m/%Y %H:%M:%S")
        for i in selected_dates
    ]

    return (
        selected_time_vector_ts,
        selected_time_vector_str,
        selected_vec,
        selected_df_out,
    )
