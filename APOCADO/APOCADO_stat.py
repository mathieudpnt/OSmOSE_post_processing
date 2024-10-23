import pytz
import pandas as pd
import datetime as dt
from astral.sun import sun
import astral
import numpy as np


def suntime_hour(begin_deploy, end_deploy, lat, lon):
    """Fetch sunrise and sunset hours for dates between date_beg and date_end
    Parameters :
        date_beg : str Date in format 'YYYY-mm-dd'. Start date of when to fetch sun hour
        date_end : str Date in format 'YYYY-mm-dd'. End date of when to fetch sun hour
        lat : str latitude in Decimal Degrees
        lon : str longitude in Decimal Degrees
    Returns :
        hour_sunrise : list of float with sunrise decimal hours for each day between date_beg and date_end
        hour_sunset : list of float with sunset decimal hours for each day between date_beg and date_end
    """
    timeZ = pytz.FixedOffset(begin_deploy.utcoffset().total_seconds() // 60)
    # Infos sur la localisation
    if np.isnan(lat):
        lat = 48  # arbitrary location
    if np.isnan(lon):
        lon = -5  # arbitrary location

    gps = astral.LocationInfo(timezone=timeZ, latitude=lat, longitude=lon)

    # List of days during when the data were recorded
    list_day = pd.date_range(begin_deploy, end_deploy).date

    h_sunrise, h_sunset, dt_dusk, dt_dawn, dt_day, dt_night = [], [], [], [], [], []
    astral.Depression = 12  # nautical twilight see def here : https://www.timeanddate.com/astronomy/nautical-twilight.html

    # For each day : find time of sunset, sun rise, begin dawn and dusk
    for day in list_day:

        suntime = sun(gps.observer, date=day, dawn_dusk_depression=astral.Depression)

        dawn_dt = pd.to_datetime(suntime["dawn"]).tz_convert(timeZ)
        dusk_dt = pd.to_datetime(suntime["sunset"]).tz_convert(timeZ)
        day_dt = pd.to_datetime(suntime["sunrise"]).tz_convert(timeZ)
        night_dt = pd.to_datetime(suntime["dusk"]).tz_convert(timeZ)

        # day_hour = day_dt.hour + day_dt.minute / 60
        # night_hour = night_dt.hour + night_dt.minute / 60
        # h_sunrise.append(day_hour)
        # h_sunset.append(night_hour)

        dt_dusk.append(dusk_dt)
        dt_dawn.append(dawn_dt)
        dt_day.append(day_dt)
        dt_night.append(night_dt)
    return dt_dusk, dt_dawn, dt_day, dt_night


def stats_diel_pattern(deployment: pd.Series, detector: str):
    """Plot detection proportions for each light regime (night/dawn/day/dawn)
    Parameters :
    Returns :
    """
    assert isinstance(deployment, pd.Series), "Not a Series passed"
    assert "datetime deployment" in deployment.index and isinstance(
        deployment["datetime deployment"], pd.Timestamp
    ), "Error datetime deployment"
    assert "datetime recovery" in deployment.index and isinstance(
        deployment["datetime recovery"], pd.Timestamp
    ), "Error datetime recovery"
    assert "datetime recovery" in deployment.index and isinstance(
        deployment["datetime recovery"], pd.Timestamp
    ), "Error datetime recovery"
    assert f"df {detector}" in deployment.index and isinstance(
        deployment[f"df {detector}"], pd.DataFrame
    ), f"Error f'df {detector}'"
    assert "latitude" in deployment.index and pd.api.types.is_numeric_dtype(
        deployment["latitude"]
    ), "Error latitude"
    assert "longitude" in deployment.index and pd.api.types.is_numeric_dtype(
        deployment["longitude"]
    ), "Error longitude"

    df_detections = deployment[f"df {detector}"]
    timeZ = pytz.FixedOffset(
        deployment["datetime deployment"].utcoffset().total_seconds() // 60
    )

    # Compute sunrise and sunset decimal hour at the dataset location
    [dt_dusk, dt_dawn, dt_day, dt_night] = suntime_hour(
        begin_deploy=deployment["datetime deployment"],
        end_deploy=deployment["datetime recovery"],
        lat=deployment["latitude"],
        lon=deployment["longitude"],
    )

    # List of days in the deployment
    list_day = [dt.date(d.year, d.month, d.day) for d in dt_day]

    # Compute dusk_duration, dawn_duration, day_duration, night_duration
    # dawn_duration = [b - a for a, b in zip(dt_dawn, dt_day)]
    # day_duration = [b - a for a, b in zip(dt_day, dt_night)]
    # dusk_duration = [b - a for a, b in zip(dt_dusk, dt_night)]
    # night_duration = [dt.timedelta(hours=24) - dawn - day - dusk for dawn, day, dusk in zip(dawn_duration, day_duration, dusk_duration)]

    # Compute duration of each light regime in regards of deployement effort
    dawn_duration, dusk_duration, day_duration, night_duration = [], [], [], []
    for idx, day in enumerate(list_day):
        # duration effort on day
        beg_day = timeZ.localize(pd.Timestamp(day).normalize())  # midnight
        end_day = beg_day + pd.Timedelta(days=1)  # midnight the following day

        light_regime_day = [
            ("Night1", beg_day, dt_dawn[idx]),
            ("Dawn", dt_dawn[idx], dt_day[idx]),
            ("Day", dt_day[idx], dt_dusk[idx]),
            ("Dusk", dt_dusk[idx], dt_night[idx]),
            ("Night2", dt_night[idx], end_day),
        ]

        # Calculate duration of each light regime on day
        durations = {}

        for regime, start_time, end_time in light_regime_day:
            # Determine overlap with deployment period for each light regime
            overlap_start = max(start_time, deployment["datetime deployment"])
            overlap_end = min(end_time, deployment["datetime recovery"])

            # Calculate duration if there is an overlap
            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                durations[regime] = duration
            else:
                durations[regime] = float("NaN")

        # Combine Night1 and Night2 durations
        night1_duration = durations.get("Night1")
        night2_duration = durations.get("Night2")

        # Combine Night1 and Night2 if both are not NaN
        if pd.notna(night1_duration) or pd.notna(night2_duration):
            night_combine = (
                night1_duration
                if pd.notna(night1_duration)
                else pd.Timedelta(0, "seconds")
            ) + (
                night2_duration
                if pd.notna(night2_duration)
                else pd.Timedelta(0, "seconds")
            )
            durations["Night"] = night_combine
        else:
            durations["Night"] = float("NaN")

        # Remove Night1 and Night2 from durations
        if "Night1" in durations:
            del durations["Night1"]
        if "Night2" in durations:
            del durations["Night2"]

        dawn_duration.append(durations["Dawn"])
        day_duration.append(durations["Day"])
        dusk_duration.append(durations["Dusk"])
        night_duration.append(durations["Night"])

    # Convert to decimal hour
    dawn_duration_dec = [
        (dawn_d.total_seconds() / 3600) if pd.notna(dawn_d) else float("NaN")
        for dawn_d in dawn_duration
    ]
    day_duration_dec = [
        (day_d.total_seconds() / 3600) if pd.notna(day_d) else float("NaN")
        for day_d in day_duration
    ]
    dusk_duration_dec = [
        (dusk_d.total_seconds() / 3600) if pd.notna(dusk_d) else float("NaN")
        for dusk_d in dusk_duration
    ]
    night_duration_dec = [
        (night_d.total_seconds() / 3600) if pd.notna(night_d) else float("NaN")
        for night_d in night_duration
    ]

    night_duration_dec2 = []
    for i in range(len(night_duration_dec)):
        night_duration_dec2.append(
            np.nansum(
                [night_duration_dec[i], dusk_duration_dec[i], dawn_duration_dec[i]]
            )
        )

    # Assign a light regime to each detection
    # before : 1 = night ; 2 = dawn ; 3 = day ; 4 = dusk
    # now : 1 = night ; 2 = day
    day_det = [
        start_datetime.date() for start_datetime in df_detections["start_datetime"]
    ]
    light_regime = []
    for idx_day, day in enumerate(list_day):
        for idx_det, d in enumerate(day_det):
            # If the detection occured during 'day'
            if d == day:
                if (
                    df_detections["start_datetime"][idx_det] > dt_dawn[idx_day]
                    and df_detections["start_datetime"][idx_det] < dt_day[idx_day]
                ):
                    # lr = 2
                    lr = 1
                    light_regime.append(lr)
                elif (
                    df_detections["start_datetime"][idx_det] > dt_day[idx_day]
                    and df_detections["start_datetime"][idx_det] < dt_night[idx_day]
                ):
                    # lr = 3
                    lr = 2
                    light_regime.append(lr)
                elif (
                    df_detections["start_datetime"][idx_det] > dt_night[idx_day]
                    and df_detections["start_datetime"][idx_det] < dt_dusk[idx_day]
                ):
                    # lr = 4
                    lr = 1
                    light_regime.append(lr)
                else:
                    # lr = 1
                    lr = 1
                    light_regime.append(lr)

    # For each day, count the number of detection per light regime
    # nb_det_night, nb_det_dawn, nb_det_day, nb_det_dusk = [], [], [], []
    nb_det_night, nb_det_day = [], []
    for idx_day, day in enumerate(list_day):
        # Find index of detections that occured during 'day'
        idx_det = [idx for idx, det in enumerate(day_det) if det == day]
        if idx_det == []:
            nb_det_night.append(0)
            # nb_det_dawn.append(0)
            nb_det_day.append(0)
            # nb_det_dusk.append(0)
        else:
            # nb_det_night.append(light_regime[idx_det[0]:idx_det[-1]].count(1))
            # nb_det_dawn.append(light_regime[idx_det[0]:idx_det[-1]].count(2))
            # nb_det_day.append(light_regime[idx_det[0]:idx_det[-1]].count(3))
            # nb_det_dusk.append(light_regime[idx_det[0]:idx_det[-1]].count(4))
            nb_det_night.append(light_regime[idx_det[0] : idx_det[-1]].count(1))
            nb_det_day.append(light_regime[idx_det[0] : idx_det[-1]].count(2))

    # For each day, compute number of detection per light regime corrected by ligh regime duration
    nb_det_night_corr = [(nb / d) for nb, d in zip(nb_det_night, night_duration_dec2)]
    # nb_det_dawn_corr = [(nb / d) for nb, d in zip(nb_det_dawn, dawn_duration_dec)]
    nb_det_day_corr = [(nb / d) for nb, d in zip(nb_det_day, day_duration_dec)]
    # nb_det_dusk_corr = [(nb / d) for nb, d in zip(nb_det_dusk, dusk_duration_dec)]

    # Normalize by daily average number of detection per hour
    # av_daily_nbdet, nb_det_night_corr_norm, nb_det_dawn_corr_norm, nb_det_day_corr_norm, nb_det_dusk_corr_norm = [], [], [], [], []
    av_daily_nbdet, nb_det_night_corr_norm, nb_det_day_corr_norm = [], [], []

    for idx_day, day in enumerate(list_day):
        # Find index of detections that occured during 'day'
        idx_det = [idx for idx, det in enumerate(day_det) if det == day]

        # Compute daily average number of detections per hour
        a = len(idx_det) / (night_duration_dec2[idx_day] + day_duration_dec[idx_day])
        av_daily_nbdet.append(a)
        if a == 0:
            nb_det_night_corr_norm.append(0)
            # nb_det_dawn_corr_norm.append(0)
            nb_det_day_corr_norm.append(0)
            # nb_det_dusk_corr_norm.append(0)
        else:
            nb_det_night_corr_norm.append(nb_det_night_corr[idx_day] - a)
            # nb_det_dawn_corr_norm.append(nb_det_dawn_corr[idx_day] - a)
            nb_det_day_corr_norm.append(nb_det_day_corr[idx_day] - a)
            # nb_det_dusk_corr_norm.append(nb_det_dusk_corr[idx_day] - a)

    LIGHTR = [nb_det_night_corr_norm, nb_det_day_corr_norm]
    # LIGHTR = [nb_det_night_corr_norm, nb_det_dawn_corr_norm, nb_det_day_corr_norm, nb_det_dusk_corr_norm]
    BoxName = ["Night", "Day"]
    # BoxName = ['Night', 'Dawn', 'Day', 'Dusk']

    lr = pd.DataFrame(LIGHTR, index=BoxName).transpose()

    return lr
