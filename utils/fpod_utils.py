import pandas as pd
import pytz
from OSmOSE.utils.timestamp_utils import strftime_osmose_format, strptime_from_text


def fpod2aplose(
    df: pd.DataFrame,
    tz: pytz.BaseTzInfo,
    dataset_name: str,
    annotation: str,
    bin_size: int = 60,
) -> pd.DataFrame:
    """
    From FPOD result DataFrame to APLOSE formatted DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        FPOD result dataframe
    tz: pytz.BaseTzInfo
        Timezone object to get non-naïve datetimes
    dataset_name: str
        dataset name
    annotation: str
        annotation name
    bin_size: int
        Duration of the detections in seconds

     Returns
    -------
    pd.DataFrame
        An APLOSE formatted DataFrame
    """
    fpod_start_dt = sorted(
        [
            tz.localize(strptime_from_text(entry, "%d/%m/%Y %H:%M"))
            for entry in df["Date heure"]
        ]
    )

    fpod_end_dt = sorted(
        [entry + pd.Timedelta(seconds=bin_size) for entry in fpod_start_dt]
    )

    data = {
        "dataset": [dataset_name] * len(df),
        "filename": [""] * len(df),
        "start_time": [0] * len(df),
        "end_time": [bin_size] * len(df),
        "start_frequency": [0] * len(df),
        "end_frequency": [0] * len(df),
        "annotation": [annotation] * len(df),
        "annotator": ["FPOD"] * len(df),
        "start_datetime": [strftime_osmose_format(entry) for entry in fpod_start_dt],
        "end_datetime": [strftime_osmose_format(entry) for entry in fpod_end_dt],
    }

    return pd.DataFrame(data)

def cpod2aplose(
    df: pd.DataFrame,
    tz: pytz.BaseTzInfo,
    dataset_name: str,
    annotation: str,
    bin_size: int = 60,
    extra_columns: list = None,
) -> pd.DataFrame:
    """
    From CPOD result DataFrame to APLOSE formatted DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        CPOD result dataframe
    tz: pytz.BaseTzInfo
        Timezone object to get non-naïve datetimes
    dataset_name: str
        dataset name
    annotation: str
        annotation name
    bin_size: int, optional
        Duration of the detections in seconds
    extra_columns: list, optional
        Additional columns added from df to data

     Returns
    -------
    pd.DataFrame
        An APLOSE formatted DataFrame
    """
    data = fpod2aplose(df,tz,dataset_name,annotation,bin_size)
    data['annotator'] = data.loc[data['annotator'] == 'FPOD'] = 'CPOD'
    if extra_columns:
        for col in extra_columns:
            if col in df.columns:
                data[col] = df[col].tolist()
            else:
                print(f"Avertissement : La colonne '{col}' n'existe pas dans df et sera ignorée.")
    return pd.DataFrame(data)


def meta_cut_aplose(
        d_meta:pd.DataFrame,
        df:pd.DataFrame
)-> pd.DataFrame:
    """
    From APLOSE formatted DataFrame with all rows to filtered DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        CPOD result dataframe
    d_meta: pd.DataFrame
        Metadata dataframe with deployments information (previously exported as json)
     Returns
    -------
    pd.DataFrame
        An APLOSE formatted DataFrame with data from beginning to end of the deployement.
        Returns the percentage of usable datas.
    """
    d_meta.loc[:,['deployment_date','recovery_date']] = d_meta[['deployment_date','recovery_date']].apply(pd.to_datetime)
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], format="%Y-%m-%dT%H:%M:%S.%f%z")

    # Add DPM column
    df['DPM'] = (df['Nfiltered']>0).astype(int)

    # Extract corresponding line
    campaign = df.iloc[0]['dataset']
    phase = d_meta.loc[d_meta['name'] == campaign].reset_index()
    start_date = phase.loc[0, 'deployment_date']
    end_date = phase.loc[0, 'recovery_date']
    df = df[(df['start_datetime'] >= start_date) & (df['start_datetime'] <= end_date)].copy()

    # Calculate the percentage of collected data on the phase length of time
    if df.empty:
        percentage_on = 0
        print("No data for this phase")
    else:
        df_end = df.loc[df.index[-1], 'start_datetime']
        df_start = df.loc[df.index[0], 'start_datetime']
        act_length = df_end - df_start
        p_length = end_date - start_date
        percentage_data = act_length * 100 / p_length
        on = int(df.loc[df.MinsOn == 1, 'MinsOn'].count())
        percentage_on = percentage_data * (on/ len(df))

    print(f"Pourcentage de données exploitables : {percentage_on}%")
    return df