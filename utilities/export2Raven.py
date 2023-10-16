import pytz
from utilities.def_func import get_csv_file, sorting_detections, get_timestamps, extract_datetime, export2Raven

# Load data

file = get_csv_file(1)

df_detections, t_detections = sorting_detections(file=file[0], box=True, tz=pytz.FixedOffset(120))
fmax = t_detections['max_freq'][0]
tz_data = df_detections['start_datetime'][0].tz

timestamps_file = get_timestamps()
wav_names = timestamps_file['filename']
wav_datetimes = [extract_datetime(d, tz=tz_data) for d in timestamps_file['timestamp']]
durations = timestamps_file['duration']
wav_tuple = (wav_names, wav_datetimes, durations)

# Export detections
timebin = 60
df_Raven = export2Raven(df=df_detections,
                        tuple_info=wav_tuple,
                        timestamps=timestamps_file,
                        offset=False,
                        timebin_new=timebin,
                        bin_height=1.5 * fmax,
                        selection_vec=True)

PG2Raven_str = file[0].split('.csv')[0] + f'_{timebin}s_positives' + '.txt'
df_Raven.to_csv(PG2Raven_str, index=False, sep='\t')
