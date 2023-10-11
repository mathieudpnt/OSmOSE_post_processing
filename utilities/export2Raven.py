import pytz
import numpy as np
from utilities.def_func import get_csv_file, sorting_detections, get_timestamps, extract_datetime, export2Raven

# Load data

files = get_csv_file(1)

df_detections, t_detections = sorting_detections(files=files, tz=pytz.UTC, box=False)
timebin_detections = t_detections['max_time'][0]
labels_detections = list(set(t_detections['labels'].explode()))
annotators_detections = list(set(t_detections['annotators'].explode()))
fmax = t_detections['max_freq'][0]

timestamps_file = get_timestamps()
wav_names = timestamps_file['filename']
wav_datetimes = [extract_datetime(d, tz=pytz.UTC) for d in timestamps_file['timestamp']]
durations = timestamps_file['duration']
wav_tuple = (wav_names, wav_datetimes, durations)

# Export detections

time_vector = [elem for i in range(len(timestamps_file)) for elem in wav_datetimes[i].timestamp() + np.arange(0, durations[i], timebin_detections).astype(int)]
time_vector_str = [str(wav_names[i]).split('.wav')[0] + '_+' + str(elem) for i in range(len(wav_names)) for elem in np.arange(0, durations[i], timebin_detections).astype(int)]

times_det_beg = [df_detections['start_datetime'][i].timestamp() for i in range(len(df_detections))]
times_det_end = [df_detections['end_datetime'][i].timestamp() for i in range(len(df_detections))]

det_vec, ranks, k = np.zeros(len(time_vector), dtype=int), [], 0
for i in range(len(times_det_beg)):
    for j in range(k, len(time_vector) - 1):
        if int(times_det_beg[i] * 1000) in range(int(time_vector[j] * 1000), int(time_vector[j + 1] * 1000)) or int(times_det_end[i] * 1000) in range(int(time_vector[j] * 1000), int(time_vector[j + 1] * 1000)):
            ranks.append(j)
            k = j
            break
        else: continue
ranks = sorted(list(set(ranks)))
det_vec[np.isin(range(len(time_vector)), ranks)] = 1

df_Raven = export2Raven(wav_tuple, time_vector, time_vector_str, 0.9 * fmax, selection_vec=det_vec)
PG2Raven_str = files[0].split('.csv')[0] + '_Raven_' + str(timebin_detections) + 's' + '.txt'
df_Raven.to_csv(PG2Raven_str, index=False, sep='\t')
