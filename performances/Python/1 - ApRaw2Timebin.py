import os
from post_processing_detections.utilities.def_func import get_detection_files, sorting_detections, t_rounder, get_timestamps

#%% LOAD DATA - User inputs

files_list = get_detection_files(1)
timestamps_file = get_timestamps(tz='Etc/GMT-1')
wav_names = timestamps_file['filename']
wav_datetimes = timestamps_file['timestamp']
df_detections, t_detections = sorting_detections(files_list)
timebin = int(t_detections['max_time'][0])

## EXPORT RESHAPPED DETECTIONS

# APLOSE FORMAT
dataset_str = list(set(df_detections['dataset']))
PG2Ap_str = "/PG_formatteddata_" + t_rounder(timestamps_file['timestamp'][0], res=600).strftime('%y%m%d') + '_' + t_rounder(timestamps_file['timestamp'].iloc[-1], res=600).strftime('%y%m%d') +'_'+ str(t_detections['max_time'][0]) + 's'+ '.csv'

df_detections.to_csv(os.path.dirname(files_list[0]) + PG2Ap_str, index=False)  
print('\n\nAplose formatted data file exported to '+ os.path.dirname(files_list[0]))

# # RAVEN FORMAT
# df_PG2Raven = pd.DataFrame()

# df_PG2Raven['Selection'] = np.arange(1,len(df_detections)+1)
# df_PG2Raven['View'], df_PG2Raven['Channel'] = [1]*len(df_detections), [1]*len(df_detections)

# datetime_begfiles, datetime_endfiles = [],[]
# for i in range(len(wav_names)):
#     datetime_begfiles.append((wav_datetimes[i]).strftime('%Y-%m-%d %H:%M:%S.%f'))
#     datetime_endfiles.append((wav_datetimes[i]+dt.timedelta(seconds=timebin)).strftime('%Y-%m-%d %H:%M:%S.%f'))


# offsets =[]
# for i in range(len(datetime_endfiles)-1):
#     offsets.append(((wav_datetimes[i]+dt.timedelta(seconds=timebin)).timestamp() - (wav_datetimes[i+1]).timestamp()))
#     offsets_cumsum=(list(np.cumsum([offsets[i] for i in range(len(offsets))])))
#     offsets_cumsum.insert(0, 0)

# test3 = [wav_names[i].split('.wav')[0] for i in range(len(wav_names))] #names of the waves without extension
# start_datetime, end_datetime = [],[] 
# for i in range(len(time_vector)):
#     if PG_vec[i] == 1:
#         test4 = [time_vector_str[i].split('_+')[0] == test3[j] for j in range(len(wav_list))] #finding which wav the detection is belonging to
#         idx_wav_Raven = [i for i, x in enumerate(test4) if x][0] #index of the wav the detection i is belonging to
#         start_datetime.append(int(time_vector[i] - wav_datetimes[0].timestamp())      + offsets_cumsum[idx_wav_Raven] )
#         end_datetime.append(int(time_vector[i] - wav_datetimes[0].timestamp())+10     + offsets_cumsum[idx_wav_Raven] )

# df_PG2Raven['Begin Time (s)'] = start_datetime     
# df_PG2Raven['End Time (s)'] = end_datetime     

# df_PG2Raven['Low Freq (Hz)'] = [0]*len(start_datetime_str)
# df_PG2Raven['High Freq (Hz)'] = [0.8*fmax]*len(start_datetime_str)

# PG2Raven_str = "/PG_formatteddata_" + t_rounder(wav_datetimes[0]).strftime('%y%m%d') + '_' + t_rounder(wav_datetimes[-1]).strftime('%y%m%d') + '_'+ str(time_bin_duration) + 's' + '.txt'

# df_PG2Raven.to_csv(os.path.dirname(pamguard_path) + PG2Raven_str, index=False, sep='\t')  
# print('\n\nRaven formatted data file exported to '+ os.path.dirname(pamguard_path))




