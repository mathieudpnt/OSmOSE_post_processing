"""
This script is used to transform a csv detection file into a Raven table
User can choose between exporting the raw detections or transform the raw detection into boxes of predefinite length

2 files are needed to execute the script
    -an Aplose formatted detection csv file
    -the corresponding file_metadata csv, this one is used to get infos from the wav such as the files list, their begin datetimes and their durations

Other important parameters to adapt
    -tz : the timezone in which the detections need to be imported from the csv -> argument tz in sorting_detections
    -timebin : the length in seconds of the raven boxes in the output table, if set to 0 the original detections will be exported (raw_detections)
    -offset : the input argument 'offset' in the export2Raven function, if set to True, the export2Raven function will return 2 outputs, df_Raven (the Raven table)
    and the offsets introduced by Raven due to the rounded durations of the wav files. If set to False, the function simply returns df_Raven and df_offset
    is set to None
    -bin_height :  it is the height of the raven boxes that are going to be exported, can be set to any user-defined value < sampling frequency
    -selection_vec : if set to True, only the boxes with detection are exported i.e. the positives. Otherwise all the boxes of length timebin are exported
    -PG2Raven_str : this string can be modified at will, this variable will be the name of the exported txt file
"""

import pytz
import os
from utilities.def_func import get_csv_file, sorting_detections, get_timestamps, extract_datetime, export2Raven

# Load data
file = get_csv_file(1)

# Import the detections in a dataframe
df_detections, t_detections = sorting_detections(file=file[0], box=True)
# df_detections, t_detections = sorting_detections(file=file[0], box=True, tz=pytz.FixedOffset(60))
fmax = t_detections['max_freq'][0]
tz_data = df_detections['start_datetime'][0].tz

timestamps_file = get_timestamps()
wav_names = timestamps_file['filename']
wav_datetimes = [extract_datetime(d, tz=tz_data) for d in timestamps_file['timestamp']]
durations = timestamps_file['duration']
wav_tuple = (wav_names, wav_datetimes, durations)

# Export detections in a Raven formatted table
timebin = 60
df_Raven, df_offset = export2Raven(df=df_detections,
                                   tuple_info=wav_tuple,
                                   timestamps=timestamps_file,
                                   offset=False,
                                   timebin_new=timebin,
                                   bin_height=1.5 * fmax,
                                   selection_vec=True)

PG2Raven_str = file[0].split('.csv')[0] + f'_{timebin}s_testtt' + '.txt'
df_Raven.to_csv(PG2Raven_str, index=False, sep='\t')
print(os.path.basename(PG2Raven_str), ' exported in ', os.path.dirname(PG2Raven_str))
