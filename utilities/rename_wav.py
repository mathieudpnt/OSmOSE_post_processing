import datetime as dt
import os
from post_processing_detections.utilities.def_func import read_header, extract_datetime, get_wav_info
from tkinter import filedialog
from tkinter import Tk
import glob
from tqdm import tqdm

#WAV files 
root = Tk()
root.withdraw()
wavpath = filedialog.askdirectory(title = 'Select wav folder')
wav_files = glob.glob(os.path.join(wavpath, "**/*.wav"), recursive=True)
wav_files.sort(key=lambda x: os.path.getmtime(x)) #sort by order or file creation date
wav_list = [os.path.basename(file) for file in wav_files]
wav_folder = [os.path.dirname(file) for file in wav_files]
wav_datetimes = [extract_datetime(file) for file in wav_list] #datetime of wav files
# durations = get_wav_info(wavpath)
durations = [read_header(file)[-1] for file in wav_files] #slower than fet_wav_info
wav_tuple = (wav_list, wav_datetimes, durations)

var1 = dt.datetime(2023, 2, 6 , 13, 40, 24, 2199*100)
var2 = dt.datetime(2014, 3, 13 , 12, 37, 40, 4617*100)
offset = round((var1-var2).total_seconds())

# date1 = dt.datetime.strptime('19740815_100032', '%Y%m%d_%H%M%S')
# date2 = dt.datetime.strptime('20230125_155632', '%Y%m%d_%H%M%S')


# wav_ts = [i.timestamp() for i in wav_datetimes[217:238]]
wav_ts = [i.timestamp() for i in wav_datetimes]
# wav_datetimes2 = [dt.datetime.fromtimestamp(i+offset) for i in wav_ts ]
wav_datetimes2 = wav_datetimes
for i in range(217, 238):
    wav_datetimes2[i] = dt.datetime.fromtimestamp(wav_ts[i]+offset)


wav_str2 = [dt.datetime.strftime(i, '7194.%y%m%d%H%M%S.wav') for i in wav_datetimes2[217:]]

# for index, file in enumerate(wav_files[217:238]):
#     os.rename(file, os.path.join(path, ''.join([str(index), '.jpg'])))
    
wav_files2 = wav_files[217:]
[os.rename(wav_files2[i], os.path.join(wav_folder[1], wav_str2[i]))  for i in tqdm(range(len(wav_files2)))]