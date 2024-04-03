import os
from post_processing_detections.utilities.def_func import read_header, sorting_annot_boxes
from tkinter import filedialog
from tkinter import Tk
import glob
from pathlib import Path
import json

#%% User input
root = Tk()
root.withdraw()
# folder1 = filedialog.askdirectory(initialdir='L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO', title="Select wav folder")
folder1 = 'L:/acoustock/Bioacoustique/DATASETS/APOCADO/PECHEURS_2022_PECHDAUPHIR_APOCADO/Campagne 2/IROISE/335556632/wav'
root = Tk()
root.withdraw()
# FilePath1 = filedialog.askopenfilename(initialdir = os.path.dirname(folder1), title="Select result file 1", filetypes=[("CSV files", "*.csv")])
FilePath1 = filedialog.askopenfilename(initialdir = os.path.dirname(folder1)+'/analysis', title="Select result file 1", filetypes=[("CSV files", "*.csv")])

t1_detections = sorting_annot_boxes(FilePath1)

time_bin = t1_detections[0]
fmax = t1_detections[1]
annotators = t1_detections[2]
labels = t1_detections[3]
df1_detections = t1_detections[-1]

wav_files1 = glob.glob(os.path.join(Path(FilePath1).parents[2], "**/*.wav"), recursive=True)
wav_names1 = [os.path.basename(file) for file in wav_files1]
test_wav1 = [j in sorted(list(set([i.split('_')[0] for i in df1_detections['filename']]))) for j in [i.split('.wav')[0] for i in wav_names1]]
wav_names1, wav_files1 = zip(*[(wav_names1[i], wav_files1[i]) for i in range(len(wav_names1)) if test_wav1[i]]) #only the wav files corresponding to the detections are kept
durations = [read_header(file)[-1] for file in wav_files1]


metadata =  {'wav_folder': folder1, 'wav_path': wav_files1, 'durations': durations, 'detection_file': FilePath1, 'fmax': fmax, 'timebin': time_bin, 'annotators': annotators, 'labels': labels}

out_file = open(os.path.join(Path(FilePath1).parents[0], 'metadata.json'), 'w+')
json.dump(metadata, out_file, indent=4)
out_file.close()



