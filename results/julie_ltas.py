from tqdm import tqdm
import json
from pathlib import Path
from utilities.def_func import extract_datetime
import pandas as pd
from utilities.LTAS import LTAS
import pytz


raven_signal = r'L:\acoustock2\Bioacoustique\HADOCC\ANALYSE\LTAS\signal_duty_cycle_OFF.ravensignal'
raven_ltas = r'L:\acoustock2\Bioacoustique\HADOCC\ANALYSE\LTAS\20240617_155800_duty_cycle_OFF.ravenltsa'
# raven_ltas = r'L:\acoustock2\Bioacoustique\HADOCC\ANALYSE\LTAS\20240621_092307_duty_cycle_ON.ravenltsa'
raven_ltas_csv = r'L:\acoustock2\Bioacoustique\HADOCC\ANALYSE\LTAS\ltsa_duty_cycle_OFF.csv'

with open(raven_ltas, 'r', errors='ignore') as file:
    for line in file:
        clean_line = line.strip()
        if 'timeResolution' in clean_line:
            parts = line.split(':')
            if len(parts) > 1:
                time_resolution_LTAS = int(parts[1].strip().strip('",'))
        if 'frequencyResolution' in clean_line:
            parts = line.split(':')
            if len(parts) > 1:
                frequency_resolution_LTAS = int(parts[1].strip().strip('",'))
                break

with open(raven_signal, 'rb') as f:
    text = f.read()
path = Path(json.loads(text.decode('latin-1'))['fileList'][0]['filePath']['path'])

begin_LTAS = extract_datetime(path.stem, tz=pytz.UTC)


LTAS_deploy = LTAS(path=raven_ltas_csv,
                   t_res=time_resolution_LTAS,
                   f_res=frequency_resolution_LTAS,
                   begin_datetime=begin_LTAS,
                   duty_cycle=25,
                   sensitivity=143
                   )

print(LTAS_deploy)

LTAS_deploy.plot_LTAS(dyn_min=30,
                      dyn_max=90
                      )

LTAS_deploy.plot_PSD(output_path=None, output_name=None)
