"""Use this script to convert the detection of an APLOSE formatted result file to another timebin.
It it also possible to filter out results based on several other parameters (see sorting_detections parameters).
"""
from pathlib import Path
from utils.def_func import sort_detections, read_yaml

parameters_dict = read_yaml(Path(r".\utils\export_result_file.yaml"))
df = sort_detections(**parameters_dict)

# %% Save DataFrame to csv

directory = parameters_dict['file'].parent
filename = (parameters_dict['file'].stem + '_' + str(parameters_dict['timebin_new']) + 's.csv')
df.to_csv(directory / filename, index=False, sep=",")
