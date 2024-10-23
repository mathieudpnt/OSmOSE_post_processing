import os
import glob
import pandas as pd
from tqdm import tqdm

# from scripts.soundscape import deployment
from utils.Deployment import Deployment
from deployment_utils import get_agreement, plot_single, get_perf

path_acoustock = [
    r"L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO",
    r"Y:\Bioacoustique\APOCADO2",
    r"Z:\Bioacoustique\DATASETS\APOCADO3",
]
list_json = [
    file_path
    for path in path_acoustock
    for file_path in glob.glob(
        os.path.join(path, r"**\**metadata.json"), recursive=True
    )
]

# %%
deploy = []
for json in tqdm(list_json):
    deploy.append(Deployment(path_json=json))
deployment_names = [d.name for d in deploy]
# with open(os.path.join(r'L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO', 'data_Deployment.pkl'), 'wb') as f:
#     pickle.dump(deploy, f)
# %%
# path_test = r"C:\Users\dupontma2\Desktop\data_local\files\C2D1 ST335556632\C2D1 ST335556632 - origin metadata file.csv"
# path_test1 = r"C:\Users\dupontma2\Desktop\data_local\files\C2D1 ST335556632\C2D1 ST335556632 - origin file_metadata.csv"
# path_test2 = r"C:\Users\dupontma2\Desktop\data_local\files\C2D1 ST335556632\C2D1 ST335556632 - origin file_metadata.csv"
# path_test3 = r"C:\Users\dupontma2\Desktop\data_local\files\C2D1 ST335556632\C2D1 ST335556632 - segment metadata file.csv"
# path_test4 = r"C:\Users\dupontma2\Desktop\data_local\files\C2D1 ST335556632\C2D1 ST335556632 - segment timestamp file.csv"
# path_test5 = r"C:\Users\dupontma2\Desktop\data_local\files\C2D1 ST335556632\C2D1 ST335556632 - pamguard.csv"
# path_test6 = r"C:\Users\dupontma2\Desktop\data_local\files\C2D1 ST335556632\C2D1 ST335556632 - thalassa.csv"
# path_test7 = r"C:\Users\dupontma2\Desktop\data_local\files\C2D1 ST335556632\C2D1 ST335556632 - aplose.csv"

# test2 = Deployment(
#     campaign="Point A",
#     deployment="phase B",
#     recorder="hihu",
#     path_metadata=path_test,
#     path_origin_metadata=path_test1,
#     path_origin_timestamp=path_test2,
#     path_segment_metadata=path_test3,
#     path_segment_timestamp=path_test4,
#     path_pamguard=path_test5,
#     path_thalassa=path_test6,
#     path_aplose=path_test7,
# )
print(deploy[37])


# %%
get_agreement(
    data=deploy[37],
    file="aplose",
    timebin=10,
    begin_date=pd.Timestamp("2022-07-06 23:59:47 +0200"),
    end_date=pd.Timestamp("2022-07-08 01:59:28 +0200"),
)
# %%
plot_single(
    data=deploy[37],
    file="aplose",
    timebin=10,
)
# %%
get_perf(
    data=deploy,
    file=["aplose", "pamguard"],
    timebin=60,
)
