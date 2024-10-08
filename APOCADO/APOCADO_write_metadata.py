import os
import glob
import json
import pandas as pd
from tqdm import tqdm
import re
from pathlib import Path
import pytz

os.chdir(r"./post_processing_detections")

# %% Write metadata files

path_csv = [
    r"L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO",
    r"Y:\Bioacoustique\APOCADO2",
    r"Z:\Bioacoustique\DATASETS\APOCADO3",
]

pattern = re.compile(r"C\d{1,2}D\d{1,2}")
matching_folders = []
for p in path_csv:
    for folder, _, _ in os.walk(p):
        if (
            pattern.match(os.path.basename(folder))
            and "archive" not in folder.lower()
            and "070722" not in folder.lower()
        ):
            matching_folders.append(folder)

(
    pamguard_csv,
    thalassa_csv,
    aplose_csv,
    file_metadata,
    metadata_spectro1,
    timestamp_csv1,
    metadata_spectro2,
    timestamp_csv2,
    ltas,
    psd,
) = ([], [], [], [], [], [], [], [], [], [])
for f in tqdm(matching_folders):
    p = glob.glob(os.path.join(f, r"pamguard\PG_rawdata_**.csv"))
    pamguard_csv.append(next(iter(p))) if p else pamguard_csv.append("")

    t = glob.glob(os.path.join(f, r"thalassa\thalassa_**.csv"))
    thalassa_csv.append(next(iter(t))) if t else thalassa_csv.append("")

    ap = glob.glob(os.path.join(f, r"aplose\**results.csv"))
    aplose_csv.append(next(iter(ap))) if ap else aplose_csv.append("")

    mt1 = glob.glob(os.path.join(f, r"**/file_metadata.csv"))
    metadata_spectro1.append(next(iter(mt1))) if mt1 else metadata_spectro1.append("")

    file_mt = os.path.join(os.path.dirname(next(iter(mt1))), "metadata.csv")
    if os.path.exists(file_mt):
        file_metadata.append(file_mt)
    else:
        raise ValueError("metadata file of original audio files error")

    ts1 = os.path.join(os.path.dirname(next(iter(mt1))), "timestamp.csv")
    if os.path.exists(ts1):
        timestamp_csv1.append(ts1)
    else:
        raise ValueError("timestamp file of original audio files error")

    mt2 = glob.glob(os.path.join(f, r"10_**/metadata.csv"))
    metadata_spectro2.append(next(iter(mt2))) if mt2 else metadata_spectro2.append("")

    ts2 = glob.glob(os.path.join(f, r"10_**/timestamp.csv"))
    timestamp_csv2.append(next(iter(ts2))) if ts2 else timestamp_csv2.append("")

    ss1 = glob.glob(os.path.join(f, r"soundscape/LTAS*.npz"))
    ltas.append(next(iter(ss1))) if ss1 else ltas.append("")

    ss2 = glob.glob(os.path.join(f, r"soundscape/PSD*.npz"))
    psd.append(next(iter(ss2))) if ss2 else ltas.append("")

deploy = pd.read_excel(
    r"L:\acoustock\Bioacoustique\DATASETS\APOCADO\PECHEURS_2022_PECHDAUPHIR_APOCADO\APOCADO - Suivi déploiements.xlsm",
    skiprows=[0],
)
deploy = deploy.loc[(deploy["check heure Raven"] == 1)].reset_index(drop=True)

deploy["duration deployment"] = [
    pd.Timestamp.combine(deploy["date recovery"][n], deploy["time recovery"][n])
    - pd.Timestamp.combine(deploy["date deployment"][n], deploy["time deployment"][n])
    for n in range(len(deploy))
]

# %%
for i in tqdm(
    range(len(matching_folders)), total=len(matching_folders), ncols=50, mininterval=1
):

    f = matching_folders[i]
    p = pamguard_csv[i]
    t = thalassa_csv[i]
    ap = aplose_csv[i]
    file_mt = file_metadata[i]
    mt1 = metadata_spectro1[i]
    ts1 = timestamp_csv1[i]
    mt2 = metadata_spectro2[i]
    ts2 = timestamp_csv2[i]
    ss1 = ltas[i]
    ss2 = psd[i]

    df_detections_extract = pd.read_csv(
        p, parse_dates=["start_datetime"], nrows=1, header="infer"
    )

    tz = pytz.FixedOffset(
        df_detections_extract["start_datetime"][0].utcoffset().total_seconds() // 60
    )

    [ID0] = list(set(df_detections_extract["dataset"]))
    platform = re.search(
        r"C\d{1,2}D\d{1,2}", ID0
    ).group()  # campaign and deployment identifier
    recorder = re.search(r"ST\d+", ID0).group().split("ST")[-1]  # instrument identifier
    ID_deploy = platform + " ST" + recorder

    rank = deploy[deploy["ID deployment"] == ID_deploy].index.item()

    dt_deployment_beg = pd.Timestamp(
        pd.Timestamp.combine(
            deploy["date deployment"][rank], deploy["time deployment"][rank]
        ),
        tz=tz,
    )
    dt_deployment_end = pd.Timestamp(
        pd.Timestamp.combine(
            deploy["date recovery"][rank], deploy["time recovery"][rank]
        ),
        tz=tz,
    )

    n_instru = deploy["recorder number"][rank]

    wav_folder = os.path.join(Path(p).parents[3], "wav")

    metadata = {
        "project": "APOCADO",
        "campaign": int(deploy["campaign"][rank]),
        "deployment": int(deploy["deployment"][rank]),
        "recorder": recorder,
        "recorder number": int(deploy["recorder number"][rank]),
        "latitude": float(deploy["latitude"][rank]),
        "longitude": float(deploy["longitude"][rank]),
        "datetime deployment": dt_deployment_beg.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "datetime recovery": dt_deployment_end.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration": str(deploy["duration deployment"][rank]),
        "vessel": deploy["vessel"][rank],
        "port": deploy["port"][rank],
        "net": deploy["net"][rank],
        "net length": int(deploy["net length"][rank]),
        "species": deploy["species"][rank],
        "wav folder": wav_folder,
        "path metadata": file_mt,
        "path origin metadata": mt1,
        "path origin timestamp": ts1,
        "path segment metadata": mt2,
        "path segment timestamp": ts2,
        "path pamguard": p,
        "path thalassa": t,
        "path LTAS": ss1,
        "path PSD": ss2,
    }

    if ap != "":
        metadata["path aplose"] = ap

    out_file = os.path.join(f, "metadata.json")

    with open(out_file, "w+") as f:
        json.dump(metadata, f, indent=1, ensure_ascii=False)
