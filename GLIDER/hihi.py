import numpy as np
from pathlib import Path
import matplotlib as mpl

from utils.glider_utils import load_glider_nav, plot_nav_state

mpl.rcdefaults()
mpl.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["figure.figsize"] = [14, 6]

# Navigation data
input_dir = Path(
    r"L:\acoustock\Bioacoustique\DATASETS\GLIDER\GLIDER SEA034\MISSION_58_OHAGEODAMS_2023\APRES_MISSION\NAV"
)
df_nav = load_glider_nav(input_dir)

# LTAS data
path_LTAS = r"C:\Users\dupontma2\Downloads\LTAS_all.npz"
npz_mat = np.load(path_LTAS, allow_pickle=True)
plot_nav_state(df_nav, npz_mat)
