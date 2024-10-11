import matplotlib as mpl
mpl.rcdefaults()
mpl.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["figure.figsize"] = [10, 6]

from utils import def_func, audio_utils, glider_utils
__all__ = ["def_func", "audio_utils", "glider_utils"]