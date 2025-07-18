import logging
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

logger = logging.getLogger("root")
logger.setLevel(logging.WARNING)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.handlers[0].formatter = logging.Formatter("%(message)s")

mpl.rcdefaults()
plt.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["figure.figsize"] = [14, 6]
