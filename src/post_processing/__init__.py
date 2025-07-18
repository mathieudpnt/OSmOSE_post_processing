import logging
import sys

logger = logging.getLogger("root")
logger.setLevel(logging.WARNING)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.handlers[0].formatter = logging.Formatter("%(message)s")
