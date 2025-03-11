# Dict are easier to modify, namedtuple easier to use
NAV_STATE = {
    100: "Going down",
    105: "Initializing",
    110: "Inflecting down",
    115: "Surfacing",
    116: "Transmitting",
    117: "Going up",
    118: "Inflecting up",
    119: "GPS activated",
    120: "Landing maneuver",
    121: "Bottom landing",
    122: "Taking off",
    123: "Ballasting",
    124: "Drifting",
}

SEA034_SENSITIVITY = -165
# Specify what should be imported when using 'from glider_config import *'
__all__ = ["NAV_STATE", "SEA034_SENSITIVITY"]
