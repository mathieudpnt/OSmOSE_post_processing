# Dict are easier to modify, namedtuple easier to use
navstate_mapping = {
    105: "Initializing",
    115: "Surfacing",
    119: "GPS activated",
    116: "Transmitting",
    110: "Inflecting down",
    100: "Going down",
    118: "Inflecting up",
    117: "Going up",
    120: "Landing maneuver",
    121: "Bottom landing",
    122: "Taking off",
    123: "Ballasting",
    124: "Drifting",
}

SEA034_SENSITIVITY = -165
# Specify what should be imported when using 'from glider_config import *'
__all__ = ["navstate_mapping"]
