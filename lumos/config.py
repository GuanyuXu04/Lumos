"""
Hardware and feature-selection defaults for the Lumos package.

These constants describe the physical waveguide setup. Edit this file to
match a different sensor configuration, or pass overrides directly to
WaveguideDataset (e.g. when running ablations).

Each captured optical row contains ``(num_leds + 1) * (num_pds + 1) + 1``
floats: a leading timestamp followed by every (LED, PD) cross-reading,
where index 0 of each axis is the "off" baseline and indices 1..N are
the active emitters/detectors.

"""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class HardwareConfig:
    # Physical properties of the waveguide rig.
    num_leds: int = 30
    num_pds: int = 6

    # Bounding box (mm) that depth points must fall inside to count as valid.
    x_range: Tuple[float, float] = (-10.0, 280.0)
    y_range: Tuple[float, float] = (-10.0, 280.0)
    z_range: Tuple[float, float] = (180.0, 250.0)


@dataclass(frozen=True)
class FeatureSelection:
    # Which LEDs and PDs to include as input features. By default, use all.
    leds: Tuple[int, ...] = tuple(range(1, 31))   # L1..L30
    pds: Tuple[int, ...] = tuple(range(1, 7))     # P1..P6


HARDWARE = HardwareConfig()
FEATURES = FeatureSelection()
