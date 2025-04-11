# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:38:19 2025

@author: altan
"""

def pmsmtorque(time):
    """
    Calculates the PMSM torque based on time and a selection parameter.

    Args:
        time: The time value.

    Returns:
        Tpmsm: The PMSM torque.
    """

    if 0 <= time <= 1.15:
        Tpmsm = 0.0
    elif 1.15 < time <= 34.455:
        Tpmsm = ((785.6) / (34.455 - 1.15)) * time - \
                ((785.6 * 1.15) / (34.455 - 1.15))
    elif 34.455 < time <= 35.15:
        Tpmsm = 785.6
    elif 35.15 < time <= 40.105:
        Tpmsm = ((785.6) / (35.15 - 40.105)) * time - \
                ((785.6 * 40.105) / (35.15 - 40.105))
    elif time > 40.105 and time <= 42.5 + 10**-8:
        Tpmsm = 0.0
    else:
        Tpmsm =0.0 # handles cases outside of the defined ranges.
    Tpmsm = -Tpmsm
    return Tpmsm