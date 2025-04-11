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
    if 0 <= time <= 0.5:
        Tpmsm = 0.0
    elif 0.5 < time <= 33.805:
        Tpmsm = ((785.55)/(33.805-0.5))*time-\
            ((785.55*0.5)/(33.805-0.5))
    elif 33.805 < time <= 34.40:
        Tpmsm = 785.55
    elif 34.40 < time <= 39.305:
        Tpmsm = ((785.55)/(34.40-39.305))*time-\
            ((785.55*39.305)/(34.40-39.305))
    elif time > 39.305 and time <= 39.5 + 10**-8:
        Tpmsm = 0.0
    else:
        Tpmsm =0.0 # handles cases outside of the defined ranges.
    Tpmsm = -Tpmsm
    return Tpmsm