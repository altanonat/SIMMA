# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:37:24 2025

@author: altan
"""
import numpy as np
from math import pi

def asynchronousmotor(omegar, nasyn, Vth, R2, wasyn, Rth, Xth, X2):
    """
    Calculates the asynchronous motor torque.

    Args:
        omegar: Rotor angular speed (rad/s).
        nasyn: Synchronous speed (rev/min).
        Vth: Thevenin voltage (V).
        R2: Rotor resistance (Ohms).
        wasyn: Synchronous angular speed (rad/s).
        Rth: Thevenin resistance (Ohms).
        Xth: Thevenin reactance (Ohms).
        X2: Rotor reactance (Ohms).

    Returns:
        Tasyn: Asynchronous motor torque (Nm).
    """
    n = (omegar * 60) / (2 * pi)
    s = (nasyn - n) / nasyn
    Tasyn = (3 / wasyn) * ((Vth**2 * (R2 / s)) / ((Rth + R2 / s)**2 + (Xth + X2)**2))
    return Tasyn