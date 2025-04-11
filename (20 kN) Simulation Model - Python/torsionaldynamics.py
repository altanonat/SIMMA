# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:40:54 2025

@author: altan
"""

import numpy as np
statedot = np.zeros(2)  # Initialize statedot as a NumPy array
def torsionaldynamics(Fx, Tasyn, Tpmsm, rwx, rrx, Jwtotal, Jrtotal):
    """
    Calculates the derivatives of the states in a torsional dynamics system.

    Args:
        Fx: Force in the x direction.
        Tasyn: Asynchronous motor torque.
        Tpmsm: PMSM torque.
        rwx: Wheel radius.
        rrx: Rotor radius.
        Jwtotal: Total wheel inertia.
        Jrtotal: Total rotor inertia.

    Returns:
        statedot: A NumPy array containing the derivatives of the states.
    """
    statedot[0] = (Tasyn + (Fx * rrx)) / Jrtotal
    statedot[1] = (-Tpmsm - (Fx * rwx)) / Jwtotal
    return statedot