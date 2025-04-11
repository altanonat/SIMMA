# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:41:39 2025

@author: altan
"""

import numpy as np
from math import exp
from polach import polach  # Import your polach function
from fastsim import fastsim  # Import your fastsim function
from asynchronousmotor import asynchronousmotor  # Import your asynchronousmotor function
from pmsmtorque import pmsmtorque        # Import your pmsmtorque function
from torsionaldynamics import torsionaldynamics # Import your torsionaldynamics function
import numba

fast_polach = numba.njit(fastmath=True)(polach)
def integration(state,statedot,a, b, c11,\
                              c22, c23, nasyn, Vth, R2, wasyn, Rth, Xth, \
                                  X2, rwx, rrx, Jwtotal, Jrtotal, \
                                      mu0, A, B, deltat, time, \
                                          ss, Ncap, G, kA, kS):
    """
    Performs numerical integration using an improved I-TR-BDF2 method.

    Args:
        state: Current state vector.
        a, b, c11, c22, c23, nasyn, Vth, R2, wasyn, Rth, Xth, X2, rwx, rrx, Jwtotal, Jrtotal, mu0, A, B, deltat, time, N, ss, tanSel: System parameters.

    Returns:
        state: Updated state vector.
    """

    timeg  = time + 0.5 * deltat
    stateg = state + 0.5 * deltat * statedot
    omegar = stateg[0]
    omegaw = stateg[1]

    # Translational velocities
    vr = omegar * rrx
    vw = omegaw * rwx

    # Tangential force in longitudinal direction
    s = (2 * (vr - vw) / (vr + vw))  # creepage
    # Slip velocity
    w = vw * s
    mu = mu0 * ((1 - A) * exp(-B * w) + A)
    
    Fx, _ = fast_polach(a, b, s, 0, ss, abs(s), c11, c22, c23, mu, Ncap, 1, G, kA, kS)

    Tasyn = asynchronousmotor(omegar, nasyn, Vth, R2, wasyn, Rth, Xth, X2)
    Tpmsm = pmsmtorque(timeg)
    statedotg = torsionaldynamics(Fx, Tasyn, Tpmsm, rwx, rrx, Jwtotal, Jrtotal)
    statedotg = (statedotg+statedot)/2 
    xkgamma = state + 0.5 * (deltat / 2) * (statedot + statedotg)
    time    = time + deltat
    state1  = state + deltat * statedot
    omegar  = state1[0]
    omegaw  = state1[1]

    # Translational velocities
    vr = omegar * rrx
    vw = omegaw * rwx

    # Tangential force in longitudinal direction
    s = (2 * (vr - vw) / (vr + vw))  # creepage
    # Slip velocity
    w = vw * s
    mu = mu0 * ((1 - A) * exp(-B * w) + A)
    
    Fx, _ = fast_polach(a, b, s, 0, ss, abs(s), c11, c22, c23, mu, Ncap, 1, G, kA, kS)

    Tasyn = asynchronousmotor(omegar, nasyn, Vth, R2, wasyn, Rth, Xth, X2)
    Tpmsm = pmsmtorque(time)
    statedot1 = torsionaldynamics(Fx, Tasyn, Tpmsm, rwx, rrx, Jwtotal, Jrtotal)
    statedot1 = (statedot1+statedot)/2
    state = (1 / (0.5 * (2 - 0.5))) * xkgamma - (((1 - 0.5)**2) / (0.5 * (2 - 0.5))) * state + ((1 - 0.5) / (2 - 0.5)) * deltat * statedot1
    return state, time
