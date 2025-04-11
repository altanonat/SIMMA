# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:08:33 2025

@author: altan
"""

import numpy as np
from math import sin,cos,exp,sqrt,pi,acos

# Define parameters
gamma = 0.0573  # in radians
nu    = 0.3  # Poisson Ratio
E     = 2 * 10**11  # Young Modulus

rry = 0.4000  # Lateral radius of roller
rrx = 0.4520  # Longitudinal radius of the roller
rwx = 0.3480  # Longitudinal radius of the wheel
rwy = 0.3000  # Lateral radius of the wheel

ss = (sin(gamma)) / (1 / ((1 / rwx) + (1 / rrx)))
rhory = 1 / rry  # Curvature of roller lateral axis
rhorx = 1 / rrx  # Curvature of roller longitudinal axis
rhowx = 1 / rwx  # Curvature of roller longitudinal axis
rhowy = 1 / rwy  # Curvature of roller longitudinal axis

G = 8 * 10**10  # Shear Modulus of rigidity in pa

mu0 = 0.4
A   = 0.6584
B   = 0.7087
kA  = 0.6258
kS  = 0.5997

freq = 8.87

# Asynchronous Motor Parameters
RC = 0.0582  # Resistance of the cable used
R1 = (0.534 / (2 * 10)) + RC
theta = acos(300 / (sqrt(3) * 21 * 41))
ZBR = (21) / (41)  # blocked rotor impedance
RBR = ZBR * cos(theta) # = R1+R2
XBR = (freq / 50) * ZBR * sin(theta)  # = R1+R2 @50 Hz
X1 = XBR * 0.4
X2 = XBR * 0.6
R2 = RBR - R1
R1 = 0.534 / (2 * 10)
XBRN = ZBR * sin(theta)  # = R1+R2 @50 Hz
I = 133
Vp = 380 / sqrt(3)
Ir = I * sin(acos(0.75))
XNLN = (Vp / Ir)
XM = (XNLN - XBRN * 0.4)
V1 = (freq / 50) * 381 / sqrt(3)
Vth = (XM / (sqrt(R1**2 + (X1 + XM)**2))) * V1
Xth = X1
Rth = ((XM / (X1 + XM))**2) * R1
wasyn = 4 * pi * freq / 10
nasyn = (60 * wasyn) / (2 * pi)

# Wheel side inertias
Jpmsmrotor = 0.95
Jpmsmpart = 0.05
Jwpart = 0.07
Jw = 4.99
Jrubber = 0.27
Jwrim = 11.53
Jwtotal = (Jpmsmrotor + Jpmsmpart + Jwpart + Jw + Jrubber + Jwrim)

# Roller side inertias
Jr = 40.6
Jasynmotor = 6.6
Jrtotal = Jr + Jasynmotor
