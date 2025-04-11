# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:33:10 2025

@author: altan
"""

import numpy as np
from math import cos, pi, sqrt
from scipy.special import ellipe, ellipk

def hertzcalc(cry, crx, cwx, cwy, nu, E, N, gamma):
    """Calculates Hertzian contact parameters (optimized)."""

    crx = crx * cos(gamma)
    cwx = cwx * cos(gamma)

    A = 1/4 * (((crx) + (cwx) + (cry) + (cwy)) - sqrt((((cwx) - (cwy))**2) +
            ((crx - cry)**2) + 2 * (((cwx - cwy)) * (crx - cry))))
    B = 1/4 * (((crx) + (cwx) + (cry) + (cwy)) + sqrt((((cwx) - (cwy))**2) +
            ((crx - cry)**2) + 2 * (((cwx - cwy)) * (crx - cry))))

    if A > B:
        A, B = B, A  # More Pythonic swap

    # Use scipy's elliptic functions
    e = 0.5
    e_prev = e + 1 # Initialize e_prev to ensure the loop starts
    
    while abs(e - e_prev) > 1e-3:
        e_prev = e
        e1 = ellipk(e**2)  # Complete elliptic integral of the first kind
        e2 = ellipe(e**2)  # Complete elliptic integral of the second kind
        e = sqrt(1 - (A / B) * (H(e1, e2, e) / D(e1, e2, e)))

    e1 = ellipk(e**2)
    e2 = ellipe(e**2)
    a = (((3 * N * D(e1, e2, e)) / (pi * A * e**2)) * (1 - (nu**2)) / E)**(1/3)
    b = a * sqrt(1 - (e**2))
    delta = e1 * (((9 * e**2) / (8 * D(e1, e2, e)))**(1/3)) * (((((1 - (nu**2)) / E)**2) *
            ((N)**2) * 8 * A / pi**2)**(1/3))

    delta0 = 0.55 * delta

    a = sqrt(delta0 / A)
    b = sqrt(delta0 / B)

    if (cwx + crx) > (cry):
        a, b = b, a

    return a, b

def H(e1, e2, e):
    return e2 - (1 - e**2) * e1

def D(e1, e2, _):
    return e1 - e2