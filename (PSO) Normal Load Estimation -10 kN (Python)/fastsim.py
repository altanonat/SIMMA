# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:37:02 2025

@author: altan
"""

import numpy as np
import math


def fastsim(a, b, nx, ny, xix, xiy, xis, c11, c22, c23, N, mu, G):
    """
    Calculates creep forces using the FASTSIM method.

    Args:
        a: Semi-major axis of the contact ellipse.
        b: Semi-minor axis of the contact ellipse.
        nx: Number of elements in the x direction.
        ny: Number of elements in the y direction.
        xix: Creepage in the x direction.
        xiy: Creepage in the y direction.
        xis: Spin creepage.
        c11, c22, c23: Contact stiffness coefficients.
        N: Normal force.
        mu: Friction coefficient.

    Returns:
        A tuple containing Fx and Fy (creep forces in the x and y directions).
    """

    dy = 2 * b / ny
    Fx = 0
    Fy = 0
    Lx = (8 * a) / (3 * G * c11)
    Ly = (8 * a) / (3 * G * c22)
    Ls = (np.pi * a**2) / (4 * G * c23 * np.sqrt(a * b))

    for j in range(1, ny + 1):  # Python range is exclusive of the upper bound
        # y coordinate of the points
        y = -b + (j - 1/2) * dy
        ay = a * np.sqrt(1 - (y / b)**2)
        # Distance between consecutive points in y direction
        dx = 2 * ay / nx
        px = 0
        py = 0
        for i in range(1, nx + 1):
            # x coordinate of the points
            x = ay - (i - 1/2) * dx
            # Rigid Slips
            wx = (xix / Lx) - (y * xis / Ls)
            wy = (xiy / Ly) + ((x + dx / 2) * xis / Ls)

            # Pressure Distribution
            pz = ((2 * N) / (np.pi * a * b)) * (1 - (x**2 / a**2) - (y**2 / b**2))

            pHx = px - dx * wx
            pHy = py - dx * wy
            pH = np.sqrt(pHx**2 + pHy**2)
            if pH <= mu * pz:
                px = pHx
                py = pHy
            else:
                px = mu * pz * (pHx / pH)
                py = mu * pz * (pHy / pH)
            Fx = Fx + px * dx * dy
            Fy = Fy + py * dx * dy

    return Fx, Fy