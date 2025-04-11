# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:35:49 2025

@author: altan
"""
import numpy as np
from math import exp, pi, sqrt, atan

def polach(a, b, xix, xiy, xis, xi, c11, c22, c23, mu, Q, fspin, G, kA, kS):
    """
    Calculates creep forces using the Polach model.

    Args:
        a: Semi-major axis of the contact ellipse.
        b: Semi-minor axis of the contact ellipse.
        xix: Creepage in the x direction.
        xiy: Creepage in the y direction.
        xis: Spin creepage.
        xi: Magnitude of creepage.
        c11, c22, c23: Contact stiffness coefficients.
        mu: Friction coefficient.
        Q: Normal force.
        fspin: Flag indicating whether to include spin effects (1 for yes, 0 for no).

    Returns:
        A tuple containing Fx and Fy (creep forces in the x and y directions).
    """
    # Constant cjj
    if xi != 0:
        cjj = sqrt((c11 * (xix / xi))**2 + (c22 * (xiy / xi))**2)
    else:
        cjj = 0

    # Contact Shear Stiffness
    C = (3/8) * (G / a) * cjj

    # Gradient of tangential stress in the area of adhesion
    epsilon = (2/3) * ((C * pi * (a**2) * b) / (Q * mu)) * xi

    # Creep Force
    F = -((2 * Q * mu) / pi) * (((kA * epsilon) / (1 + (kA * epsilon)**2)) + atan(kS * epsilon))

    # If initially creepages in all directions are equal to zero
    if xi == 0:
        Fx = 0
        Fy = 0
    else:
        Fx = F * (xix / xi)
        Fy = F * (xiy / xi)

    if fspin == 1:
        # Spin effect
        # Effect of spin on the lateral force
        if abs(xiy + (xis * a)) <= abs(xiy):
            xiyc = xiy
            xic = sqrt((xiyc**2) + (xix**2))

            epsilon = (8/3) * ((G * b * sqrt(a * b)) / (Q * mu)) * (c23 * abs(xiyc) / (1 + 6.3 * (1 - exp(-(a/b)))))
            rho = (epsilon**2 - 1) / (epsilon**2 + 1)
            Km = -epsilon * (((rho**3) / 3) - ((rho**2) / 2) + (1/6)) - ((1/3) * (sqrt((1 - (rho**2))**3)))

            # Creep force left in case of spin
            Fs = (-9/16) * (a) * (Q) * (mu) * (Km) * (1 + 6.3 * (1 - exp(-(a/b)))) * (xis / xic)
            Fy = Fy + Fs
            # Components of the creep force
            xi = xic  # Corrected creep
            xiy = xiyc  # Corrected creep in y direction
            Fx = F * (xix / xi)
            F = sqrt(Fy**2 + Fx**2)

        elif abs(xiy + (xis * a)) > abs(xiy):
            xiyc = xiy + (xis * a)
            xic = sqrt((xiyc**2) + (xix**2))
            epsilon = (8/3) * ((G * b * sqrt(a * b)) / (Q * mu)) * (c23 * abs(xiyc) / (1 + 6.3 * (1 - exp(-(a/b)))))
            rho = (epsilon**2 - 1) / (epsilon**2 + 1)
            Km = -epsilon * (((rho**3) / 3) - ((rho**2) / 2) + (1/6)) - ((1/3) * (sqrt((1 - (rho**2))**3)))
            # Creep force left in case of spin
            Fs = (-9/16) * (a) * (Q) * (mu) * (Km) * (1 + 6.3 * (1 - exp(-(a/b)))) * (xis / xic)
            Fy = Fy + Fs
            # Components of the creep force
            xi = xic  # Corrected creep
            xiy = xiyc  # Corrected creep in y direction
            Fx = F * (xix / xi)
            F = sqrt(Fy**2 + Fx**2)

    return Fx, Fy
