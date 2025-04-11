# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:48:40 2025

@author: altan
"""

from math import sqrt

def kalkercoefficients(a, b, nu):
    """
    Calculates Kalker's coefficients c11, c22, c23, and c33.

    Args:
        a: Semi-major axis of the contact ellipse.
        b: Semi-minor axis of the contact ellipse.
        nu: Poisson's ratio.

    Returns:
        A tuple containing c11, c22, c23, and c33.
    """
    abratio = b / a

    # Kalker's Coefficients c11
    k1 = 2.3464 + 1.5443 * nu + 7.9577 * (nu**2)
    k2 = 0.961669 - 0.043513 * nu + 2.402357 * (nu**2)
    k3 = -0.0160185 + 0.0055475 * nu - 0.0741104 * (nu**2)
    k4 = 0.10563 + 0.61285 * nu - 7.26904 * (nu**2)
    c11 = (k1) + (k2 / abratio) + (k3 / (abratio**2)) + (k4 / sqrt(abratio))

    # Kalker's Coefficients c22
    k1 = 2.34641 - 0.27993 * nu + 0.19763 * (nu**2)
    k2 = 0.96167 + 0.52684 * nu + 1.22642 * (nu**2)
    k3 = -0.0160185 - 0.0126292 * nu - 0.0011272 * (nu**2)
    k4 = 0.10563 + 0.78197 * nu - 1.12348 * (nu**2)
    c22 = (k1) + (k2 / abratio) + (k3 / (abratio**2)) + (k4 / sqrt(abratio))

    # Kalker's Coefficients c23
    k1 = 0.29677 + 0.22524 * nu + 0.71899 * (nu**2)
    k2 = 1.01321 + 0.20407 * nu - 0.72375 * (nu**2)
    k3 = 0.0092415 + 0.0854262 * nu + 0.319940 * (nu**2)
    k4 = (8.4835e-4) - (3.211e-3 * nu) - (1.7484e-2 * (nu**2))
    c23 = (k1) + (k2 / abratio) + (k3 / (abratio**2)) + (k4 / (abratio**3))

    # Kalker's Coefficients c33
    k1 = 0.72795 - 1.00202 * nu - 0.32695 * (nu**2)
    k2 = 0.461755 + 1.002340 * nu + 0.081441 * (nu**2)
    k3 = 0.023739 - 0.110640 * nu + 0.249008 * (nu**2)
    k4 = -0.0012999 + 0.0063653 * nu - 0.0129114 * (nu**2)
    c33 = (k1) + (k2 * abratio) + (k3 * (abratio**2)) + (k4 * (abratio**3))

    return c11, c22, c23, c33