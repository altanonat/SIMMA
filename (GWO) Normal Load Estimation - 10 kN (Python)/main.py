# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:06:02 2025

@author: altan
"""

import numpy as np
import scipy.io as sio
import time
# from timeit import default_timer as timer
from math import exp
import matplotlib.pyplot as plt
from parameters import *
from hertzcalc import hertzcalc
from polach import polach
from fastsim import fastsim
from asynchronousmotor import asynchronousmotor
from pmsmtorque import pmsmtorque
from torsionaldynamics import torsionaldynamics
from integration import integration
from kalkercoefficients import kalkercoefficients
import scipy.signal as signal
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import numba

fast_polach = numba.njit(fastmath=True)(polach)

# Set the random seed based on the year
np.random.seed(2025)

# Particle Swarm Parameters
Nmin = 4200
Nmax = 60000

## Change this variable to set number of models
nop  = 5 # Numberof Particles

# (Variable initialization - vectorized where possible)
Ncap    = np.float32(np.linspace(Nmin, Nmax, nop))  # More efficient than loop
Ncap0   = (Nmin + Nmax) / 2
Ncapest = Ncap0
Velp    = np.float32(np.full(nop, (Nmax - Nmin) / (nop - 1)))

# Initialize Alpha, Beta, and Delta positions and scores
Alpha_pos = np.zeros(1)
Alpha_score = float('inf')  # Use -inf for maximization problems

Beta_pos = np.zeros(1)
Beta_score = float('inf')  # Use -inf for maximization problems

Delta_pos = np.zeros(1)
Delta_score = float('inf')  # Use -inf for maximization problems

costfp    = np.float32(np.zeros(nop))
costgbest = 1e5

tidx   = 1

# Data import - İmport the measurements
data  = sio.loadmat('5v10kn.mat')
vals  = data['Data'][0, 0]
keys  = data['Data'][0, 0].dtype.descr
key   = keys[3][0]
val   = np.squeeze(vals[key][0][0])
keyff = keys[3][0]
keyf  = keys[2][0]
valff = np.squeeze(vals[keyff][0][0])
valf  = np.squeeze(vals[keyf][0][0])

runtime = 42.5
omegar  = 11.144
omegaw  = 14.485
ome1s   = np.float32(valff['ome1'][18499:27000])
ome2s   = np.float32(-valff['ome2'][18499:27000])
times   = np.float32(valff['t'][18499:27000]-92.5)

deltat = 0.0005
ni     = int(round(runtime / deltat) + 1)
ns     = 2

# (Initialization)
state    = np.float32(np.array([omegar, omegaw]))
statedot = np.float32(np.zeros(2))

timeSys  = 0.0
tcntr    = 0
tplot    = np.float32(np.zeros(ni))
tplot[0] = timeSys

a,b             = hertzcalc(rhory,rhorx,rhowx,rhowy,nu,E,Ncapest,gamma);
c11,c22,c23,c33 = kalkercoefficients(a,b,nu);

# Pre-allocate arrays for speed
stateplot    = np.float32(np.zeros((ns, ni)))
statedotplot = np.float32(np.zeros((ns, ni)))
splot        = np.float32(np.zeros(ni))
Ncapplot     = np.float32(np.zeros(ni))

statep       = np.zeros((nop, 2))
statep[:, 0] = omegar
statep[:, 1] = omegaw
statepdot    = np.zeros((nop, 2))
sp           = np.zeros(nop)

idx = 0
# Runtime measurement
start_time = time.monotonic_ns()
for i in range(ni):
    stateplot[:, i]    = statep[idx, :]
    statedotplot[:, i] = statepdot[idx, :]
    splot[i]           = sp[idx]
    Ncapplot[i]        = Ncap[idx]
    for j in range(nop):
        vr    = statep[j, 0] * rrx
        vw    = statep[j, 1] * rwx
        s     = 2 * (vr - vw) / (vr + vw)
        w = vw * sp[j]
        mu = mu0 * ((1 - A) * exp(-B * w) + A)
        
        a, b    = hertzcalc(rhory, rhorx, rhowx, rhowy, nu, E, Ncap[j], gamma)
        c11,c22,c23,c33 = kalkercoefficients(a,b,nu);
        Fx, Fy  = fast_polach(a,b,s,0,ss,abs(s),c11,c22,c23,mu,Ncap[j],1, G, kA, kS)
        
        Tasyn = asynchronousmotor(statep[j, 0], nasyn, Vth, R2, wasyn, Rth, Xth, X2)
        Tpmsm = pmsmtorque(timeSys)

        statepdot[j, :] = torsionaldynamics(Fx, Tasyn, Tpmsm, rwx, rrx, \
                                            Jwtotal, Jrtotal)
        statep[j, :],timeSys    = integration(statep[j, :], statepdot[j, :], \
                                    a, b, c11, c22, c23, nasyn, Vth, R2, \
                                    wasyn, Rth, Xth, X2, rwx, rrx, Jwtotal, \
                                    Jrtotal, mu0, A, B, deltat, timeSys,\
                                    ss, Ncap[j], G, kA, kS)
        
        sp[j]  = 2 * (statep[j,0] * rrx - statep[j, 1] * rwx) / (statep[j, 1] * rwx + statep[j, 0] * rrx)
        timeSys = timeSys - deltat
    
    # Measurement update part
    if (abs(timeSys - times[tidx]) <= 1e-4) and timeSys!=0:
        smeas   = 2 * (ome2s[tidx] * rrx - ome1s[tidx] * rwx) / (ome1s[tidx] * rwx + ome2s[tidx] * rrx)
        costfp  = np.abs(sp - smeas)
        idx     = np.argmin(costfp)
        Ncapest = Ncap[idx]
        
        # Update Alpha, Beta, and Delta
        for k in range(nop):
            if costfp[k] < Alpha_score:
                Alpha_score = costfp[k]  # Update Alpha
                Alpha_pos   = Ncap[k]
    
            if costfp[k] > Alpha_score and costfp[k] < Beta_score:
                Beta_score = costfp[k]  # Update Beta
                Beta_pos   = Ncap[k]
    
            if costfp[k] > Alpha_score and costfp[k] > Beta_score and costfp[k] < Delta_score:
                Delta_score = costfp[k]  # Update Delta
                Delta_pos   = Ncap[k]
        
        # # Update positions
        Xpos = (Alpha_pos + Beta_pos + Delta_pos) / 3
    
        # # Reset mechanism
        d   = Xpos - Ncap
        r1p = np.random.normal(0,1,nop)

        if np.mean(abs(d)) < 1e-20:
            Ncap   = np.linspace(Nmin, Nmax, nop)
            costfp = np.full(nop, 1e5)
            # Reset Alpha, Beta, and Delta
            Alpha_pos   = np.zeros(1)
            Alpha_score = float('inf')  # Use -inf for maximization problems
    
            Beta_pos   = np.zeros(1)
            Beta_score = float('inf')  # Use -inf for maximization problems
    
            Delta_pos   = np.zeros(1)
            Delta_score = float('inf')  # Use -inf for maximization problems
            
        else:
            Ncap = Ncap+r1p*d
            Ncap = np.clip(Ncap, Nmin, Nmax)
        
        statep[:, 0] = ome2s[tidx]
        statep[:, 1] = ome1s[tidx]
        
        tidx = tidx+ 1

    tplot[tcntr] = timeSys
    timeSys      = timeSys+deltat
    if tcntr == ni:  # fixed index error
        break
    
    tcntr    = tcntr+1
    end_time = time.monotonic_ns()
print(f"Elapsed time: {(end_time - start_time)/1e9:.4f} seconds")

fig   = plt.figure(1)
Nreal = 10000 * np.ones(tplot.shape)
plt.plot(tplot, Ncapplot, '--', label='Estimation', color='black', linestyle='-', linewidth=1)
plt.plot(tplot, Nreal, label='Real Value', color='grey', linestyle='--', linewidth=1)
plt.title('Normal Load Estimation')
plt.xlabel('Time')
plt.ylabel(r'$\hat{N}$ (Newtons)')
plt.ylim([0, 60000])
plt.legend(fontsize='small', loc='best')
plt.grid(True)
ax = plt.gca()
tick_spacing = 10000
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Force scientific notation
# --- Save the figure as a PDF with specified width and forced aspect ratio ---
width_cm = 8
width_in = width_cm / 2.54  # Convert cm to inches

# To keep the original aspect ratio, we don't need to force the height.
# We can just set the width and let matplotlib determine the height.
fig.set_size_inches(width_in, fig.get_size_inches()[1] * (width_in / fig.get_size_inches()[0]))

plt.savefig('Ncap_5Mod_GWO_10kN.pdf', bbox_inches='tight',pad_inches=0.1)  # Save as PDF
plt.show()

# Moving average window with 1 second window length
window_size = 2001
if window_size % 2 == 0:
    window_size += 1 # Ensure odd window size for correct centering.
window = np.ones(window_size) / window_size
smoothed_Ncapplot = np.convolve(Ncapplot, window, mode='same')

fig   = plt.figure(2)
plt.plot(tplot, smoothed_Ncapplot, '--', label='Estimation', color='black', linestyle='-', linewidth=1)
plt.plot(tplot, Nreal, label='Real Value', color='grey', linestyle='--', linewidth=1)
plt.title('Normal Load Estimation')
plt.xlabel('Time')
plt.ylabel(r'$\hat{N}$ (Newtons)')
plt.xlim([25, 34])
plt.ylim([0, 20000])
plt.legend(fontsize='small', loc='best')
plt.grid(True)
ax = plt.gca()
ticky_spacing = 5000
tickx_spacing = 1
ax.yaxis.set_major_locator(ticker.MultipleLocator(ticky_spacing))
ax.xaxis.set_major_locator(ticker.MultipleLocator(tickx_spacing))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Force scientific notation
# --- Save the figure as a PDF with specified width and forced aspect ratio ---
width_cm = 8
width_in = width_cm / 2.54  # Convert cm to inches

# To keep the original aspect ratio, we don't need to force the height.
# We can just set the width and let matplotlib determine the height.
fig.set_size_inches(width_in, fig.get_size_inches()[1] * (width_in / fig.get_size_inches()[0]))

plt.savefig('Ncap_Filtrered_5Mod_GWO_10kN.pdf', bbox_inches='tight',pad_inches=0.1)  # Save as PDF
plt.show()

plt.figure(3)
plt.plot(tplot, stateplot[0,:], '--', linewidth=1)
plt.plot(times, ome2s, linewidth=2)
plt.title('Roller Angular Velocity')

plt.figure(4)
plt.plot(tplot, stateplot[1,:], '--', linewidth=1)
plt.plot(times, ome1s, linewidth=2)
plt.title('Roller Angular Velocity')

rmse = np.sqrt(np.mean((Nreal[50000:68000] - Ncapplot[50000:68000]) ** 2))

print(f"Root Mean Squared Error (RMSE): {rmse}")