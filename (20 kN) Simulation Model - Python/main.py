# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:06:02 2025

@author: altan
"""

import numpy as np
import scipy.io as sio
from timeit import default_timer as timer
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
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker

# Data import - İmport measurements
data  = sio.loadmat('10v20kn.mat')
vals  = data['Data'][0, 0]
keys  = data['Data'][0, 0].dtype.descr
key   = keys[3][0]
val   = np.squeeze(vals[key][0][0])
keyff = keys[3][0]
keyf  = keys[2][0]
key   = keys[0][0]
valff = np.squeeze(vals[keyff][0][0]) # Doubly filtered measurements
val   = np.squeeze(vals[key][0][0]) # Measurements
valf  = np.squeeze(vals[keyf][0][0]) #Single filtered Measurements

runtime = 39.5
omegar  = 22.0755
omegaw  = 28.6980
ome1s   = valff['ome1'][35649:43550]
ome2s   = -valff['ome2'][35649:43550]
times   = valff['t'][35649:43550]-178.25
cFmeas  = valff['T'][35649:43550]

deltat = 0.0005
ni     = int(round(runtime / deltat) + 1)
ns     = 2

# (Initialization)
state    = np.array([omegar, omegaw])
statedot = np.zeros(2)

timeSys  = 0.0
tcntr    = 0
tplot    = np.zeros((ni,1))
tplot[0] = timeSys

# Pre-allocate arrays for speed
stateplot    = np.zeros((ns, ni))
torqueplot   = np.zeros((ni, 1))
creepfplot   = np.zeros((ni, 1))
statedotplot = np.zeros((ns, ni))
statep       = np.zeros((1, 2))
splot        = np.zeros((ni,1))
statepdot    = np.zeros((1, 2))

a,b             = hertzcalc(rhory,rhorx,rhowx,rhowy,nu,E,N,gamma)
c11,c22,c23,c33 = kalkercoefficients(a,b,nu)

statep[:, 0] = omegar
statep[:, 1] = omegaw

# Runtime measurement
start_time = timer()
for i in range(ni):
    vr     = omegar * rrx
    vw     = omegaw * rwx
    slip   = 2 * (vr - vw) / (vr + vw)
    
    w  = vw * slip
    mu = mu0 * ((1 - A) * exp(-B * w) + A)
    
    Fx, Fy  = polach(a,b,slip,0,ss,abs(slip),c11,c22,c23,mu,N,1, G, kA, kS)
    
    Tasyn = asynchronousmotor(statep[0, 0], nasyn, Vth, R2, wasyn, Rth, Xth, X2)
    Tpmsm = pmsmtorque(timeSys)
    
    stateplot[:, i]    = statep[0, :]
    torqueplot[i, 0]   = Tpmsm
    creepfplot[i, 0]   = Fx
    statedotplot[:, i] = statepdot[0, :]
    splot[i,0]         = slip
    tplot[tcntr]       = timeSys

    statepdot[0, :]      = torsionaldynamics(Fx, Tasyn, Tpmsm, rwx, rrx, \
                                        Jwtotal, Jrtotal)
    statep[0, :],timeSys = integration(statep[0, :], statepdot[0, :], \
                                a, b, c11, c22, c23, nasyn, Vth, R2, \
                                wasyn, Rth, Xth, X2, rwx, rrx, Jwtotal, \
                                Jrtotal, mu0, A, B, deltat, timeSys,\
                                ss, N, G, kA, kS)
    omegar = statep[0, 0]
    omegaw = statep[0, 1]
    tcntr  = tcntr+1

end_time = timer()
print(f"Elapsed time: {end_time - start_time:.4f} seconds")

fig = plt.figure(1)
plt.plot(tplot, stateplot[0,:], label='Model', color='black', linestyle='-', linewidth=0.5)
plt.plot(times, ome2s, label='Measurement', color='grey', linestyle='--', linewidth=2, alpha=0.5)
plt.title('Roller Angular Velocity')
plt.xlabel('Time')
plt.ylabel(r'$\omega_{r}$')
# plt.title('Cost Function Values for Different Normal Load Parameters')
plt.legend(fontsize='small')
plt.grid(True)
ax = plt.gca()
plt.ylim([22.0, 22.8])
tick_spacing = 0.2
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# --- Save the figure as a PDF with specified width and forced aspect ratio ---
width_cm = 8
width_in = width_cm / 2.54  # Convert cm to inches
# To keep the original aspect ratio, we don't need to force the height.
# We can just set the width and let matplotlib determine the height.
fig.set_size_inches(width_in, fig.get_size_inches()[1] * (width_in / fig.get_size_inches()[0]))
################################
plt.savefig('Roller_Angular_Velocity_20kN.pdf', bbox_inches='tight',  pad_inches=0.1)  # Save as PDF
plt.show()
################################
fig = plt.figure(2)
plt.plot(tplot, stateplot[1,:], '--', label='Model', color='black', linestyle='-', linewidth=0.5)
plt.plot(times, ome1s, label='Measurement', color='grey', linestyle='--', linewidth=2, alpha=0.5)
plt.title('Wheel Angular Velocity')
plt.xlabel('Time')
# plt.ylim([14.2, 15.0])
plt.ylabel(r'$\omega_{w}$')
# plt.title('Cost Function Values for Different Normal Load Parameters')
plt.legend(fontsize='small')
plt.grid(True)
plt.ylim([28.6, 29.6])
tick_spacing = 0.2
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# --- Save the figure as a PDF with specified width and forced aspect ratio ---
width_cm = 8
width_in = width_cm / 2.54  # Convert cm to inches

# To keep the original aspect ratio, we don't need to force the height.
# We can just set the width and let matplotlib determine the height.
fig.set_size_inches(width_in, fig.get_size_inches()[1] * (width_in / fig.get_size_inches()[0]))
################################
plt.savefig('Wheel_Angular_Velocity_20kN.pdf', bbox_inches='tight',pad_inches=0.1)  # Save as PDF
plt.show()
################################
smeas  = 2*(((ome2s*rrx))-(ome1s*rwx))/((ome1s*rwx)+(ome2s*rrx))
statemodel = np.zeros((43550-35649,1))
serror = np.zeros((43550-35649,1))
k = 0

for c in range(len(tplot)):
    if (c % (0.005 / deltat)) <= 10**-4 and c!=0:
        statemodel[k,0] = 2 * (((stateplot[0, c] * rrx) - (stateplot[1, c] * rwx)) /
                             ((stateplot[0, c] * rrx) + (stateplot[1, c] * rwx)))
        k += 1

# Ensure statemodel and statemeas are the same length
# statemodel = statemodel[:len(smeas)] #Trims statemodel if it is longer than statemeas.
serror = np.abs(statemodel - smeas)
################################
plt.figure(3)
plt.plot(times, serror, '--', linewidth=1)
plt.grid(True)
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Force scientific notation
# Save a single array
################################
np.save('times.npy', times)
np.save('serror_10kN.npy', serror)
# plt.plot(times, ome1s, linewidth=2)
plt.title('Creepage Error')
################################
plt.figure(4)
plt.plot(times, statemodel, '--', linewidth=1)
plt.plot(times, smeas, linewidth=2)
plt.grid(True)
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Force scientific notation
################################
np.save('smeas_10kN.npy', smeas)
np.save('statemodel_10kN.npy', statemodel)
plt.title('Creepages')
################################
fig = plt.figure(5)
plt.plot(tplot, -torqueplot, '--', color='black', linestyle='-', linewidth=1)
plt.title('PMSM Torque Request (Nm)')
plt.xlabel('Time')
# plt.ylim([14.2, 15.0])
plt.ylabel(r'$T_{pmsm}$')
# plt.title('Cost Function Values for Different Normal Load Parameters')
# plt.legend(fontsize='small')
plt.grid(True)
# --- Save the figure as a PDF with specified width and forced aspect ratio ---
width_cm = 8
width_in = width_cm / 2.54  # Convert cm to inches

# To keep the original aspect ratio, we don't need to force the height.
# We can just set the width and let matplotlib determine the height.
fig.set_size_inches(width_in, fig.get_size_inches()[1] * (width_in / fig.get_size_inches()[0]))
################################
plt.savefig('PMSM_Torque_Request_20kN.pdf', bbox_inches='tight',pad_inches=0.1)  # Save as PDF
plt.show()
################################
fig = plt.figure(6)
plt.plot(tplot, creepfplot, label='Model', color='black', linestyle='--', linewidth=0.5)
plt.plot(times, cFmeas, label='Measurement', color='grey', linestyle='-', linewidth=1, alpha=0.5)
plt.title('Creep Force (N)')
plt.xlabel('Time')
# plt.ylim([14.2, 15.0])
plt.ylabel(r'Creep Force')
# plt.title('Cost Function Values for Different Normal Load Parameters')
plt.legend(fontsize='small', loc='best')
plt.grid(True)
plt.ylim([0, 2500])
tick_spacing = 500
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# --- Save the figure as a PDF with specified width and forced aspect ratio ---
width_cm = 8
width_in = width_cm / 2.54  # Convert cm to inches

# To keep the original aspect ratio, we don't need to force the height.
# We can just set the width and let matplotlib determine the height.
fig.set_size_inches(width_in, fig.get_size_inches()[1] * (width_in / fig.get_size_inches()[0]))
################################
plt.savefig('Creep_Force_20kN.pdf', bbox_inches='tight',pad_inches=0.1)  # Save as PDF
plt.show()
