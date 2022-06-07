#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:49:24 2021

@author: 

Tutorial 03: Cube with 2 walls and feed-back

https://unicode-table.com/en/
"""
import numpy as np
import pandas as pd
import dm4bem
import matplotlib.pyplot as plt

Kp = 1    # P-controler gain

# Specific heat (J/kgK)

ca = 1006  # air
cc = 880   # concrete
ci = 850   # insulation
cg = 730   # glass
cw = 2700  # wood

# Density (kg/m3)

da = 1.208  # air
dc = 2300   # concrete
di = 22     # insulation
dg = 2210   # glass
dw = 532    # wood

# Lambda (W/mK)

l_air = 0.244             # air
l_concrete = 1.8        # concrete
l_insulation = 0.0345   # insulation
l_glass = 1.4               # glass
l_wood = 0.15           # wood

# épaisseurs (m)
w_wood = 0.03
w_glass = 0.02
w_concrete =0.13
w_insulation = 0.1

# Geometry
# --------
l = 7.23            # m length of the room
L=4.49              # m width of the room
H=2.40              # height oh the swedish room  
Sg = 2                # m² surface of the glass
Sc = Si = 2*(l+L)*H   # m² surface of concrete & insulation of the 5 walls

Sc1 =13.632 #c for concrete
Sc2 =14.904
Sc3 =5.664
Sc4=10.632
Si1=13.632  #i for insulation
Si3=5.664
Si4=10.632
Sg=3        # g for all glass window
Sw2=14.904  # wood inthe wall 2
Swb=2.1     # wooden bathroom's door
Swc=2.1     # wooden corridor's door



# Thermal network
A = np.zeros([33, 28])
A[0, 0] = 1
A[1, 0], A[1, 1] = -1, 1
A[2, 1], A[2, 2] = -1, 1
A[3, 2], A[3, 3] = -1, 1
A[4, 3], A[4, 4] = -1, 1
A[5, 4], A[5, 25] = -1, 1
A[6, 25], A[6, 5] = 1, -1
A[7, 5], A[7, 6] = -1, 1
A[8, 6], A[8, 7] = -1, 1
A[9, 7], A[9, 8] = -1, 1
A[10, 8], A[10,9] = -1, 1
A[11, 9], A[11, 26] = -1, 1
A[12, 11], A[12, 25] = -1, 1
A[13, 10], A[13, 11] = -1, 1
A[14, 26], A[14, 10] = -1, 1
A[15, 12] = 1
A[16, 12], A[16, 25] = -1, 1
A[17, 25], A[17, 13] = 1, -1
A[18, 13], A[18, 14] = -1, 1
A[19, 14], A[19, 15] = -1, 1
A[20, 15], A[20, 16] = -1, 1
A[21, 16], A[21, 17] = -1, 1
A[22, 17], A[22, 27] = -1, 1
A[23, 18], A[23, 25] = -1, 1
A[24, 19], A[24, 18] = -1, 1
A[25, 27], A[25, 19] = -1, 1
A[26, 26], A[26, 20] = 1, -1
A[27, 20], A[27, 21] = -1, 1
A[28, 21], A[28, 22] = -1, 1
A[29, 22], A[29, 23] = -1, 1
A[30, 23], A[30, 24] = -1, 1
A[31, 24], A[31, 27] = -1, 1
A[32,25] = 1

# Construction de G

g=np.zeros([33])

hout = 25  #coefficient convection de l’air
hin = 8 

#Mur extérieur : 
g[0] = hout * Sc1
g[1] = 2*l_concrete * Sc1 / w_concrete
g[2] = 2*l_concrete * Sc1 / w_concrete
g[3] = 2*l_insulation * Si1 / w_insulation
g[4]= 2*l_insulation * Si1 / w_insulation
g[5] = hin * Si1

# Mur pièce principale - bathroom : 
g[6] = hin * Sw2
g[7] = 2*l_wood * Sw2 / w_wood
g[8] = 2*l_wood * Sw2 / w_wood
g[9] = 2*l_concrete * Sc2 / w_concrete
g[10] = 2*l_concrete * Sc2 / w_concrete
g[11] = hin * Sc2

# Door, bathroom : 
g[12] = hin * Swb
g[13] = 2*l_wood * Swb / w_wood
g[14] = hin * Swb

#Windows : 
g[15] = hout * Sg
g[16] = hin * Sg

#Mur pièce principale - corridor : 
g[17] = hin * Si3
g[18] = 2*l_insulation * Si3 / w_insulation
g[19] = 2*l_insulation * Si3 / w_insulation
g[20] = 2*l_concrete * Sc3 / w_concrete
g[21] = 2*l_concrete * Sc3 / w_concrete
g[22] = hin * Sc3

# Door pièce principale - corridor :
g[23] = hin * Swc
g[24] = 2*l_wood * Swc / w_wood
g[25] = hin * Swc

# Mur bathroom - corridor :
g[26] = hin * Si4
g[27] = 2*l_insulation * Si4 / w_insulation
g[28] = 2*l_insulation * Si4 / w_insulation
g[29] = 2*l_concrete * Sc4 / w_concrete
g[30] = 2*l_concrete * Sc4 / w_concrete
g[31] = hin * Sc4
g[32] = Kp

G = np.diag(g)


# Construction of C

c = np.zeros([28])

# Mur extérieur : (concrete and insulation)

c[1] = cc * Sc1 * w_concrete * l_concrete
# c[3] = ci * Si1 * w_insulation * l_insulation

# Mur principal room to bathroom:
# S_concrete2 = S_wood2
c[6] = cc * Sc2 * w_concrete * l_concrete
#c[9] = cw * Sw2 * w_wood *l_wood

# Mur principal room to corridor:
# S_concrete3 =S_insulation3

#c[14] = ci * Si3 * w_insulation * l_insulation
c[17] = cc * Sc3 * w_concrete * l_concrete

# Mur bathroom to corridor:

# S_concrete4 =S_insulation4
#c[21] = ci * Si4 * w_insulation * l_insulation
c[24] = cc * Sc4 * w_concrete * l_concrete

C = np.diag(c)

# construction of b
b = np.zeros(33)
b[[0, 15, 32]] = 0 + np.array([1, 1, 20])

# construction of f
f = np.zeros(28)
# f[[0, 4, 5, 12, 13]] = 0 + np.array([1, 1, 1, 1, 1])
f[[0,4,5,12,13]] = np.array([1000,1000,500,500,1000])

# b = np.zeros(33)
# b[[0, 15, 32]] = 1

# f = np.zeros(8)
# f[[0, 4, 6, 7]] = 1

y = np.zeros(28)
y[27] = 1

u = np.hstack([b[np.nonzero(b)], f[np.nonzero(f)]])

[As, Bs, Cs, Ds] = dm4bem.tc2ss(A, G, b, C, f, y)

# yss = (-Cs @ np.linalg.inv(As) @ Bs + Ds) @ u
# ytc = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)

# print(np.array_str(yss, precision=3, suppress_small=True))
# print(np.array_str(ytc, precision=3, suppress_small=True))
# print(f'Max error in steady-state between thermal circuit and state-space:\
#  {max(abs(yss - ytc)):.2e}')
# df = pd.DataFrame(data=[yss, ytc])

dtmax = min(-2. / np.linalg.eig(As)[0])
print(f'Maximum time step: {dtmax:.2f} s')

# dt = 5
dt = 360


duration = 3600 * 24 * 0.5       # [s]

n = int(np.floor(duration / dt))

t = np.arange(0, n * dt, dt)    # time


# Vectors of state and input (in time)
n_tC = As.shape[0]              # no of state variables (temps with capacity)
# u = [To To To Tsp Phio Phii Qaux Phia]
u = np.zeros([8, n])
u[0:3, :] = np.ones([3, n])

temp_exp = np.zeros([n_tC, t.shape[0]])
temp_imp = np.zeros([n_tC, t.shape[0]])

I = np.eye(n_tC)
for k in range(n - 1):
    # temp_exp[:, k + 1] = (I + dt * As) @\
    #     temp_exp[:, k] + dt * Bs @ u[:, k]
    temp_imp[:, k + 1] = np.linalg.inv(I - dt * As) @\
        (temp_imp[:, k] + dt * Bs @ u[:, k])
        
# y_exp = Cs @ temp_exp + Ds @  u
y_imp = Cs @ temp_imp + Ds @  u

fig, ax = plt.subplots()
# ax.plot(t / 3600, y_exp.T, t / 3600, y_imp.T)
ax.plot( t / 3600, y_imp.T)
ax.set(xlabel='Time [h]',
       ylabel='$T_i$ [°C]',
       title='Step input: To = 1°C')
plt.show()


b = np.zeros(33)
b[[0, 15, 32]] = 1
f = np.zeros(28)

ytc = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)
print('Steady-state indoor temperature obtained with:')
print(f'- DAE model: {ytc[6]:.4f} °C')
print('- response to step input:', y_imp[0,len(y_imp[0])-1], '°C')

 #weather data

filename = 'FRA_Lyon.074810_IWEC.epw'
start_date = '2000-01-03 12:00:00'
end_date = '2000-02-05 18:00:00'

# Read weather data from Energyplus .epw file
[data, meta] = dm4bem.read_epw(filename, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data
weather.index = weather.index.map(lambda t: t.replace(year=2000))
weather = weather[(weather.index >= start_date) & (
    weather.index < end_date)]

surface_orientation = {'slope': 90,
                        'azimuth': 0,
                        'latitude': 45}
albedo = 0.2
rad_surf1 = dm4bem.sol_rad_tilt_surf(weather, surface_orientation, albedo)
rad_surf1['Φt1'] = rad_surf1.sum(axis=1)

data = pd.concat([weather['temp_air'], rad_surf1['Φt1']], axis=1)
data = data.resample(str(dt) + 'S').interpolate(method='linear')
data = data.rename(columns={'temp_air': 'To'})


data['Ti'] = 20 * np.ones(data.shape[0])
data['Qa'] = 0 * np.ones(data.shape[0])

t = dt * np.arange(data.shape[0])

ε_wLW = 0.9     # long wave wall emmisivity (concrete)
α_wSW = 0.2     # absortivity white surface
ε_gLW = 0.9     # long wave glass emmisivity (glass pyrex)
τ_gSW = 0.83    # short wave glass transmitance (glass)
α_gSW = 0.1     # short wave glass absortivity

σ = 5.67e-8     # W/m².K⁴ Stefan-Bolzmann constant

u = pd.concat([data['To'], data['To'], data['To'], data['Ti'],
                α_wSW * Sc1* data['Φt1'],
                τ_gSW * α_wSW * Sg * data['Φt1'],
                data['Qa'],
                α_gSW * Sc1 * data['Φt1']], axis=1)

#temp_exp = 20 * np.ones([As.shape[0], u.shape[0]])
temp_imp = np.zeros([n_tC, t.shape[0]])

for k in range(u.shape[0] - 1):
     temp_imp[:, k + 1] = np.linalg.inv(I - dt * As) @\
             (temp_imp[:, k] + dt * Bs @ u[:, k])
             
y_imp = Cs @ temp_imp + Ds @ u.to_numpy().T
q_HVAC = Kp * (data['Ti'] - y_imp[0, :])

fig, axs = plt.subplots(2, 1)
# plot indoor and outdoor temperature
axs[0].plot(t / 3600, y_imp[0, :], label='$T_{indoor}$')
axs[0].plot(t / 3600, data['To'], label='$T_{outdoor}$')
axs[0].set(xlabel='Time [h]',
            ylabel='Temperatures [°C]',
            title='Simulation for weather')
axs[0].legend(loc='upper right')

# plot total solar radiation and HVAC heat flow
axs[1].plot(t / 3600,  q_HVAC, label='$q_{HVAC}$')
axs[1].plot(t / 3600, data['Φt1'], label='$Φ_{total}$')
axs[1].set(xlabel='Time [h]',
            ylabel='Heat flows [W]')
axs[1].legend(loc='upper right')

fig.tight_layout()

input()







