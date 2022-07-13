import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *

def FuncOrbite(t, y, mu):
    r = np.copy(y[:3])
    v = np.copy(y[3:])
    return np.concatenate((v, -mu * r / np.linalg.norm(r)**3), axis=0)

def Euler(h, y, N, func):
    sol = np.zeros((N, y.size))
    sol[0] = y
    T = np.linspace(0, N*h, N)
    for t in range(1, N):
        k1 = func(T[t-1], y, mu)
        y = y + h * k1
        sol[t] = y
    return sol

def RK4(h, y, N, func):
    sol = np.zeros((N, y.size))
    sol[0] = y
    T = np.linspace(0, N*h, N)
    for t in range(1, N):
        k1 = func(t, y, mu)
        k2 = func(t + h/2, y + h * k1 / 2, mu)
        k3 = func(t + h/2, y + h * k2 / 2, mu)
        k4 = func(t + h, y + h * k3, mu)
        y = y + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
        sol[t] = y
    return sol

mu = 3.986004418e14
R = 6371000
y0 = np.array([6.8e6, 0, 0, 0, 0, 7656.2204773874])
N = 50000
h = 5000 / N
T = np.linspace(0, N*h, N)
model = RK4(h, y0, N, FuncOrbite)
radius_model = np.array([sqrt(model[i, 0]**2 + model[i, 1]**2 + model[i, 2]**2) for i in range(len(model[:,0]))])

positions = pd.read_csv("positions.csv")
atmosphere = pd.read_csv("atmosphere.csv")
radius = pd.Series([sqrt(positions.iloc[i, 1]**2 + positions.iloc[i, 2]**2 + positions.iloc[i, 3]**2) for i in range(len(positions.iloc[:,0]))])
height = pd.Series([radius.iat[i] - 6370000 for i in range(len(radius))])
fuu = []
for i in range(len(height)):
    for j in range(1390, 1410):
        if height[i] < atmosphere.iloc[j, 0]:
            fuu.append(atmosphere.iloc[j, 1])
            break
atmos2 = pd.Series([fuu[i] for i in range(len(fuu))])
deriv = [7656.2204773874]
for i in range(len(radius) - 1):
    deriv.append(sqrt((positions.iloc[i + 1, 1] - positions.iloc[i, 1])**2 + (positions.iloc[i + 1, 2] - positions.iloc[i, 2])**2 + (positions.iloc[i + 1, 3] - positions.iloc[i, 3])**2) / (positions.iloc[i+1, 0] - positions.iloc[i, 0]))
velocity = pd.Series(deriv)
positions = pd.concat([positions, radius, height, atmos2, velocity], axis=1)
print(model[-1, 0] - positions.iloc[-1, 1])
print(model[-1, 2] - positions.iloc[-1, 3])
print(positions.iloc[:, 4:])

T_red = np.linspace(0, N*h, len(radius))
radius_model_red = np.zeros(len(T_red))
model_red = np.zeros((len(T_red), 3))
k = 0
for i in range(len(T)):
    if T[i] >= T_red[k]:
        radius_model_red[k] = radius_model[i]
        model_red[k, :] = model[i, :3]
        k += 1
difference_rad = np.array([radius_model_red - positions.iloc[:, 4]])
difference = np.array([np.linalg.norm(model_red[i, :] - positions.iloc[i, 1:4]) for i in range(len(model_red))])
np.savetxt("difference_rad.txt", difference_rad)
np.savetxt("difference.txt", difference_rad)

fig1, ax1 = plt.subplots()
ax1.plot(model[:, 0], model[:, 2], label='Model')
ax1.plot(positions.iloc[:, 1], positions.iloc[:, 3], label='Data')
ax1.legend()
fig2, ax2 = plt.subplots()
ax2.plot(T, radius_model, label='Model')
ax2.plot(positions.iloc[:, 0], positions.iloc[:, 4], label='Data')
ax2.legend()
fig3, ax3 = plt.subplots()
ax3.plot(T_red, radius_model_red - positions.iloc[:, 4], label='Diff_rad')
ax3.plot(T_red, difference, label='Diff')
ax3.legend()
plt.show()