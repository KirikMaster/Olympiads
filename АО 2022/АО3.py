import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import *

def func3(t, y):
    r = np.copy(y[:2])
    v = np.copy(y[2:])
    v_norm = norm(v)
    return np.concatenate((v, [(-f*v[1] - alpha*v_norm*v[0])/m, (-m*g + f*v[0] - alpha*v_norm*v[1])/m]), axis=0)

def Euler(h, y, N, func):
    sol = np.zeros((N, y.size))
    sol[0] = y
    T = np.linspace(0, N*h, N)
    for t in range(1, N):
        k1 = func(T[t-1], y)
        y = y + h * k1
        sol[t] = y
    return sol

def RK4(h, y, N, func):
    sol = np.zeros((N, y.size))
    sol[0] = y
    T = np.linspace(0, N*h, N)
    for t in range(1, N):
        k1 = func(t, y)
        k2 = func(t + h/2, y + h * k1 / 2)
        k3 = func(t + h/2, y + h * k2 / 2)
        k4 = func(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
        sol[t] = y
    return sol

m = 0.005 # кг - масса цилиндра
D = 0.05 # м - диаметр цилиндра
L = 0.2 # м - длина цилиндра
rho = 1.2 # кг/м^3 - плотность воздуха
g = 9.8 # м/с^2 - ускорение свободного падения
n = 10 # об/сек - частота вращения цилиндра
w = 2 * pi * n # с^-1 - угловая скорость цилиндра
S = D * L # м^2 - площадь поперечного сечения цилиндра
f = rho * D**2 * L * w
alpha = rho / 2 * S

y0 = np.array([0, 0, 5, 0])
N = 500
h = 0.01
solution = RK4(h, y0, N, func3)

fig, ax = plt.subplots(2, 1)
ax[0].plot(solution[:, 0], solution[:, 1], label='coordinate')
ax[1].plot(solution[:, 0], np.sqrt(solution[:, 2]**2 + solution[:, 3]**2), label='velocity')
ax[0].grid()
ax[1].grid()
ax[0].set_title('Траектория движения цилиндра')
ax[1].set_title('Годограф скорости цилиндра')
ax[0].set_xlabel('x', size=12)
ax[0].set_ylabel('y', size=12)
ax[1].set_xlabel('x', size=12)
ax[1].set_ylabel('v', size=12)
plt.show()