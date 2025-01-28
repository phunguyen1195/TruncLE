import multiprocessing
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from fractions import Fraction
import time
import sys
from concurrent.futures import ProcessPoolExecutor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from fractions import Fraction
import time

cores = multiprocessing.cpu_count()

dyn_lorenz = {"sigma":16.0, "R":45.92, "b": 4}
x0 = np.array([10.0, 1.0, 0.0])
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 1.0, 0.0])
v3 = np.array([0.0, 0.0, 1.0])

fp1 = np.array([0,0,0])

x_dot = []
x_norm = []
le_array = []

def lorenz (x0, dyn):
    sigma = dyn['sigma']
    R = dyn['R']
    b = dyn['b']
    x = x0[0]
    y = x0[1]
    z = x0[2]
    return np.array([sigma * (y - x), 
                     x * (R - z) - y, 
                     x * y - b * z])

def linearized_lorenz (x0, dyn, y_lorenz):
    sigma = dyn['sigma']
    R = dyn['R']
    b = dyn['b']
    x = x0[0]
    y = x0[1]
    z = x0[2]
    return np.array([sigma * (y - x), 
                     (R - y_lorenz[2])*x - y - y_lorenz[0]*z,
                      y_lorenz[1]*x + y_lorenz[0]*y - b*z])



def RungeKutta (dyn, f, dt, x0):
    k1 = f(x0, dyn)
    k2 = f(x0+0.5*k1*dt,dyn)
    k3 = f(x0 + 0.5*k2*dt, dyn)
    k4 = f(x0 + k3*dt, dyn)

    x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) *dt
    return x

def RungeKutta_linearized (dyn, f, dt, x0, y):
    k1 = f(x0, dyn, y)
    k2 = f(x0+0.5*k1*dt,dyn, y)
    k3 = f(x0 + 0.5*k2*dt, dyn, y)
    k4 = f(x0 + k3*dt, dyn, y)

    x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * dt

    return x

def f_t (dyn, f, linearized_f, dt, x0, T):
    x = np.empty(shape=(len(x0),T))
    v1_prime = np.empty(shape=(len(x0),T))
    v2_prime = np.empty(shape=(len(x0),T))
    v3_prime = np.empty(shape=(len(x0),T))
    x[:, 0] = x0
    v1_prime[:, 0] = v1
    v2_prime[:, 0] = v2
    v3_prime[:, 0] = v3
    le = np.array([0,0,0])
    for i in range(1,T):
        x[:, i] = RungeKutta(dyn, f, dt, x[:, i-1])

        v1_prime[:, i] = RungeKutta_linearized(dyn, linearized_f, dt, v1_prime[:, i-1], x[:, i-1])
        v2_prime[:, i] = RungeKutta_linearized(dyn, linearized_f, dt, v2_prime[:, i-1], x[:, i-1])
        v3_prime[:, i] = RungeKutta_linearized(dyn, linearized_f, dt, v3_prime[:, i-1], x[:, i-1])


        norm1 = np.linalg.norm(v1_prime[:, i])
        v1_prime[:, i] = v1_prime[:, i]/norm1

        GSC1 = np.dot(v2_prime[:, i], v1_prime[:, i])
        v2_prime[:, i] = v2_prime[:, i] - GSC1*v1_prime[:, i]

        norm2 = np.linalg.norm(v2_prime[:, i])
        v2_prime[:, i] = v2_prime[:, i]/norm2

        GSC2 = np.dot(v3_prime[:, i], v1_prime[:, i])
        GSC3 = np.dot(v3_prime[:, i], v2_prime[:, i])

        v3_prime[:, i] = v3_prime[:, i] - GSC2*v1_prime[:, i] - GSC3*v2_prime[:, i]
        norm3 = np.linalg.norm(v3_prime[:, i])
        v3_prime[:, i] = v3_prime[:, i]/norm3

        le = le + np.log2(np.array([norm1,norm2,norm3]))
        le_array.append(le/(i*dt))

    return x, le/(T*dt)

var_T = 1000
var_dt = 0.001
def cal_le (x):
    x_full, le = f_t(dyn_lorenz, lorenz, linearized_lorenz, var_dt, x, var_T)
    return [x_full[:,1],le]

lx = np.linspace(- 25, 25, 51)
ly = np.linspace(- 25, 25, 51)
lz = np.linspace(- 0.1, 25, 51)
X = np.array(np.meshgrid(lx,ly,lz))


X_reshaped = X.T.reshape(X.T.shape[0]*X.T.shape[1]*X.T.shape[2],3)

le_list = []
x_1 = []
points = list(X_reshaped)
with ProcessPoolExecutor(max_workers=cores-1) as executor:
    for r in executor.map(cal_le, points, chunksize=10):
        x_1.append(r[0])
        le_list.append(r[1])


zs = np.array(le_list)
x_1_list = np.array(x_1)

np.save('precal_lorenz_' +str(var_dt)+ '_' + str(var_T), zs)
np.save('precal_lorenz_points_' +str(var_dt)+ '_' + str(var_T), X_reshaped)