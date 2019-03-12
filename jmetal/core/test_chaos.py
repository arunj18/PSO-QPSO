import numpy as np
import matplotlib.pyplot as plt
import random 
#from mpl_toolkits.mplot3d import Axes3D
def lorenz(x, y, z):
    s = 10.
    r = 28.
    b = 8/3.0
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot



def lorenz_map(low, high, no_particles): # 2-D Lorenz chaos map
    x_high,x_low = (22.0,-22.0)
    y_high,y_low = (30.0,-30.0)
    z_high,z_low = (55.0,0.0)
    res_1 = []
    res_2 = []
    xs_list,ys_list,zs_list = [],[],[]
    dt = 0.01
    xs = np.empty((no_particles+1))
    ys = np.empty((no_particles+1))
    zs = np.empty((no_particles+1))
    xs[0], ys[0], zs[0] = (np.random.rand(), np.random.rand(), np.random.rand())
    for i in range(no_particles):
        # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    xs_list.extend(xs)
    ys_list.extend(ys)
    zs_list.extend(zs)
    #print(len(xs_list))
    choice = random.randint(0,2)
    if (choice==0):
        for i in range(len(xs_list)):
            res_1.append(low[0] + (high[0]-low[0])*((xs_list[i]-x_low)/(x_high-x_low)))
            res_2.append(low[1] + (high[1]-low[1])*((ys_list[i]-y_low)/(y_high-y_low)))
    elif (choice==1):
        for i in range(len(ys_list)):
            res_1.append(low[0] + (high[0]-low[0])*((ys_list[i]-y_low)/(y_high-y_low)))
            res_2.append(low[1] + (high[1]-low[1])*((zs_list[i]-z_low)/(z_high-z_low)))
    else:
        for i in range(len(zs_list)):
            res_1.append(low[0] + (high[0]-low[0])*((zs_list[i]-z_low)/(z_high-z_low)))
            res_2.append(low[1] + (high[1]-low[1])*((xs_list[i]-x_low)/(x_high-x_low)))
    plt.plot(res_1,res_2)
    result=[[],[]]
    result[0].extend(res_1)
    result[1].extend(res_2)
    plt.show()
    #print(result)
    return result
#    pass

def tent(x1):
    if x1 < 0.5:
        return 2 * x1
    return 2 * (1 - x1)
def tent_map(low,high,no_particles):
    high,low = (1.0,0.0)
    res_1, res_2 = [],[]
    xs = np.empty((no_particles+1))
    ys = np.empty((no_particles+1))
    xs[0], ys[0] = (np.random.rand(),np.random.rand())
    for i in range(1,no_particles+1):
        xs[i] = tent(xs[i-1])
        ys[i] = tent(ys[i-1])
    plt.plot(xs,ys)
    plt.show()
lorenz_map([-10.0,-2.0],[10.0,2.0],200)
tent_map(1,2,200)

