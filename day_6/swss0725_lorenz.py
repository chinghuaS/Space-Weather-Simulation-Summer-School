# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:08:23 2022

@author: hua08
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# x_dot2+g*sin(x)/l=0
# x=[x1(angle), x2(state)]
# x1=theta; x2=theta_dot
x0=np.array([np.pi/3,0])
l=3
g=9.81
damp=.3
t=np.linspace(0,15,1000)

def pendulum(x, t):
    xdot=np.zeros(2)
    xdot[0]=x[1]
    xdot[1]=-g*np.sin(x[0])/l
    return xdot

def pendulum_damp(x,t):
    xdot=np.zeros(2)
    xdot[0]=x[1]
    xdot[1]=-g/l*np.sin(x[0]) - damp*x[1]
    return xdot

# solve ODE
y=odeint(pendulum, x0, t)
y_d=odeint(pendulum_damp, x0, t)

#%% plot
fig, ax =plt.subplots(2,1)
ax[0].plot(y)
ax[1].plot(y_d)



#%% Lorenz36(x,t, sigma ,ro, beta)

#sol=odeint(func,initial, time, arg=(sigma ,ro, beta))

t=np.linspace(0, 20, 1000)
x0=np.array([5,5,5])
sigma=10
ro=28
beta=8/3


def Lorenz36(x, t, sigma, ro, beta):
    xdot=np.zeros(3)
    xdot[0]=sigma*(x[1]-x[0])
    xdot[1]=x[0]*(ro-x[2])-x[1]
    xdot[2]=x[0]*x[1]-beta*x[2]
    return xdot
#%%
y_l36=odeint(Lorenz36, x0, t, args=(sigma, ro, beta))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(y_l36.T[0],y_l36.T[1],y_l36.T[2])

#%%

for i in range(20):
    xr=[np.random.uniform(-20,20),np.random.uniform(-30,30),np.random.uniform(0,50)]
    yr_l36=odeint(Lorenz36, xr, t, args=(sigma, ro, beta))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(yr_l36.T[0],yr_l36.T[1],yr_l36.T[2])



