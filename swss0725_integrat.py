# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:10:05 2022

@author: hua08
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def func(x):
    """  """
    return np.cos(x)+x*np.sin(x)
def func_del(x):
    """  """
    return x*np.cos(x)


def RHS(y,t):
    return -2*y

y0=3 # y(t0)=3
t0=0
tf=2

time=np.linspace(t0,tf)
y_true=odeint(RHS, y0, time)
#solution =odeint(func, y0, t)
#%%
fig=plt.figure()
plt.plot(time,y_true,'k-o',linewidth=2)
plt.grid()
plt.xlabel('time')
plt.ylabel(r'$y(T)$')
plt.legend('truth')
#sys.exit()
#%% 1st order

h=.2
current_time=t0
y_current=y0
timeline=np.array([t0])
sol_rk1=np.array([y0])


# y_curr= y_pri+ h*RHS(t_pri)
while current_time < tf-h:
    k1=RHS(y_current, current_time)
    y_next=y_current+h*k1

    sol_rk1=np.append(sol_rk1,y_next)
    y_current=y_next
    
    current_time=current_time+h
    timeline=np.append(timeline, current_time)

#%% 2nd order

h=.2
current_time=t0
y_current=y0
timeline=np.array([t0])
sol_rk2=np.array([y0])


# y_curr= y_pri+ h*RHS(t_pri)
while current_time < tf-h:
    
    # solve ODE
    k1=RHS(y_current, current_time)
    k2 =RHS(y_current+k1*h/2, current_time+h/2)
    y_next=y_current+h*k2
    
    # save 
    sol_rk2=np.append(sol_rk2,y_next)
    y_current=y_next
    
    # forward
    current_time=current_time+h
    timeline=np.append(timeline, current_time)
 
#%% 4th order

h=.2
current_time=t0
y_current=y0
timeline=np.array([t0])
sol_rk4=np.array([y0])

# y_curr= y_pri+ h*RHS(t_pri)
while current_time < tf-h:
    
    # solve ODE
    k1 =RHS(y_current, current_time)
    k2 =RHS(y_current+k1*h/2, current_time+h/2)
    k3 =RHS(y_current+k2*h/2, current_time+h/2)
    k4 =RHS(y_current+k3*h,current_time+h)
    
    y_next=y_current+(k1+2*k2+2*k3+k4)*h/6
    
    # save 
    sol_rk4=np.append(sol_rk4,y_next)
    y_current=y_next
    
    # forward
    current_time=current_time+h
    timeline=np.append(timeline, current_time)

#%% plot
fig=plt.figure()
plt.plot(time,y_true,'k-',linewidth=2)
plt.plot(timeline, sol_rk1)
plt.plot(timeline, sol_rk2)
plt.plot(timeline, sol_rk4)
plt.grid()
plt.xlabel('time')
plt.ylabel(r'$y(T)$')
plt.legend(['truth', 'Runge-Kutta 1st', 'Runge-Kutta 2nd', 'Runge-Kutta 4th'])
sys.exit()
