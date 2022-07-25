# -*- coding: utf-8 -*-
"""
0725
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    """  """
    return np.cos(x)+x*np.sin(x)
def func_del(x):
    """  """
    return x*np.cos(x)


x=np.linspace(-6,6,1000)

y=func(x)
y2=func_del(x)

# fig, fg=plt.subplots(1,1) # create an emty fig
# fg.plot(x,y);
# fg.plot(x,y2);



#%% forward: current to following
der_arr=np.array([])
x_arr=np.array([])

h=.25   # step_size
x_der=-6    # initial point

def func_d(x,h):
    """slope func"""
    return (func(x+h)-func(x))/h


while x_der<=6:
    y_der=func_d(x_der,h)
    der_arr=np.append(der_arr,y_der)
    x_arr=np.append(x_arr,x_der)
    x_der=x_der+h   # go forward
    

#%% backward: current th previous
der_arrb=np.array([])
x_arrb=np.array([])

x_derb=6    # initial point

def func_d(x,h):
    """slope func"""
    return (func(x)-func(x-h))/h


while x_derb>=-6:
    y_der=func_d(x_derb,h)
    der_arrb=np.append(der_arrb,y_der)
    x_arrb=np.append(x_arrb,x_derb)
    x_derb=x_derb-h   # go backward

#%% mid: current's the middle (noch nicht)
der_arrm=np.array([])
x_arrm=np.array([])

x_derm=6    # initial point

def func_m(x,h):
    """slope func"""
    return (func(x+h/2)-func(x-h/2))/h


while x_derb>=-6:
    y_der=func_m(x_derm,h)
    der_arrm=np.append(der_arrm,y_der)
    x_arrm=np.append(x_arrm,x_derm)
    x_derm=x_derm-h   # go forward


#%%
fig, fg=plt.subplots(1,1) # create an emty fig
#fg.plot(x,y);
#fg.plot(x,y2);
fg.plot(x_arr,der_arr);
fg.plot(x_arrb,der_arrb);
fg.plot(x_arrm,der_arrm);























