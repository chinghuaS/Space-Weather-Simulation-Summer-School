# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:06:16 2022

@author: hua08

Solution of a 1D Poisson equation: -u_xx = f
Domain: [0,1]
BC: u(0) = u(1) = 0
with f = (3*x + x^2)*exp(x)

Analytical solution: -x*(x-1)*exp(x)

Finite differences (FD) discretization: second-order diffusion operator
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    #We have to load this
from math import pi
plt.close()

"Number of points"
N = 8
Dx = 1/N
x = np.linspace(0,1+Dx,N+2)

"System matrix and RHS term"
A = (1/Dx**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1))
F = 2*(2*x**2+5*x-2)*np.exp(x)
#%% 1st & 2nd order approximation
A[0,:]=np.concatenate([[1],np.zeros(N+1)])
F[0]=0

A[-1,:]=(.5/Dx)*np.concatenate([np.zeros(N-1),[-1,0,1]])
F[-1]=0

U = np.linalg.solve(A,F)
u=U
ua = 2*x*(3-2*x)*np.exp(x)


#%% plot
"Plotting solution"
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
#plt.axis([0, 1,0, 0.5])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

#%% reeor ratio
# err8=error
# err16/err8 ~=.25

