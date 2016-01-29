#!/usr/bin/env python2
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import sys
sys.path.append("./..")
import scipy as sp
from matplotlib import pyplot as plt
import seaborn as sns

import GPdc
D=4
f,a = plt.subplots(5,D)

ke = GPdc.kernel(GPdc.SQUEXP,D,sp.array([0.75,0.1,0.2,0.3,0.4]))
km = GPdc.kernel(GPdc.MAT52,D,sp.array([0.75,0.1,0.2,0.3,0.4]))
#support
ns = 100
xax = sp.linspace(-1,1,ns)
Xo = sp.vstack([0.,0.,0.,0.]*ns)
X0 = sp.vstack([[i,0.,0.,0.] for i in xax])
X1 = sp.vstack([[0.,i,0.,0.] for i in xax])
X2 = sp.vstack([[0.,0.,i,0.] for i in xax])
X3 = sp.vstack([[0.,0.,0.,i] for i in xax])
Xax=[X0,X1,X2,X3]

#noderivative  --------------------------------------------------------------
d0 = [[sp.NaN]]
d1 = [[sp.NaN]]
a[0,0].set_ylabel('[n][n]')
for i in xrange(D):
    y = [ke(Xo[j,:],Xax[i][j,:],d0,d1) for j in xrange(ns)]
    ym = [km(Xo[j,:],Xax[i][j,:],d0,d1) for j in xrange(ns)]
    a[0,i].plot(xax,y,'b')
    a[0,i].plot(xax,ym,'r')

#firstderivative0  --------------------------------------------------------------
d0 = [[sp.NaN]]
d1 = [0]
a[1,0].set_ylabel('[n][0]')
for i in xrange(D):
    y = [ke(Xo[j,:],Xax[i][j,:],d0,d1) for j in xrange(ns)]
    ym = [km(Xo[j,:],Xax[i][j,:],d0,d1) for j in xrange(ns)]
    a[1,i].plot(xax,y,'b')
    a[1,i].plot(xax,ym,'r')
    
#firstderivative1  --------------------------------------------------------------
d0 = [[sp.NaN]]
d1 = [1]
a[2,0].set_ylabel('[n][1]')
for i in xrange(D):
    y = [ke(Xo[j,:],Xax[i][j,:],d0,d1) for j in xrange(ns)]
    ym = [km(Xo[j,:],Xax[i][j,:],d0,d1) for j in xrange(ns)]
    a[2,i].plot(xax,y,'b')
    a[2,i].plot(xax,ym,'r')
    
#firstderivative2  --------------------------------------------------------------
d0 = [[sp.NaN]]
d1 = [2]
a[3,0].set_ylabel('[n][2]')
for i in xrange(D):
    y = [ke(Xo[j,:],Xax[i][j,:],d0,d1) for j in xrange(ns)]
    ym = [km(Xo[j,:],Xax[i][j,:],d0,d1) for j in xrange(ns)]
    a[3,i].plot(xax,y,'b')
    a[3,i].plot(xax,ym,'r')
    
#firstderivative3  --------------------------------------------------------------
d0 = [[sp.NaN]]
d1 = [3]
a[4,0].set_ylabel('[n][3]')
for i in xrange(D):
    y = [ke(Xo[j,:],Xax[i][j,:],d0,d1) for j in xrange(ns)]
    ym = [km(Xo[j,:],Xax[i][j,:],d0,d1) for j in xrange(ns)]
    a[4,i].plot(xax,y,'b')
    a[4,i].plot(xax,ym,'r')
plt.show()
