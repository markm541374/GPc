#!/usr/bin/env python2
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import OPTutils
import scipy as sp
from matplotlib import pyplot as plt
import GPdc

d=1

X = sp.empty([0,d])
Y = sp.empty([0,1])
S = sp.empty([0,1])
D = []

ns = 5
ub = sp.array([1.]*d)
lb = sp.array([-1.]*d)
for i in xrange(ns):
    x = sp.random.uniform(size=d)*(ub-lb)+lb
    s = 1e-3
    dv = [sp.NaN]
    y = OPTutils.quad(x,s,dv)[0]
    
    X = sp.vstack([X,x])
    Y = sp.vstack([Y,y])
    S = sp.vstack([S,s])
    D.append(dv)

G = GPdc.GPcore(X,Y,S,D,GPdc.kernel(0,d,sp.array([1.1,0.3,0.3,0.3])))
np=200
sup = sp.linspace(-1,1,np)
Xp = sp.vstack([sp.array([i,0.,0.]) for i in sup])
Dp = [[sp.NaN]]*np
[m,v] = G.infer_diag_post(Xp,Dp)
m2 = G.infer_m(Xp,Dp)
            
f,a = plt.subplots(4)
a[0].plot(sup,m2.flatten(),'g')
a[0].plot(sup,m.flatten())
a[0].plot(X.flatten(),Y.flatten(),'r.')
a[1].plot(sup,v.flatten())
e = G.infer_EI(Xp,Dp)
a[2].plot(sup,e.flatten())
l = G.infer_LCB(Xp,Dp,1.)
a[3].plot(sup,l.flatten(),'r')
#---------------------------------------------------------------------------------
X=sp.vstack([ i for i in sp.linspace(-1,1,ns)])#sp.random.uniform(size=[ns,d])*(ub-lb)+lb
X = sp.matrix(sp.linspace(-1,1,ns)).T
Y=sp.empty([ns,1])
for i in xrange(ns):
    Y[i,0] = 1.#OPTutils.quad(X[i,:],s,dv)[0]
S = sp.ones([ns,1])*s
D = [[sp.NaN]]*ns

G = GPdc.GPcore(X,Y,S,D,GPdc.kernel(GPdc.SQUEXP,d,sp.array([1.1,0.3,0.3,0.3])))
G.printc()
np=200
sup = sp.linspace(-1,1,np)
Xp = sp.vstack([sp.array([i]) for i in sup])
Dp = [[sp.NaN]]*np
[m,v] = G.infer_diag_post(Xp,Dp)
m2 = G.infer_m(Xp,Dp)
            
f,a = plt.subplots(4)
a[0].plot(sup,m2.flatten(),'g')
#a[0].plot(sup,m.flatten())
a[0].plot(sp.array(X).flatten(),Y.flatten(),'r.')
a[1].plot(sup,v.flatten())
e = G.infer_EI(Xp,Dp)
a[2].plot(sup,e.flatten())
l = G.infer_LCB(Xp,Dp,1.)
a[3].plot(sup,l.flatten(),'r')

plt.show()