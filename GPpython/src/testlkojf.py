#!/usr/bin/env python2
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import lkojf
import scipy as sp
import time
from matplotlib import pyplot as plt
import GPdc

L = lkojf.lkojf()
nf = 400
L.makedata_default(nf)
print L.llk(sp.array([1.3,0.13,0.2,1e-2]))#


np = 200
X = sp.array([sp.linspace(0,nf,np)]).T
print X
H = sp.empty([np,1])
T = sp.empty([np,1])
for i in xrange(np):
    [H[i,0],T[i,0]] = L.llks(sp.array([0.3,0.13,0.2,1e-4]),int(X[i,0]))


lb = sp.array([0.,0.,-4.,-0.2*float(nf),-0.2*float(nf)])
ub = sp.array([4.,3.,3.,1.2*float(nf),1.2*float(nf)])
MLEH =  GPdc.searchMLEhyp(X,H,sp.zeros([np,1]),[[sp.NaN]]*(np),lb,ub,GPdc.SQUEXPPS,mx=10000)
G = GPdc.GPcore(X.copy(),H,sp.zeros([np,1]),[[sp.NaN]]*(np),GPdc.kernel(GPdc.SQUEXPPS,1,sp.array(MLEH)))

[m,v] = G.infer_diag(X,[[sp.NaN]]*(np))

S = sp.empty([np,1])
for i in xrange(np):
    S[i,0] = -MLEH[2]*(X[i,0]-MLEH[3])*(X[i,0]-MLEH[4])
f,a = plt.subplots(1)
a.plot(X.flatten(),S.flatten())

f,a = plt.subplots(2)
a[0].plot(X.flatten(),H.flatten(),'g.')
a[1].plot(X.flatten(),T.flatten())

s = sp.sqrt(v).flatten()
a[0].fill_between(X.flatten(),m.flatten()-2*s,m.flatten()+2*s, facecolor='lightblue',edgecolor='lightblue')
a[0].plot(X.flatten(),m.flatten(),'b')


plt.show()