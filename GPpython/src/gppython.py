# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "mark"
__date__ = "$22-Nov-2015 21:25:42$"

import os
import ctypes as ct
import scipy as sp
from scipy import stats as sps
from scipy import linalg as spl
from matplotlib import pyplot as plt
import time
import GPdc
import GPd
nt = 12



X = sp.matrix(sp.linspace(-1,1,nt)).T
D = [[sp.NaN]]*(nt)

hyp = sp.array([1.5,0.15])
kf = GPdc.gen_sqexp_k_d(hyp)
print kf(sp.array([0.]),sp.array([0.]),[sp.NaN],[sp.NaN])

Kxx = GPd.buildKsym_d(kf,X,D)

Y = spl.cholesky(Kxx,lower=True)*sp.matrix(sps.norm.rvs(0,1.,nt)).T+sp.matrix(sps.norm.rvs(0,1e-3,nt)).T
S = sp.matrix([1e-6]*nt).T
f0 = plt.figure()
a0 = plt.subplot(111)
a0.plot(sp.array(X).flatten(),Y,'g.')


"""
f1 = plt.figure()
a1 = plt.subplot(111)
hsup = sp.logspace(-2,2,100)
G = [GPdc.GPcore(X,Y,S,D,GPdc.gen_sqexp_k_d([1.5,i])).llk() for i in hsup]
a1.semilogx(hsup,G)
"""

lb = sp.matrix([-2.]*2).T
ub = sp.matrix([2.]*2).T


#GPdc.libGP.HypSearchMLE(ct.c_int(1),ct.c_int(len(Dx)),X.ctypes.data_as(ct.POINTER(ct.c_double)),Y.ctypes.data_as(ct.POINTER(ct.c_double)),S.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*len(Dx))(*Dx),lb.ctypes.data_as(ct.POINTER(ct.c_double)),ub.ctypes.data_as(ct.POINTER(ct.c_double)),hy.ctypes.data_as(ct.POINTER(ct.c_double)),lk.ctypes.data_as(ct.POINTER(ct.c_double)))
MLEH =  GPdc.searchMLEhyp(X,Y,S,D,lb,ub,0,mx=2000)
print MLEH
G = GPdc.GPcore(X,Y,S,D,GPdc.gen_sqexp_k_d(MLEH))
MLEH = sp.array([1.,0.1])
np=180
sup = sp.linspace(-1,1,np)
Dp = [[sp.NaN]]*np
Xp = sp.matrix(sup).T

[m,v] = G.infer_diag(Xp,Dp)
a0.plot(sup,m)
sq = sp.sqrt(v)

a0.fill_between(sup, sp.array(m-2.*sq).flatten(), sp.array(m+2.*sq).flatten(), facecolor='lightblue',edgecolor='lightblue')




m = sp.array([0.,1.,0.,0.1])
s = sp.array([1.,3.,0.3,0.1])
r = sp.zeros(4)

GPdc.libGP.EI(m.ctypes.data_as(ct.POINTER(ct.c_double)),s.ctypes.data_as(ct.POINTER(ct.c_double)),ct.c_double(0.9),ct.c_int(4),r.ctypes.data_as(ct.POINTER(ct.c_double)))
print r
from tools import EI 
for i in xrange(4):
    print EI(0.9,m[i],s[i])[0,0],
    
G = GPdc.GP_EI_direct(X,Y,S,D,GPdc.gen_sqexp_k_d(MLEH))
lb = sp.array([-1.])
ub = sp.array([1.])
xmin = sp.array([42.])
ymin = sp.array([42.])
print "xxxx"
GPdc.libGP.getnext(ct.c_int(G.s),lb.ctypes.data_as(ct.POINTER(ct.c_double)),ub.ctypes.data_as(ct.POINTER(ct.c_double)),xmin.ctypes.data_as(ct.POINTER(ct.c_double)),ymin.ctypes.data_as(ct.POINTER(ct.c_double)))

print xmin
print ymin

G = GPdc.GP_EI_random(X,Y,S,D,GPdc.gen_sqexp_k_d(MLEH))
lb = sp.array([-1.])
ub = sp.array([1.])
xmin = sp.array([42.])
ymin = sp.array([42.])
print "xxxx"
GPdc.libGP.getnext(ct.c_int(G.s),lb.ctypes.data_as(ct.POINTER(ct.c_double)),ub.ctypes.data_as(ct.POINTER(ct.c_double)),xmin.ctypes.data_as(ct.POINTER(ct.c_double)),ymin.ctypes.data_as(ct.POINTER(ct.c_double)))

print xmin
print ymin
plt.show()