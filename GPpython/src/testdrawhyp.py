#!/usr/bin/env python2
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import ESutils
import scipy as sp
from scipy import linalg as spl
from scipy import stats as sps
from matplotlib import pyplot as plt

import GPdc
#test on a single point
nt=5
X = ESutils.draw_support(1, sp.array([-1.]),sp.array([1.]),nt,ESutils.SUPPORT_UNIFORM)
D = [[sp.NaN]]*(nt)
hyp = sp.array([1.5,0.15])
kf = GPdc.gen_sqexp_k_d(hyp)
Kxx = GPdc.buildKsym_d(kf,X,D)
Y = spl.cholesky(Kxx,lower=True)*sp.matrix(sps.norm.rvs(0,1.,nt)).T+sp.matrix(sps.norm.rvs(0,1e-3,nt)).T
S = sp.matrix([1e-6]*nt).T


G = GPdc.GPcore(X,Y,S,D,GPdc.kernel(GPdc.SQUEXP,2,hyp))

ng = 40
A = sp.empty([ng,ng])

sup = sp.logspace(-3,2,ng)
for i,hi in enumerate(sup):
    for j,hj in enumerate(sup):
        A[i,j] = GPdc.GP_LKonly(X,Y,S,D,GPdc.kernel(GPdc.SQUEXP,2,sp.array([hi,hj]))).plk(sp.array([0.,-1.]),sp.array([1.,1.]))
A = -sp.log10(-A+sp.amax(A)+1.)
plt.figure()
plt.contour(sp.log10(sup),sp.log10(sup),A,30)
X = ESutils.drawhyp_plk(X,Y,S,D,GPdc.SQUEXP,sp.array([0.,-1.]),sp.array([1.,1.]),125)
plt.plot(sp.log10(X[:,1]),sp.log10(X[:,0]),'b.')


#lots of points
nt=42
X = ESutils.draw_support(1, sp.array([-1.]),sp.array([1.]),nt,ESutils.SUPPORT_UNIFORM)
D = [[sp.NaN]]*(nt)
hyp = sp.array([1.5,0.15])
kf = GPdc.gen_sqexp_k_d(hyp)
Kxx = GPdc.buildKsym_d(kf,X,D)
Y = spl.cholesky(Kxx,lower=True)*sp.matrix(sps.norm.rvs(0,1.,nt)).T+sp.matrix(sps.norm.rvs(0,1e-3,nt)).T
S = sp.matrix([1e-6]*nt).T


G = GPdc.GPcore(X,Y,S,D,GPdc.kernel(GPdc.SQUEXP,2,hyp))

ng = 80
A = sp.empty([ng,ng])

sup = sp.logspace(-2,2,ng)
for i,hi in enumerate(sup):
    for j,hj in enumerate(sup):
        A[i,j] = GPdc.GP_LKonly(X,Y,S,D,GPdc.kernel(GPdc.SQUEXP,2,sp.array([hi,hj]))).plk(sp.array([0.,-1.]),sp.array([1.,1.]))

A = -sp.log10(-A+sp.amax(A)+1.)

plt.figure()
plt.contour(sp.log10(sup),sp.log10(sup),A,30)
#plt.imshow(A)

X = ESutils.drawhyp_plk(X,Y,S,D,GPdc.SQUEXP,sp.array([0.,-1.]),sp.array([1.,1.]),125)

plt.plot(sp.log10(X[:,1]),sp.log10(X[:,0]),'b.')
plt.show()