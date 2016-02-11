#!/usr/bin/env python2
#encoding: UTF-8

# Having a look at how noise affects convergence when using rosenbock


import scipy as sp
from matplotlib import pyplot as plt
import GPdc
import OPTutils
import search
import os
import pickle


d=2
lb = sp.array([[-1.]*d])
ub = sp.array([[1.]*d])
pwr = 0.2
cfn = lambda s:((1e-6)/s)**pwr
ojf = OPTutils.genbranin(cfn=cfn)
braninmin = 0.39788735772973816
kindex = GPdc.MAT52
prior = sp.array([0.]+[-1.]*d)
sprior = sp.array([1.]*(d+1))
kernel = [kindex,prior,sprior]
nreps = 4
bd = 35
s = 1e-7,1e-9
f,a = plt.subplots(3)

for s in slist:
    
    names = ["../cache/braninrecc/EIMLE_"+str(int(100*sp.log10(s)))+"_"+str(pwr)+"_"+str(i)+".p" for i in xrange(nreps)]
    results = search.multiMLEFS(ojf,lb,ub,kernel,s,bd,names)
    yr = [r[10].flatten() for r in results]
    C = results[0][5]

        
    Z = sp.vstack(yr)-braninmin
    m = sp.mean(sp.log10(Z),axis=0)
    v = sp.var(sp.log10(Z),axis=0)
    
    sq = sp.sqrt(v)
    print len(C)
    print m.size
    a[2].fill_between(sp.array([sum(C[:j]) for j in xrange(len(C))]),(m-sq).flatten(),(m+sq).flatten(),facecolor='lightblue',edgecolor='lightblue',alpha=0.5)
    a[2].plot([sum(C[:j]) for j in xrange(len(C))],m.flatten(),'x-')
    


sx = sp.logspace(0,-8,100)
cx = map(cfn,sx)
a[0].loglog(sx,cx)
f.savefig("tmp.png")
plt.show()
