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
def cfn(s):
    print s
    print 'cfn'+str(((1e-6)/s)**pwr)
    return ((1e-6)/s)**pwr
ojf = OPTutils.genbranin(cfn=cfn)
braninmin = 0.39788735772973816
kindex = GPdc.MAT52
prior = sp.array([0.]+[-1.]*d)
sprior = sp.array([1.]*(d+1))
kernel = [kindex,prior,sprior]
nreps = 1
bd = 30
slist = [1e-7]
f,a = plt.subplots(3)

for s in slist:
    
    names = ["../cache/tmp/EIMLE_"+str(int(100*sp.log10(s)))+"_"+str(pwr)+"_"+str(i)+".p" for i in xrange(nreps)]
    results = search.multiMLEFS(ojf,lb,ub,kernel,s,bd,names)
    yr = [r[11].flatten() for r in results]
    C = results[0][5]
    
    names = ["../cache/tmp/PESFS_"+str(int(100*sp.log10(s)))+"_"+str(pwr)+"_"+str(i)+".p" for i in xrange(nreps)]
    results = search.multiPESFS(ojf,lb,ub,kernel,s,bd,names)
    zr = [r[11].flatten() for r in results]
    C = results[0][5]
    
        
    Z = sp.vstack(yr)-braninmin
    m = sp.mean(sp.log10(Z),axis=0)
    v = sp.var(sp.log10(Z),axis=0)
    
    sq = sp.sqrt(v)
    print len(C)
    print m.size
    a[2].fill_between(sp.array([sum(C[:j]) for j in xrange(len(C))]),(m-sq).flatten(),(m+sq).flatten(),facecolor='lightblue',edgecolor='lightblue',alpha=0.5)
    a[2].plot([sum(C[:j]) for j in xrange(len(C))],m.flatten(),'x-')
    
    
    Z = sp.vstack(zr)-braninmin
    m = sp.mean(sp.log10(Z),axis=0)
    v = sp.var(sp.log10(Z),axis=0)
    sq = sp.sqrt(v)
    a[2].fill_between(sp.array([sum(C[:j]) for j in xrange(len(C))]),(m-sq).flatten(),(m+sq).flatten(),facecolor='lightgreen',edgecolor='lightgreen',alpha=0.5)
    a[2].plot([sum(C[:j]) for j in xrange(len(C))],m.flatten(),'.-')
    
    
    
    
    f.savefig("tmp.png")
    


plt.show()
