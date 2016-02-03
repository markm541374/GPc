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
ojf = OPTutils.genbanana(cfn=cfn)
kindex = GPdc.MAT52
prior = sp.array([0.]+[-1.]*d)
sprior = sp.array([1.]*(d+1))
kernel = [kindex,prior,sprior]
nreps = 20
bd = 40
slist = [1e-5,1e-7 ,1e-9]
f,a = plt.subplots(3)

for s in slist:
    yr=[]
    zr=[]
    for i in xrange(nreps):
        fname = "../cache/rosennoise/EIMLE_"+str(int(100*sp.log10(s)))+"_"+str(pwr)+"_"+str(bd)+"_"+str(i)+".p"
        if os.path.exists(fname):
            [X,Y,S,D,R,C,T,Tr,Xmin,Ymin,Rreg, Yreg] = pickle.load(open(fname,'rb'))
        else:
            [X,Y,S,D,R,C,T,Tr,Xmin,Ymin,Rreg, Yreg] = search.MLEFS(ojf,lb,ub,kernel,s,bd)
            pickle.dump([X,Y,S,D,R,C,T,Tr,Xmin,Ymin,Rreg, Yreg],open(fname,'wb'))
        yr.append([Yreg])
        
        
        fname = "../cache/rosennoise/PESFS_"+str(int(100*sp.log10(s)))+"_"+str(pwr)+"_"+str(bd)+"_"+str(i)+".p"
        if os.path.exists(fname):
            [X,Y,S,D,R,C,T,Tr,Xmin,Ymin,Rreg, Yreg] = pickle.load(open(fname,'rb'))
        else:
            [X,Y,S,D,R,C,T,Tr,Xmin,Ymin,Rreg, Yreg] = search.PESFS(ojf,lb,ub,kernel,s,bd)
            pickle.dump([X,Y,S,D,R,C,T,Tr,Xmin,Ymin,Rreg, Yreg],open(fname,'wb'))
        zr.append([Yreg])
        
        
        
    
    Z = sp.vstack(yr)
    m = sp.mean(sp.log10(Z),axis=0)
    v = sp.var(sp.log10(Z),axis=0)
    
    sq = sp.sqrt(v)
    a[2].fill_between(sp.array([sum(C[:j]) for j in xrange(len(C))]),(m-sq).flatten(),(m+sq).flatten(),facecolor='lightblue',edgecolor='lightblue',alpha=0.5)
    a[2].plot([sum(C[:j]) for j in xrange(len(C))],m.flatten(),'x-')
    
    Z = sp.vstack(zr)
    m = sp.mean(sp.log10(Z),axis=0)
    v = sp.var(sp.log10(Z),axis=0)
    sq = sp.sqrt(v)
    a[2].fill_between(sp.array([sum(C[:j]) for j in xrange(len(C))]),(m-sq).flatten(),(m+sq).flatten(),facecolor='lightgreen',edgecolor='lightgreen',alpha=0.5)
    a[2].plot([sum(C[:j]) for j in xrange(len(C))],m.flatten(),'.-')
    
    
    f.savefig("tmp.png")
    
s=1e-1
ba = 8
Cz=[]
Rz=[]
for i in xrange(6):
    fname = "../cache/rosennoise/PESVS_"+str(int(100*sp.log10(s)))+"_"+str(pwr)+"_"+str(ba)+"_"+str(i)+".p"
    if os.path.exists(fname):
        [X,Y,S,D,R,C2,T,Tr,Xmin,Ymin,Rreg, Yreg] = pickle.load(open(fname,'rb'))
    else:
        [X,Y,S,D,R,C2,T,Tr,Xmin,Ymin,Rreg, Yreg] = search.PESVS(ojf,lb,ub,kernel,s,ba,lambda x,s:cfn(s),-9,-1)
        pickle.dump([X,Y,S,D,R,C2,T,Tr,Xmin,Ymin,Rreg, Yreg],open(fname,'wb'))
    Cz.append([sum(C2[:j]) for j in xrange(len(C2))])
    Rz.append(sp.log(Yreg).flatten())
    a[1].plot(C2)
    #a[2].plot(sp.array([sum(C2[:j]) for j in xrange(len(C2))]).flatten(),(sp.log(Yreg)).flatten(),'ro-')

[sup,m,sd]=OPTutils.bounds(Cz,Rz)
a[2].fill_between(sup.flatten(),(m-sd).flatten(),(m+sd).flatten(),facecolor='salmon',edgecolor='salmon',alpha=0.5)
a[2].plot(sup,m.flatten(),'darkred')
sx = sp.logspace(0,-8,100)
cx = map(cfn,sx)
a[0].loglog(sx,cx)
f.savefig("tmp.png")
plt.show()
