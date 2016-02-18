#!/usr/bin/env python2
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import scipy as sp
import numpy.random as npr
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import time
import GPdc
import search

days = 28
t0=time.clock()
df = pd.read_csv('../data/DemandData_Historic-2015.csv')
t1=time.clock()
print 'read time {0:e}'.format(t1-t0)
N = df.shape[0]
n = min(N,days*48)

dlb = 0.
dub = float(days)
print '{0:d} datapoints'.format(n)
X = sp.array([df.index.values[:n]]).T/48.
Y = sp.array([df.indo.values[:n]]).T/1000.
offs  = sp.mean(Y)
Y-=offs
f,a = plt.subplots(4)

a[0].plot(X,Y,'g.')

S = sp.ones([n,1])*1e-1
D = [[sp.NaN]]*n

def ojf(x,s,d,override=False):
    #print "called ojf: "+str(x)
    hyp = [10**i for i in x.flatten()]
    
    hyp.insert(2,1.)
    print hyp
    t0=time.clock()
    llk = GPdc.GP_LKonly(X,Y,S,D,GPdc.kernel(GPdc.MATPP,1,sp.array(hyp))).llk()
    t1=time.clock()
    if llk<-1.:
        out = sp.log(-llk)+1.
    else:
        out = -llk
    print "--->llk: {0} {1}    t: {2}".format(llk,out,t1-t0)
    
    return [out,1.]

def ojfa(x,s,d,override=False):
    #print "called ojf: "+str(x)
    hyp = [10**i for i in x.flatten()[1:]]
    
    hyp.insert(2,1.)
    print hyp
    t0=time.clock()
    sub = x.flatten()[0]
    npts = int((1.-0.1*sub)*n)
    print "subsampling {0} of {1} at x[0]={2}".format(npts,n,x.flatten()[0])
    ps = npr.choice(range(n),size=npts, replace=False)
    Xd = sp.vstack([X[i] for i in ps])
    Yd = sp.vstack([Y[i] for i in ps])
    Sd = sp.vstack([S[i] for i in ps])
    Dd = [[sp.NaN]]*npts
    
    llk = GPdc.GP_LKonly(Xd,Yd,Sd,Dd,GPdc.kernel(GPdc.MATPP,1,sp.array(hyp))).llk()
    t1=time.clock()
    if llk<-1.:
        out = sp.log(-llk)+1.
    else:
        out = -llk
    print "--->llk: {0} {1}    t: {2}".format(llk,out,t1-t0)
    
    return [out,t1-t0]

#ojf(sp.array([ 0.,-0.5,1., 0., -0.5]),1.,[[sp.NaN]])
"""
lb = sp.array([[-2.,-2.,-2.,-2.]])
ub = sp.array([[2.,2.,2.,2.]])
s=1e-3
budget = 60
fname = '../cache/pesfith.p'
ki = [GPdc.MAT52,sp.array([5.,-1.,-1.,-1.,-1.]),sp.array([3.,1.,1.,1.,1.])]
state = search.PESFS(ojf,lb,ub,ki,s,budget,fname)"""

d=4
kindex = GPdc.MAT52CS
prior = sp.array([0.]+[-2.]+[-1.]*d+[-2.])
sprior = sp.array([1.]+[1.]+[1.]*d+[2.])
kernel = [kindex,prior,sprior]

lb = sp.array([[0.,-2.,-1.,-2.]])
ub = sp.array([[2.,0.,2.,0.]])
budget = 20.
fname = '../cache/ippesfith.p'
state = search.PESIPS(ojfa,lb,ub,kernel,budget,fname)


a[2].plot(state[1],'b')
a[2].plot(state[10],'r')
a[2].plot(state[11],'g')

a[3].plot(state[5])
print state[0]
print state[1]
print state[4]
hr = state[4][-1,:].flatten()[1:]
hyp = [10**i for i in hr.flatten()]
hyp.insert(2,1.)
print "found: {0}".format(hyp)
t0=time.clock()
g = GPdc.GPcore(X,Y,S,D,GPdc.kernel(GPdc.MATPP,1,sp.array(hyp)))
t1=time.clock()
print 'training time {0:e}'.format(t1-t0)
print sp.log(-g.llk())+1.
print sp.log(-GPdc.GPcore(X,Y,S,D,GPdc.kernel(GPdc.MATPP,1,sp.array([10.,0.1,1.01,5.,0.2]))).llk())+1
ns = 4000
sup = sp.linspace(dlb,dub+8.,ns)

mp = g.infer_m_partial(sup,[[sp.NaN]]*ns,GPdc.MAT52PER,sp.array(hyp[:3]))
a[1].plot(sup,mp.flatten(),'g')
t=mp.copy()

mp = g.infer_m_partial(sup,[[sp.NaN]]*ns,GPdc.MAT52,sp.array(hyp[3:5]))
a[1].plot(sup,mp.flatten(),'c')
t+=mp

a[0].plot(sup,t.flatten(),'r')
[m,v] = g.infer_diag(sup,[[sp.NaN]]*ns)
sq = sp.sqrt(v)



a[0].fill_between(sup,(m-2.*sq).flatten(),(m+2.*sq).flatten(),edgecolor='lightblue',facecolor='lightblue')
a[0].plot(sup,m.flatten(),'b')




plt.show()