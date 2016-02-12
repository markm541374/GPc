#!/usr/bin/env python2
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import time
import GPdc
days = 100
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
f,a = plt.subplots(2)

a[0].plot(X,Y,'g.')

S = sp.ones([n,1])*1e-6
D = [[sp.NaN]]*n
t0=time.clock()
g = GPdc.GPcore(X,Y,S,D,GPdc.kernel(GPdc.DEV,1,sp.array([10.,0.2,1.,10.,2.,7.,2.,0.1])))
t1=time.clock()
print 'training time {0:e}'.format(t1-t0)
t0=time.clock()
l = GPdc.GP_LKonly(X,Y,S,D,GPdc.kernel(GPdc.DEV,1,sp.array([10.,0.2,1.,10.,2.,7.,2.,0.1]))).llk()
t1=time.clock()
print 'llk time {0:e}'.format(t1-t0)

ns = 300
sup = sp.linspace(dlb,dub+8.,ns)

mp = g.infer_m_partial(sup,[[sp.NaN]]*ns,GPdc.MAT52PER,sp.array([10.,0.2,1.]))
a[1].plot(sup,mp.flatten(),'g')
t=mp.copy()

mp = g.infer_m_partial(sup,[[sp.NaN]]*ns,GPdc.MAT52,sp.array([2.,0.1]))
a[1].plot(sup,mp.flatten(),'c')
t+=mp

mp = g.infer_m_partial(sup,[[sp.NaN]]*ns,GPdc.MAT52PER,sp.array([10.,2.,7.]))
a[1].plot(sup,mp.flatten(),'r')
t+=mp
a[0].plot(sup,t.flatten(),'r')
[m,v] = g.infer_diag(sup,[[sp.NaN]]*ns)
sq = sp.sqrt(v)



a[0].fill_between(sup,(m-2.*sq).flatten(),(m+2.*sq).flatten(),edgecolor='lightblue',facecolor='lightblue')
a[0].plot(sup,m.flatten(),'b')




plt.show()