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

df = pd.read_csv('../data/DemandData_Historic-2015.csv')
N = df.shape[0]
n = min(N,days*48)

dlb = 0.
dub = float(days)
print '{0:d} datapoints'.format(n)
X = sp.array([df.index.values[:n]]).T/48.
Y = sp.array([df.indo.values[:n]]).T/1000.
f,a = plt.subplots(1)

a.plot(X,Y,'g.')

S = sp.ones([n,1])*1e-6
D = [[sp.NaN]]*n
t0=time.clock()
g = GPdc.GPcore(X,Y,S,D,GPdc.kernel(GPdc.MAT52,1,sp.array([10.,0.5])))
t1=time.clock()
print 'training time {0:e}'.format(t1-t0)

ns = 3000
sup = sp.linspace(dlb,dub,ns)
t0=time.clock()
[m,v] = g.infer_diag(sup,[[sp.NaN]]*ns)
t1=time.clock()
print 'inference time {0:e}'.format(t1-t0)
sq = sp.sqrt(v)

t0 =time.clock()
a.fill_between(sup,(m-2.*sq).flatten(),(m+2.*sq).flatten(),edgecolor='lightblue',facecolor='lightblue')
a.plot(sup,m.flatten(),'b')
t1=time.clock()
print 'plotting time {0:e}'.format(t1-t0)

plt.show()