#!/usr/bin/env python2
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import OPTutils
import scipy as sp
from matplotlib import pyplot as plt
import GPdc


d=2
lb = sp.array([[-1.]*d])
ub = sp.array([[1.]*d])

[ojf,truexmin] = OPTutils.gensquexpdraw(d,sp.array([-1.]*d),sp.array([1.]*d))

kindex = GPdc.SQUEXP

mprior = sp.array([0.]+[-1.]*d)
sprior = sp.array([1.]*(d+1))
maxf = 4000
s = 1e-6
ninit = 10

#para = [kindex,hlb,hub,maxf,s,ninit]
para = [kindex,mprior,sprior,maxf,s,ninit]


OE = OPTutils.EIMLE(ojf,lb,ub,para)

for i in xrange(40):
    OE.step()
    
f,a = plt.subplots(4)
a[0].plot(OE.Ymin,'rx-')
a[1].plot(OE.T,'r')
a[2].plot([sum(OE.C[:i]) for i in xrange(len(OE.C))],'r')
a[3].semilogy(OE.compX(truexmin)[1,:].flatten(),'r')

print OE.X
print OE.Y

plt.show()