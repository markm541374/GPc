#!/usr/bin/env python2
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import eprop
import scipy as sp
from matplotlib import pyplot as plt

#basic ep test on three values
m = sp.array([1.,2.,0.5])
v = sp.array([[1.,0.5,0.2],[0.5,2.,0.5],[0.2,0.5,1.]])

f = plt.figure()
a0 = plt.subplot(111)
for i in xrange(len(m)):
    plt.plot(m[i],i+1,'bo')
    plt.plot([m[i]-sp.sqrt(v[i,i]),m[i]+sp.sqrt(v[i,i])],[i+1,i+1],'b')
plt.axis([-5,5,0,len(m)+1])

mu,vu = eprop.expectation_prop(m,v,sp.array([-1.,-2.,-1.]),-sp.ones(3),sp.zeros(3),5)

for i in xrange(len(m)):
    plt.plot(mu[i],i+1.1,'ro')
    plt.plot([mu[i]-sp.sqrt(vu[i,i]),mu[i]+sp.sqrt(vu[i,i])],[i+1.1,i+1.1],'r')
    
print "----------------"
print m
print mu
print v
print vu
plt.show()