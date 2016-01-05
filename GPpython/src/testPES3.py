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
import PES

#-------------------------------------------------------------------------
#3d
nt=88
d=3
lb = sp.array([-1.]*d)
ub = sp.array([1.]*d)
[X,Y,S,D] = ESutils.gen_dataset(nt,d,lb,ub,GPdc.SQUEXP,sp.array([1.5,0.35,0.30,0.35]))
NMINS = 4
G = PES.makeG(X,Y,S,D,GPdc.SQUEXP,sp.array([0.,-1.,-1.,-1.]),sp.array([1.,1.,1.,1.]),10)
Z=PES.drawmins_inplane(G,NMINS,sp.array([-1.]*d),sp.array([1.]*d),axis=1,value = 0.,SUPPORT=500,SLICELCB_PARA=1.)
print Z
Ga = [GPdc.GPcore(*PES.addmins_inplane(G,X,Y,S,D,Z[i,:],axis=1,value=0.)+[G.kf]) for i in xrange(NMINS)]

PES.PESgain(G,Ga,Z,sp.array([0.,0.,0.]),[[sp.NaN]],1e-3)
plt.show()