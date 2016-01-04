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
#2d
nt=48
d=3
lb = sp.array([-1.]*d)
ub = sp.array([1.]*d)
[X,Y,S,D] = ESutils.gen_dataset(nt,d,lb,ub,GPdc.SQUEXP,sp.array([1.5,0.35,0.30,0.35]))

G = PES.makeG(X,Y,S,D,GPdc.SQUEXP,sp.array([0.,-1.,-1.,-1.]),sp.array([1.,1.,1.,1.]),10)
Z=PES.drawmins_inplane(G,8,sp.array([-1.]*d),sp.array([1.]*d),axis=1,value = 0.,SUPPORT=500,SLICELCB_PARA=1.)
print Z
Ga = GPdc.GPcore(*PES.addmins_inplane(G,X,Y,S,D,Z[0,:],axis=1,value=0.)+[G.kf])

np=180
sup = sp.linspace(-1,1,np)
Dp = [[sp.NaN]]*np
Xp0 = sp.vstack([sp.array([i,Z[0,1],Z[0,2]]) for i in sup])
Xp1 = sp.vstack([sp.array([Z[0,0],i,Z[0,2]]) for i in sup])
Xp2 = sp.vstack([sp.array([Z[0,0],Z[0,1],i]) for i in sup])

[mp0,Vp0] = Ga.infer_diag_post(Xp0,Dp)
[mp1,Vp1] = Ga.infer_diag_post(Xp1,Dp)
[mp2,Vp2] = Ga.infer_diag_post(Xp2,Dp)
f,a = plt.subplots(3)
s = sp.sqrt(Vp0[0,:])
a[0].fill_between(sup,sp.array(mp0[0,:]-2.*s).flatten(),sp.array(mp0[0,:]+2.*s).flatten(),facecolor='lightblue',edgecolor='lightblue')
a[0].plot(sup,mp0[0,:].flatten())
a[0].plot(Z[0,0].flatten(),[0],'r.')

s = sp.sqrt(Vp1[0,:])
a[1].fill_between(sup,sp.array(mp1[0,:]-2.*s).flatten(),sp.array(mp1[0,:]+2.*s).flatten(),facecolor='lightblue',edgecolor='lightblue')
a[1].plot(sup,mp1[0,:].flatten())
a[1].plot(Z[0,1].flatten(),[0],'r.')

s = sp.sqrt(Vp2[0,:])
a[2].fill_between(sup,sp.array(mp2[0,:]-2.*s).flatten(),sp.array(mp2[0,:]+2.*s).flatten(),facecolor='lightblue',edgecolor='lightblue')
a[2].plot(sup,mp2[0,:].flatten())
a[2].plot(Z[0,2].flatten(),[0],'r.')
##before
[mp0,Vp0] = G.infer_diag_post(Xp0,Dp)
[mp1,Vp1] = G.infer_diag_post(Xp1,Dp)
[mp2,Vp2] = G.infer_diag_post(Xp2,Dp)
f,a = plt.subplots(3)
s = sp.sqrt(Vp0[0,:])
a[0].fill_between(sup,sp.array(mp0[0,:]-2.*s).flatten(),sp.array(mp0[0,:]+2.*s).flatten(),facecolor='lightblue',edgecolor='lightblue')
a[0].plot(sup,mp0[0,:].flatten())
#a[0].plot(Z[0,0].flatten(),[0],'r.')

s = sp.sqrt(Vp1[0,:])
a[1].fill_between(sup,sp.array(mp1[0,:]-2.*s).flatten(),sp.array(mp1[0,:]+2.*s).flatten(),facecolor='lightblue',edgecolor='lightblue')
a[1].plot(sup,mp1[0,:].flatten())
#a[1].plot(Z[0,1].flatten(),[0],'r.')

s = sp.sqrt(Vp2[0,:])
a[2].fill_between(sup,sp.array(mp2[0,:]-2.*s).flatten(),sp.array(mp2[0,:]+2.*s).flatten(),facecolor='lightblue',edgecolor='lightblue')
a[2].plot(sup,mp2[0,:].flatten())
#a[2].plot(Z[0,2].flatten(),[0],'r.')

plt.show()