# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from scipy import stats as sps
from scipy import linalg as spl
import scipy as sp
from matplotlib import pyplot as plt

import GPdc

nt=12
X = sp.matrix(sp.linspace(-1,1,nt)).T
D = [[sp.NaN]]*(nt)

hyp = sp.array([1.5,0.15])
kf = GPdc.gen_sqexp_k_d(hyp)

Kxx = GPdc.buildKsym_d(kf,X,D)

Y = spl.cholesky(Kxx,lower=True)*sp.matrix(sps.norm.rvs(0,1.,nt)).T+sp.matrix(sps.norm.rvs(0,1e-3,nt)).T
S = sp.matrix([1e-6]*nt).T
f0 = plt.figure()
a0 = plt.subplot(111)
a1=a0.twinx()
a0.plot(sp.array(X[:,0]).flatten(),Y,'g.')


lb = sp.array([-2.,-2.])
ub = sp.array([2.,2.])
MLEH =  GPdc.searchMLEhyp(X,Y,S,D,lb,ub,GPdc.SQUEXP,mx=10000)

print MLEH
G = GPdc.GPcore(X,Y,S,D,GPdc.kernel(GPdc.SQUEXP,1,MLEH))
print G.llk()

np=180
sup = sp.linspace(-1,1,np)
Dp = [[sp.NaN]]*np
Xp = sp.vstack([sp.array([i]) for i in sup])

[m,v] = G.infer_diag(Xp,Dp)
a0.plot(sup,m.flatten())
sq = sp.sqrt(v)

a0.fill_between(sup, sp.array(m-2.*sq).flatten(), sp.array(m+2.*sq).flatten(), facecolor='lightblue',edgecolor='lightblue')

lcb = G.infer_LCB(Xp,Dp,2.)
a0.plot(sup,sp.array(lcb).flatten(),'g')

ei = G.infer_EI(Xp,Dp)
a1.plot(sup,sp.array(ei).flatten(),'r')

plt.show()
