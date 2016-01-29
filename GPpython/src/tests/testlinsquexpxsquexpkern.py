# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from scipy import stats as sps
from scipy import linalg as spl
import scipy as sp
from matplotlib import pyplot as plt

import GPdc

hyp = sp.array([1.5,0.25])
kf = GPdc.gen_sqexp_k_d(hyp)
nt=18
x = sp.linspace(-1,1,nt)
Dtmp = [[sp.NaN]]*nt
Kxx = GPdc.buildKsym_d(kf,sp.matrix(x).T,Dtmp)

Z = spl.cholesky(Kxx,lower=True)*sp.matrix(sps.norm.rvs(0,1.,nt)).T
Z = sp.vstack([Z,spl.cholesky(Kxx,lower=True)*sp.matrix(sps.norm.rvs(0,1.,nt)).T])
Z = sp.vstack([Z,spl.cholesky(Kxx,lower=True)*sp.matrix(sps.norm.rvs(0,1.,nt)).T])


y = [-0.4*(i-0.2)+sps.norm.rvs(scale=0.005) for i in x]
y+=[-0.6*(i+0.2)+sps.norm.rvs(scale=0.005) for i in x]
y+=[0.1*(i+0.8)+sps.norm.rvs(scale=0.005) for i in x]
X = sp.vstack([sp.array([i,0.]) for i in x])
X = sp.vstack([X,[sp.array([i,0.4]) for i in x]])
X = sp.vstack([X,[sp.array([i,-0.3]) for i in x]])

Y = sp.matrix(y).T+Z
#X = sp.matrix([[0.,-9.],[0.2,5.],[0.4,12.],[0.6,3.],[0.8,9.]])

#Y = sp.matrix([0.2,0.1,0.,-0.15,-0.3]).T

D = [[sp.NaN]]*X.shape[0]
S = sp.matrix([0.005**2]*X.shape[0]).T
f0 = plt.figure()
a0 = plt.subplot(111)
a0.plot(sp.array(X[:,0]).flatten(),Y,'g.')


lb = sp.array([-2.,-1.,-2.,-2.,-2.,-2.,-2.])
ub = sp.array([2.,1.,2.,2.,2.,2.,2.])
MLEH =  GPdc.searchMLEhyp(X,Y,S,D,lb,ub,GPdc.LINSQUEXPXSQUEXP,mx=10000)
print "xxx"
GPdc.kernel(GPdc.LINSQUEXPXSQUEXP,2,MLEH)
print "yyyy"
print MLEH
G = GPdc.GPcore(X,Y,S,D,GPdc.kernel(GPdc.LINSQUEXPXSQUEXP,2,MLEH))
print G.llk()

print "zzz"
np=180
sup = sp.linspace(-1,1,np)
Dp = [[sp.NaN]]*np
Xp = sp.vstack([sp.array([i,0.2]) for i in sup])

[m,v] = G.infer_diag(Xp,Dp)
print "wwww"
a0.plot(sup,m)
sq = sp.sqrt(v)

a0.fill_between(sup, sp.array(m-1.*sq).flatten(), sp.array(m+1.*sq).flatten(), facecolor='lightblue',edgecolor='lightblue')

plt.show()