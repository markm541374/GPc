# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "mark"
__date__ = "$02-Dec-2015 21:11:58$"

from scipy import stats as sps
from scipy import linalg as spl
import scipy as sp
from matplotlib import pyplot as plt
import GPd
import GPdc
import time

sp.random.seed(21345)
nt = 16
Dm = 2
X = sp.random.uniform(-1,1,size=[nt,Dm])
D = [[sp.NaN]]*(nt)

hyp = sp.array([1.]+[0.15]*Dm)
kf = GPdc.gen_sqexp_k_d(hyp)

Kxx = GPd.buildKsym_d(kf,X,D)

Y = spl.cholesky(Kxx,lower=True)*sp.matrix(sps.norm.rvs(0,1.,nt)).T+sp.matrix(sps.norm.rvs(0,1e-3,nt)).T
S = sp.matrix([1e-3]*nt).T

f0 = plt.figure()
a0 = plt.subplot(111)
a0.plot(sp.array(X[:,0]).flatten(),Y,'g.')

G = GPdc.GP_EI_random(X,Y,S,D,GPdc.gen_sqexp_k_d(hyp))

print X


lb = sp.array([-1.]*Dm)
ub = sp.array([1.]*Dm)

print "xxxx"
randmin=[]
randrange = [int(i) for i in sp.logspace(0,5,150)]
randtime = []
for i in randrange:
    t0 = time.clock()
    [ymin,xmin] = G.getnext(lb,ub,i)
    t1 = time.clock()
    #print xmin
    #print ymin
    randtime.append(t1-t0)
    randmin.append(ymin)

[ymin,xmin] = G.getnext(lb,ub,5000000)
randbest = [xmin,ymin[0]]

G = GPdc.GP_EI_direct(X,Y,S,D,GPdc.gen_sqexp_k_d(hyp))

lb = sp.array([-1.]*Dm)
ub = sp.array([1.]*Dm)
xmin = sp.array([42.]*Dm)
ymin = sp.array([42.])


print "xxxx"
dirmin=[]
dirrange = [int(i) for i in sp.logspace(0,4,150)]
dirtime = []
for i in dirrange:
    t0=time.clock()
    [ymin,xmin] = G.getnext(lb,ub,i)
    t1=time.clock()
    dirtime.append(t1-t0)
    dirmin.append(ymin)
    
[ymin,xmin] = G.getnext(lb,ub,50000)
direbest = [xmin,ymin[0]]
plt.show()

print randbest
print direbest
f0 = plt.figure()
plt.semilogx(randrange,[-i for i in randmin],'r')
plt.semilogx(dirrange,[-i for i in dirmin],'b')

f1 = plt.figure()
plt.semilogx(randtime,[-i for i in randmin],'r')
plt.semilogx(dirtime,[-i for i in dirmin],'b')

f2 = plt.figure()
plt.plot(randrange,randtime,'r')
plt.plot(dirrange,dirtime,'b')

f0.savefig("../figs/n_y.png")
f1.savefig("../figs/t_y.png")
f2.savefig("../figs/n_t.png")
plt.show()
