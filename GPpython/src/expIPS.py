#!/usr/bin/env python2
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import OPTutils
import scipy as sp
from matplotlib import pyplot as plt
import GPdc
import ESutils
import search
from tqdm import tqdm, tqdm_gui
import DIRECT


seed=123
sp.random.seed(seed)
d=2
lb = sp.array([[-1.]*d])
ub = sp.array([[1.]*d])
sl=0.
su=1.
sfn = lambda x:1e-8
fls = 1.0
sls = (su-sl)*fls
dcc=1.0
cfn = lambda x: sp.exp(-dcc*x.flatten()[0])
[ojf,truexmin,ymin] = OPTutils.gensquexpIPdraw(d,lb,ub,sl,su,sfn,sls,cfn)

#what are the best mins in planes away from true?
def planemin(xp):
    def dirwrap(x,y):
        z,c = ojf(sp.hstack([xp,x]),-1,[sp.NaN],override=True)
        return (z,0)
    [xm,ym,ierror] = DIRECT.solve(dirwrap,lb,ub,user_data=[], algmethod=1, maxf=20000, logfilename='/dev/null')
    ye,c = ojf(sp.hstack([0.,xm]),-1,[sp.NaN],override=True)
    r=ye-ymin
    return [xm,ym,ye,r]

plane00 = planemin(0.)
plane025 = planemin(0.25)
plane05 = planemin(0.5)
plane001 = planemin(0.01)
plane10 = planemin(1.)
nreps=8
bd=50

kindex = GPdc.MAT52CS
prior = sp.array([0.]+[-1.]*(d+1)+[-2.])
sprior = sp.array([1.]*(d+2)+[2.])
kernel = [kindex,prior,sprior]

names = ["../cache/IPS/PESIPS_"+str(dcc)+"_"+str(fls)+"_"+str(seed)+"_"+str(i)+".p" for i in xrange(nreps)]
results = search.multiPESIPS(ojf,lb,ub,kernel,bd,names)

f,a = plt.subplots(3)
aot = a[0].twinx()
for r in results:
    a[0].plot(r[0][:,0].flatten(),'b')
    a[0].set_ylabel("augx")
    aot.plot(r[5],'r')
    a[1].semilogx(sp.log10(r[11].flatten()-ymin),'b')
    a[1].set_ylabel("regret")
    a[2].semilogx([sum(r[5][:j]) for j in xrange(len(r[5]))],sp.log10(r[11].flatten()-ymin),'b')
    a[2].set_ylabel("regret/c")


subset = [0.,0.25,1.]
c = ['g','r','c']

print "XXX"
print [truexmin,ymin]
print plane00
#a[2].plot([0,bd],[sp.log10(plane00[3])]*2,c[0],linestyle='--')
print plane025
a[2].plot([1.,bd],[sp.log10(plane025[3])]*2,color=c[1],linestyle='--')
print plane10
a[2].plot([1.,bd],[sp.log10(plane10[3])]*2,color=c[2],linestyle='--')
print plane001
a[2].plot([1.,bd],[sp.log10(plane001[3])]*2,color='k',linestyle='--')
print "XXX"

kindex = GPdc.MAT52CS
priora = sp.array([0.]+[-1.]*(d)+[-2.])
spriora = sp.array([1.]*(d+1)+[2.])
kernela = [kindex,priora,spriora]

for i,xs in enumerate(subset):
    def ojfa(x,s,d,override=False):
        return ojf(sp.hstack([[xs],x.flatten()]),s,d,override=override)
    names = ["../cache/IPS/PESIS_"+str(xs)+"_"+str(dcc)+"_"+str(fls)+"_"+str(seed)+"_"+str(k)+".p" for k in xrange(nreps)]
    results = search.multiPESIS(ojfa,lb,ub,kernela,bd,names)
    
    for r in results:
        a[0].plot(r[5],color=c[i])
        z=sp.array([ojf(sp.hstack([0.,r[4][j,:]]) ,0.,[sp.NaN],override=True)[0] for j in xrange(r[4].shape[0])])
        a[1].plot(sp.log10(z-ymin),color=c[i])
        
        a[2].plot([sum(r[5][:j]) for j in xrange(len(r[5]))],sp.log10(z-ymin),color=c[i])
        
    
plt.show()
