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

O = OPTutils.opt(ojf,lb,ub)
for i in xrange(40):
    O.step()




kindex = GPdc.SQUEXP

mprior = sp.array([0.]+[-1.]*d)
sprior = sp.array([1.]*(d+1))
maxf = 4000
s = 1e-6
ninit = 10

#para = [kindex,hlb,hub,maxf,s,ninit]
para = [kindex,mprior,sprior,maxf,s,ninit]


OE = OPTutils.EIMLE(ojf,lb,ub,para)

for i in xrange(30):
    OE.step()

para = dict()
para['kindex'] = GPdc.SQUEXP
para['mprior'] = sp.array([0.]+[-1.]*d)
para['sprior'] = sp.array([1.]*(d+1))
para['s'] = 1e-6
para['ninit'] = 10
para['maxf'] = 2500
para['DH_SAMPLES'] = 8
para['DM_SAMPLES'] = 8
para['DM_SUPPORT'] = 400
para['DM_SLICELCBPARA'] = 1.
OP = OPTutils.PESFX(ojf,lb,ub,para)
for i in xrange(30):
    try:
        OP.step()
    except RuntimeError as e:
        print e
        break
        
f,a = plt.subplots(4)
a[0].plot(OE.Ymin,'rx-')
a[1].plot(OE.T,'r')
a[2].plot([sum(OE.C[:i]) for i in xrange(len(OE.C))],'r')
a[3].semilogy(OE.compX(truexmin)[1,:].flatten(),'r')

a[0].plot(O.Ymin,'bx-')
a[1].plot(O.T,'b')
a[2].plot([sum(O.C[:i]) for i in xrange(len(O.C))],'b')
a[3].semilogy(O.compX(truexmin)[1,:].flatten(),'b')

a[0].plot(OP.Ymin,'gx-')
a[1].plot(OP.T,'g')
a[2].plot([sum(OP.C[:i]) for i in xrange(len(OP.C))],'g')
a[3].semilogy(OP.compX(truexmin)[1,:].flatten(),'g')
a[3].semilogy(OP.compX(truexmin)[0,:].flatten(),'g')
print OP.X
print OE.X
print OP.Y
plt.show()