# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import OPTutils
import scipy as sp
from matplotlib import pyplot as plt
import GPdc

ojf = OPTutils.quad
d=4
lb = sp.array([[-1.]*d])
ub = sp.array([[1.]*d])

O = OPTutils.opt(ojf,lb,ub)
for i in xrange(35):
    O.step()




kindex = GPdc.SQUEXP

mprior = sp.array([0.]+[-1.]*d)
sprior = sp.array([1.]*(d+1))
maxf = 4000
s = 1e-6
ninit = 12

#para = [kindex,hlb,hub,maxf,s,ninit]
para = [kindex,mprior,sprior,maxf,s,ninit]


OE = OPTutils.LCBMLE(ojf,lb,ub,para)

for i in xrange(23):
    OE.step()

f,a = plt.subplots(3)
a[0].plot(OE.Ymin,'r')
a[1].plot(OE.T,'r')
a[2].plot([sum(OE.C[:i]) for i in xrange(len(OE.C))],'r')

a[0].plot(O.Ymin,'b')
a[1].plot(O.T,'b')
a[2].plot([sum(O.C[:i]) for i in xrange(len(O.C))],'b')

print OE.X
print OE.Y
plt.show()