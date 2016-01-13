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
for i in xrange(20):
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
for i in xrange(10):
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
para['cfn'] = lambda x,s: ((1e-6)/s)**0.2
para['logsl'] = -8.
para['logsu'] = -2.
"""
import cProfile, pstats, StringIO
pr = cProfile.Profile()
pr.enable()
"""

OV = OPTutils.PESVS(ojf,lb,ub,para)
for i in xrange(10):
    try:
        OV.step()
    except RuntimeError as e:
        print e
        break

"""
pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()
"""

OS = OPTutils.PESFX(ojf,lb,ub,para)
for i in xrange(10):
    try:
        OS.step()
    except RuntimeError as e:
        print e
        break


f,a = plt.subplots(7)

OE.plot(truexmin,a,'r')
OV.plot(truexmin,a,'g')
OS.plot(truexmin,a,'b')


plt.show()