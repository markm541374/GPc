# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import OPTutils
import scipy as sp
from matplotlib import pyplot as plt
import GPdc
import ESutils
from tqdm import tqdm_gui
d=2
lb = sp.array([[-1.]*d])
ub = sp.array([[1.]*d])
[ojf,truexmin] = OPTutils.gensquexpdraw(d,sp.array([-1.]*d),sp.array([1.]*d),ignores=1e-6)

O = OPTutils.opt(ojf,lb,ub)
for i in tqdm_gui(xrange(40),gui=True):
    O.step()


kindex = GPdc.SQUEXP
mprior = sp.array([0.]+[-1.]*d)
sprior = sp.array([1.]*(d+1))
volper=1e-8
s = 1e-6
ninit = 10
para = [kindex,mprior,sprior,volper,s,ninit]

OE = OPTutils.EIMLE(ojf,lb,ub,para)
#OL = OPTutils.LCBMLE(ojf,lb,ub,para)
for i in tqdm_gui(xrange(30),gui=True):
    OE.step()
    #OL.step()


para = dict()
para['kindex'] = GPdc.SQUEXP
para['mprior'] = sp.array([0.]+[-1.]*d)
para['sprior'] = sp.array([1.]*(d+1))
para['s'] = 1e-6
para['ninit'] = 10
#para['maxf'] = 2500
para['volper'] = 1e-7
para['DH_SAMPLES'] = 8
para['DM_SAMPLES'] = 8
para['DM_SUPPORT'] = 400
para['DM_SLICELCBPARA'] = 1.
para['SUPPORT_MODE'] = ESutils.SUPPORT_SLICELCB
OP = OPTutils.PESFS(ojf,lb,ub,para)
for i in tqdm_gui(xrange(30),gui=True):
    OP.step()
    

f,a = plt.subplots(7)
O.plot(truexmin,a,'b')
OE.plot(truexmin,a,'r')
OP.plot(truexmin,a,'g')
#OL.plot(truexmin,a,'c')

plt.show()