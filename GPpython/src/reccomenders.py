# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import scipy as sp
import os
import time
import DIRECT
import logging

logger = logging.getLogger(__name__)

import GPdc

def argminrecc(optstate,**kwargs):
    logger.info('argmin reccomender')
    xinc = optstate.x[0]
    yinc = 1e99
    for x,y in zip(optstate.x,optstate.y):
        if y<yinc:
            xinc = x
            yinc=y
    return xinc,{'yinc':yinc}

argminpara = dict()
argmin = argminrecc, argminpara

def gpmaprecc(optstate,**para):
    if para['onlyafter']>len(optstate.y) or not len(optstate.y)%para['everyn']==0:
        return [sp.NaN for i in para['lb']],{'didnotrun':True}
    logger.info('gpmap reccomender')
    d=len(para['lb'])
    x=sp.vstack(optstate.x)
    y=sp.vstack(optstate.y)
    s= sp.vstack([e[0] for e in optstate.ev])
    dx=[e[1] for e in optstate.ev]
    MAP = GPdc.searchMAPhyp(x,y,s,dx,para['mprior'],para['sprior'], para['kindex'])
    logger.info('MAPHYP {}'.format(MAP))
    G = GPdc.GPcore(x,y,s,dx,GPdc.kernel(para['kindex'],d,MAP))
    def directwrap(xq,y):
        xq.resize([1,d])
        a = G.infer_m(xq,[[sp.NaN]])
        return (a[0,0],0)
    [xmin,ymin,ierror] = DIRECT.solve(directwrap,para['lb'],para['ub'],user_data=[], algmethod=0, volper=para['volper'], logfilename='/dev/null')
    return [i for i in xmin],{'MAPHYP':MAP,'ymin':ymin}

gpmapprior = {
                'ev':[1e-9,[sp.NaN]],
                'lb':[-1.-1.],
                'ub':[1.,1.],
                'mprior':sp.array([1.,0.,0.]),
                'sprior':sp.array([1.,1.,1.]),
                'kindex':GPdc.MAT52,
                'volper':1e-6,
                'onlyafter':10,
                'everyn':1
                }

gpmap = gpmaprecc,gpmapprior