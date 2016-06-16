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

def argminrecc(optstate,**para):
    if para['onlyafter']>len(optstate.y) or not len(optstate.y)%para['everyn']==0:
        return [sp.NaN for i in para['lb']],{'didnotrun':True}
    
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
    s= sp.vstack([e['s'] for e in optstate.ev])
    dx=[e['d'] for e in optstate.ev]
    MAP = GPdc.searchMAPhyp(x,y,s,dx,para['mprior'],para['sprior'], para['kindex'])
    logger.info('MAPHYP {}'.format(MAP))
    G = GPdc.GPcore(x,y,s,dx,GPdc.kernel(para['kindex'],d,MAP))
    def directwrap(xq,y):
        xq.resize([1,d])
        a = G.infer_m(xq,[[sp.NaN]])
        return (a[0,0],0)
    [xmin,ymin,ierror] = DIRECT.solve(directwrap,para['lb'],para['ub'],user_data=[], algmethod=1, volper=para['volper'], logfilename='/dev/null')
    return [i for i in xmin],{'MAPHYP':MAP,'ymin':ymin}

gpmapprior = {
                'ev':{'s':1e-9,'d':[sp.NaN]},
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

def gpmapasrecc(optstate,**para):
    if para['onlyafter']>len(optstate.y) or not len(optstate.y)%para['everyn']==0:
        return [sp.NaN for i in para['lb']],{'didnotrun':True}
    logger.info('gpmapas reccomender')
    d=len(para['lb'])
    
    x=sp.hstack([sp.vstack(optstate.x),sp.vstack([e['xa'] for e in optstate.ev])])
    
    
    y=sp.vstack(optstate.y)
    s= sp.vstack([e['s'] for e in optstate.ev])
    dx=[e['d'] for e in optstate.ev]
    MAP = GPdc.searchMAPhyp(x,y,s,dx,para['mprior'],para['sprior'], para['kindex'])
    logger.info('MAPHYP {}'.format(MAP))
    G = GPdc.GPcore(x,y,s,dx,GPdc.kernel(para['kindex'],d+1,MAP))
    def directwrap(xq,y):
        xq.resize([1,d])
        xe = sp.hstack([xq,sp.array([[0.]])])
        #print xe
        a = G.infer_m(xe,[[sp.NaN]])
        return (a[0,0],0)
    [xmin,ymin,ierror] = DIRECT.solve(directwrap,para['lb'],para['ub'],user_data=[], algmethod=1, volper=para['volper'], logfilename='/dev/null')
    logger.info('reccsearchresult: {}'.format([xmin,ymin,ierror]))
    return [i for i in xmin],{'MAPHYP':MAP,'ymin':ymin}

gpmapasprior = {
                'ev':{'s':1e-9,'d':[sp.NaN],'xa':0.},
                #'ev':[1e-9,[sp.NaN]],
                'lb':[-1.-1.],
                'ub':[1.,1.],
                'mprior':sp.array([1.,0.,0.,0.]),
                'sprior':sp.array([1.,1.,1.,1.]),
                'kindex':GPdc.MAT52,
                'volper':1e-6,
                'onlyafter':10,
                'everyn':1,
                }

gpasmap = gpmapasrecc,gpmapasprior