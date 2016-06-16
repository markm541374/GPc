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
import PES
import ESutils
#start with random

def randomaq(optstate,persist,**para):
    logger.info('randomaq')
    q = sp.random.uniform(size=len(para['lb']))
    return [l+x*(u-l) for l,u,x in zip(para['lb'],para['ub'],q)],para['ev'],persist,dict()


randomprior = {'ev':{'s':0.,'d':[sp.NaN]},'lb':[-1.-1.],'ub':[1.,1.]}

random = randomaq,randomprior

# and grid

def bruteaq(optstate,persist,**para):
    if persist==None:
        persist = {'pwr':0,'idx':0,'d':len(para['ub'])}

    
    pwr = persist['pwr']
    idx = persist['idx']
    d = persist['d']
    k=2**pwr
    q=[0]*d
    logger.info('bruteaq griddiv={}'.format(k))
    for j in xrange(d):
        
        a,b = divmod(idx,k**(d-j-1))
        idx=b
        q[j]=(2*a+1)/float(2*k)
    
    
    if persist['idx']+1>= k**d:
        persist['pwr']+=1
        persist['idx']=0
    else:
        persist['idx']+=1
    return [l+x*(u-l) for l,u,x in zip(para['lb'],para['ub'],q)],para['ev'],persist,dict()

bruteprior = {'ev':{'s':0.,'d':[sp.NaN]},'lb':[-1.-1.],'ub':[1.,1.]}

brute = bruteaq, bruteprior

#EIMAP
def EIMAPaq(optstate,persist,ev=None, ub = None, lb=None, nrandinit=None, mprior=None,sprior=None,kindex = None,directmaxiter=None):
    if persist==None:
        persist = {'n':0,'d':len(ub)}
    n = persist['n']
    d = persist['d']
    if n<nrandinit:
        persist['n']+=1
        return randomaq(optstate,persist,ev=ev,lb=lb,ub=ub)
    logger.info('EIMAPaq')
    #logger.debug(sp.vstack([e[0] for e in optstate.ev]))
    #raise
    x=sp.vstack(optstate.x)
    y=sp.vstack(optstate.y)
    s= sp.vstack([e['s'] for e in optstate.ev])
    dx=[e['d'] for e in optstate.ev]
    MAP = GPdc.searchMAPhyp(x,y,s,dx,mprior,sprior, kindex)
    logger.info('MAPHYP {}'.format(MAP))

    G = GPdc.GPcore(x,y,s,dx,GPdc.kernel(kindex,d,MAP))
    def directwrap(xq,y):
        xq.resize([1,d])
        a = G.infer_lEI(xq,[ev['d']])
        return (-a[0,0],0)
    
    [xmin,ymin,ierror] = DIRECT.solve(directwrap,lb,ub,user_data=[], algmethod=0, maxf = directmaxiter, logfilename='/dev/null')
    #logger.debug([xmin,ymin,ierror])
    persist['n']+=1
    return [i for i in xmin],ev,persist,{'MAPHYP':MAP,'logEImin':ymin,'DIRECTmessage':ierror}
    
EIMAPprior = {
                'ev':{'s':1e-9,'d':[sp.NaN]},
                'lb':[-1.-1.],
                'ub':[1.,1.],
                'nrandinit':10,
                'mprior':sp.array([1.,0.,0.]),
                'sprior':sp.array([1.,1.,1.]),
                'kindex':GPdc.MAT52,
                'directmaxiter':10000
                }

EIMAP = EIMAPaq, EIMAPprior


#PES with fixed s ev
def PESfsaq(optstate,persist,**para):
    if persist==None:
        persist = {'n':0,'d':len(para['ub'])}
    n = persist['n']
    d = persist['d']
    if n<para['nrandinit']:
        persist['n']+=1
        
        return randomaq(optstate,persist,**para)
    logger.info('PESfsaq')
    #logger.debug(sp.vstack([e[0] for e in optstate.ev]))
    #raise
    x=sp.vstack(optstate.x)
    y=sp.vstack(optstate.y)
    s= sp.vstack([e['s'] for e in optstate.ev])
    dx=[e['d'] for e in optstate.ev]
    
    pesobj = PES.PES(x,y,s,dx,para['lb'],para['ub'],para['kindex'],para['mprior'],para['sprior'],DH_SAMPLES=para['DH_SAMPLES'],DM_SAMPLES=para['DM_SAMPLES'], DM_SUPPORT=para['DM_SUPPORT'],DM_SLICELCBPARA=para['DM_SLICELCBPARA'],mode=para['SUPPORT_MODE'],noS=para['noS'])
    
    [xmin,ymin,ierror] = pesobj.search_pes(para['ev']['s'],volper=para['volper'])
    #logger.debug([xmin,ymin,ierror])
    return [i for i in xmin],para['ev'],persist,{'HYPdraws':[k.hyp for k in pesobj.G.kf],'mindraws':pesobj.Z,'DIRECTmessage':ierror,'PESmin':ymin}

PESfsprior = {
            'ev':{'s':1e-9,'d':[sp.NaN]},
            'lb':[-1.-1.],
            'ub':[1.,1.],
            'volper':1e-5,
            'mprior':sp.array([1.,0.,0.]),
            'sprior':sp.array([1.,1.,1.]),
            'kindex':GPdc.MAT52,
            'directmaxiter':10000,
            'DH_SAMPLES':8,
            'DM_SAMPLES':8,
            'DM_SUPPORT':800,
            'SUPPORT_MODE':[ESutils.SUPPORT_LAPAPR],
            'DM_SLICELCBPARA':1.,
            'noS':False,
            'nrandinit':10
            }

PESfs = PESfsaq,PESfsprior



#PES with variable s ev give costfunction
def PESvsaq(optstate,persist,**para):
    if persist==None:
        persist = {'n':0,'d':len(para['ub'])}
    n = persist['n']
    d = persist['d']
    if n<para['nrandinit']:
        persist['n']+=1
        
        return randomaq(optstate,persist,**para)
    logger.info('PESvsaq')
    #logger.debug(sp.vstack([e[0] for e in optstate.ev]))
    #raise
    x=sp.vstack(optstate.x)
    y=sp.vstack(optstate.y)
    s= sp.vstack([e['s'] for e in optstate.ev])
    dx=[e['d'] for e in optstate.ev]
    
    pesobj = PES.PES(x,y,s,dx,para['lb'],para['ub'],para['kindex'],para['mprior'],para['sprior'],DH_SAMPLES=para['DH_SAMPLES'],DM_SAMPLES=para['DM_SAMPLES'], DM_SUPPORT=para['DM_SUPPORT'],DM_SLICELCBPARA=para['DM_SLICELCBPARA'],mode=para['SUPPORT_MODE'],noS=para['noS'])
    
    [xmin,ymin,ierror] = pesobj.search_acq(para['cfn'],para['logsl'],para['logsu'],volper=para['volper'])
    
    logger.debug([xmin,ymin,ierror])
    para['ev']['s']=10**xmin[-1]
    xout = [i for i in xmin[:-1]]
    return xout,para['ev'],persist,{'HYPdraws':[k.hyp for k in pesobj.G.kf],'mindraws':pesobj.Z,'DIRECTmessage':ierror,'PESmin':ymin}

    return

PESvsprior = {
            'ev':{'s':1e-7,'d':[sp.NaN]},
            'lb':[-1.-1.],
            'ub':[1.,1.],
            'volper':1e-5,
            'mprior':sp.array([1.,0.,0.]),
            'sprior':sp.array([1.,1.,1.]),
            'kindex':GPdc.MAT52,
            'directmaxiter':10000,
            'DH_SAMPLES':8,
            'DM_SAMPLES':8,
            'DM_SUPPORT':800,
            'SUPPORT_MODE':[ESutils.SUPPORT_LAPAPR],
            'DM_SLICELCBPARA':1.,
            'noS':False,
            'nrandinit':10,
            'cfn':lambda x,ev:42.,
            'logsu':-3,
            'logsl':-10
            }
            
PESvs = PESvsaq,PESvsprior

def PESbsaq(optstate,persist,**para):
    if persist==None:
        persist = {'n':0,'d':len(para['ub'])}
    n = persist['n']
    d = persist['d']
    if n<para['nrandinit']:
        persist['n']+=1
        para['ev']['xa'] = sp.random.uniform(para['xal'],para['xau'])
        return randomaq(optstate,persist,**para)
    logger.info('PESssaq')
    
    x=sp.hstack([sp.vstack(optstate.x),sp.vstack([e['xa'] for e in optstate.ev])])
    
    y=sp.vstack(optstate.y)
    s= sp.vstack([e['s'] for e in optstate.ev])
    dx=[e['d'] for e in optstate.ev]
    
    pesobj = PES.PES_inplane(x,y,s,dx,[para['xal']]+para['lb'],[para['xau']]+para['ub'],para['kindex'],para['mprior'],para['sprior'],0,0,DH_SAMPLES=para['DH_SAMPLES'], DM_SAMPLES=para['DM_SAMPLES'], DM_SUPPORT=para['DM_SUPPORT'],DM_SLICELCBPARA=para['DM_SLICELCBPARA'],mode=para['SUPPORT_MODE'])

    [xmin,ymin,ierror] = pesobj.search_acq(para['cfn'],lambda s:para['ev']['s'],volper=para['volper'])
    logger.debug([xmin,ymin,ierror])
    para['ev']['xa']=xmin[0]
    xout = [i for i in xmin[1:]]
    return xout,para['ev'],persist,{'HYPdraws':[k.hyp for k in pesobj.G.kf],'mindraws':pesobj.Z,'DIRECTmessage':ierror,'PESmin':ymin}


PESbsprior = {
            'ev':{'s':1e-9,'d':[sp.NaN],'xa':0.},
            'lb':[-1.-1.],
            'ub':[1.,1.],
            'volper':1e-5,
            'mprior':sp.array([1.,0.,0.,0.]),
            'sprior':sp.array([1.,1.,1.,1.]),
            'kindex':GPdc.MAT52,
            'directmaxiter':10000,
            'DH_SAMPLES':8,
            'DM_SAMPLES':8,
            'DM_SUPPORT':800,
            'SUPPORT_MODE':[ESutils.SUPPORT_LAPAPR],
            'DM_SLICELCBPARA':1.,
            'noS':False,
            'nrandinit':10,
            'cfn':lambda x,ev:42.,
            'xau':1.,
            'xal':0.
            }
            
PESbs = PESbsaq,PESbsprior