# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import OPTutils
import ESutils
from tqdm import tqdm, tqdm_gui
import scipy as sp
import os
import pickle
import math
from pathos.multiprocessing import Pool
import copy
def multiMLEFS(ojf,lb,ub,ki,s,b,fnames):
    def f(fn):
        return MLEFS(ojf,lb,ub,ki,s,b,fn)
    p = Pool(8)
    return p.map(f,fnames)
        
def MLEFS(ojf,lb,ub,ki,s,b,fname):
    #use kernel ki and evaluate ojf with variance s at step for a budget b
    sp.random.seed(int(os.urandom(4).encode('hex'), 16))
    d = lb.size
    volper=1e-8
    ninit = 10
    para = [ki[0],ki[1],ki[2],volper,s,ninit]
    if os.path.exists(fname):
        print "starting from "+str(fname)
        OE = OPTutils.EIMLE(ojf,lb,ub,para,initstate=pickle.load(open(fname,'rb')))
    else:
        print "fresh start"
        OE = OPTutils.EIMLE(ojf,lb,ub,para)
    if sum(OE.C)>=b:
        print "no further steps needed"
        k=0
        while sum(OE.C[:k])<b:
            k+=1
        state = [OE.X[:k,:],OE.Y[:k,:],OE.S[:k,:],OE.D[:k],OE.R[:k,:],OE.C[:k],OE.T[:k],OE.Tr[:k],OE.Ymin[:k],OE.Xmin[:k,:],OE.Yreg[:k,:], OE.Rreg[:k,:]]
    else:
        while sum(OE.C)<b:
            OE.step()
        state = [OE.X,OE.Y,OE.S,OE.D,OE.R,OE.C,OE.T,OE.Tr,OE.Ymin,OE.Xmin,OE.Yreg, OE.Rreg]
        print "saving as "+str(fname)
        pickle.dump(state,open(fname,'wb'))
    return state

def multiPESFS(ojf,lb,ub,ki,s,b,fnames):
    def f(fn):
        return PESFS(ojf,lb,ub,ki,s,b,fn)
    p = Pool(8)
    return map(f,fnames)
        
def PESFS(ojf,lb,ub,ki,s,b,fname):
    para = dict()
    para['kindex'] = ki[0]
    para['mprior'] = ki[1]
    para['sprior'] = ki[2]
    para['s'] = s
    para['ninit'] = 10
    para['volper'] = 1e-6
    para['DH_SAMPLES'] = 8
    para['DM_SAMPLES'] = 8
    para['DM_SUPPORT'] = 1200
    para['DM_SLICELCBPARA'] = 1.
    para['SUPPORT_MODE'] = [ESutils.SUPPORT_SLICELCB,ESutils.SUPPORT_SLICEPM]
    if os.path.exists(fname):
        print "starting from "+str(fname)
        OE = OPTutils.PESFS(ojf,lb,ub,para,initstate=pickle.load(open(fname,'rb')))
    else:
        print "fresh start"
        OE = OPTutils.PESFS(ojf,lb,ub,para)
    if sum(OE.C)>=b:
        print "no further steps needed"
        state = [OE.X,OE.Y,OE.S,OE.D,OE.R,OE.C,OE.T,OE.Tr,OE.Ymin,OE.Xmin, OE.Yreg, OE.Rreg]
    else:
        while sum(OE.C)<b:
            OE.step()
        state = [OE.X,OE.Y,OE.S,OE.D,OE.R,OE.C,OE.T,OE.Tr,OE.Ymin,OE.Xmin,OE.Yreg, OE.Rreg]
        pickle.dump(state,open(fname,'wb'))
    return state

def multiPESVS(ojf,lb,ub,ki,s,b,cfn,lsl,lsu,fnames):
    def f(fn):
        return PESVS(ojf,lb,ub,ki,s,b,cfn,lsl,lsu,fn)
    p = Pool(8)
    return p.map(f,fnames)
        
def PESVS(ojf,lb,ub,ki,s,b,cfn,lsl,lsu,fname):
    para = dict()
    para['kindex'] = ki[0]
    para['mprior'] = ki[1]
    para['sprior'] = ki[2]
    para['s'] = s
    para['ninit'] = 10
    para['volper'] = 1e-6
    para['DH_SAMPLES'] = 8
    para['DM_SAMPLES'] = 8
    para['DM_SUPPORT'] = 1200
    para['DM_SLICELCBPARA'] = 1.
    para['SUPPORT_MODE'] = [ESutils.SUPPORT_SLICELCB,ESutils.SUPPORT_SLICEPM]
    para['cfn'] = cfn
    para['logsl'] = lsl
    para['logsu'] = lsu
    para['s'] = 10**lsu
    if os.path.exists(fname):
        print "starting from "+str(fname)
        OE = OPTutils.PESVS(ojf,lb,ub,para,initstate=pickle.load(open(fname,'rb')))
    else:
        print "fresh start"
        OE = OPTutils.PESVS(ojf,lb,ub,para)
    
    
    if sum(OE.C)>=b:
        print "no further steps needed"
        state = [OE.X,OE.Y,OE.S,OE.D,OE.R,OE.C,OE.T,OE.Tr,OE.Ymin,OE.Xmin,OE.Yreg, OE.Rreg]
    else:
        pbar = tqdm(total=(b-sum(OE.C)))
        while sum(OE.C)<b:
            print "XXXXXXXXXXx"+str(sum(OE.C))+" "+str(b)+" "+str(sum(OE.C)<b)
            pbar.update(OE.C[-1])
            OE.step()
        state = [OE.X,OE.Y,OE.S,OE.D,OE.R,OE.C,OE.T,OE.Tr,OE.Ymin,OE.Xmin,OE.Yreg, OE.Rreg]
        pickle.dump(state,open(fname,'wb'))
    return state