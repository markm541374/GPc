# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import OPTutils
import ESutils
from tqdm import tqdm, tqdm_gui
import scipy as sp

def MLEFS(ojf,lb,ub,ki,s,b):
    #use kernel ki and evaluate ojf with variance s at step for a budget b
    d = lb.size
    volper=1e-8
    ninit = 10
    para = [ki[0],ki[1],ki[2],volper,s,ninit]
    OE = OPTutils.EIMLE(ojf,lb,ub,para)
    ns = int(b/OE.C[0]) - ninit
    for i in tqdm(xrange(ns)):
        OE.step()
    return [OE.X,OE.Y,OE.S,OE.D,OE.R,OE.C,OE.T,OE.Tr,OE.Xmin,OE.Ymin,OE.Rreg, OE.Yreg]

def PESFS(ojf,lb,ub,ki,s,b):
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
    OE = OPTutils.PESFS(ojf,lb,ub,para)
    ns = int(b/OE.C[0]) - para['ninit']
    for i in tqdm(xrange(ns)):
        OE.step()
    return [OE.X,OE.Y,OE.S,OE.D,OE.R,OE.C,OE.T,OE.Tr,OE.Xmin,OE.Ymin,OE.Rreg, OE.Yreg]

def PESVS(ojf,lb,ub,ki,s,b,cfn,lsl,lsu):
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

    OE = OPTutils.PESVS(ojf,lb,ub,para)
    pbar = tqdm(total=(b-sum(OE.C)))
    while sum(OE.C)<b:
        print "XXXXXXXXXXx"+str(sum(OE.C))+" "+str(b)+" "+str(sum(OE.C)<b)
        pbar.update(OE.C[-1])
        OE.step()
    return [OE.X,OE.Y,OE.S,OE.D,OE.R,OE.C,OE.T,OE.Tr,OE.Xmin,OE.Ymin,OE.Rreg, OE.Yreg]