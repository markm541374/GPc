# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "mark"
__date__ = "$22-Nov-2015 21:27:19$"

import scipy as sp
import ctypes as ct
libGP = ct.cdll.LoadLibrary('../../dist/Release/GNU-Linux/libGPshared.so')
libGP.k.restype = ct.c_double

ctpd = ct.POINTER(ct.c_double)
cint = ct.c_int
class GP_LKonly:
    def __init__(self, X_s, Y_s, S_s, D_s, kf):
        [n ,D] = X_s.shape
        R = ct.c_double()
        Dc = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_s]
        libGP.newGP_LKonly(cint(D),ct.cint(n),X_s.ctypes.data_as(ctpd),Y_s.ctypes.data_as(ctpd),S_s.ctypes.data_as(ctpd),(cint*len(Dc))(*Dc), cint(kf.Kindex), kf.hyp.ctypes.data_as(ctpd),ct.byref(R))
        self.l = R.value
        return
    
    def llk(self):
        return self.l

class GPcore:
    def __init__(self, X_s, Y_s, S_s, D_s, kf):
        if type(kf) is kernel:
            self.size = 1
            kf = [kf]
        else:
            self.size = len(kf)
        allhyp = sp.hstack([k.hyp for k in kf])
        
        [self.n ,self.D] = X_s.shape
        Dx = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_s]
        self.s = libGP.newGP_hypset(cint(self.D),cint(self.n),cint(kf[0].Kindex),X_s.ctypes.data_as(ctpd),Y_s.ctypes.data_as(ctpd),S_s.ctypes.data_as(ctpd),(cint*len(Dx))(*Dx),allhyp.ctypes.data_as(ctpd),cint(self.size))
        self.Y_s=Y_s
        
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_s]
        
        libGP.presolv(self.s,cint(self.size))
        
        return
    
    def __del__(self):
        libGP.killGP(self.s)
        return
    
    def printc(self):
        print self.size
        libGP.ping(self.s, cint(self.size))
        return
    
    def infer_m(self,X_i,D_i):
        ns=X_i.shape[0]
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_i]
        R=sp.vstack([sp.empty(ns)]*self.size)
        libGP.infer_m(self.s, cint(self.size), ns,X_i.ctypes.data_as(ctpd),(cint*len(D))(*D),R.ctypes.data_as(ctpd))
        return R
    
    def infer_full(self,X_i,D_i):
        ns=X_i.shape[0]
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_i]
        R=sp.vstack([sp.empty([ns+1,ns])]*self.size)
        libGP.infer_full(self.s, cint(self.size), ns,X_i.ctypes.data_as(ctpd),(cint*len(D))(*D),R.ctypes.data_as(ctpd))
        m = sp.vstack([R[i*(ns+1),:] for i in xrange(self.size)])
        V = sp.vstack([R[(ns+1)*i+1:(ns+1)*(i+1),:] for i in xrange(self.size)])
        return [m,V]
    
    def infer_diag(self,X_i,D_i):
        ns=X_i.shape[0]
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_i]
        R=sp.vstack([sp.empty([2,ns])]*self.size)
        libGP.infer_diag(self.s,cint(self.size), ns,X_i.ctypes.data_as(ctpd),(cint*len(D))(*D),R.ctypes.data_as(ctpd))
        m = sp.vstack([R[i*2,:] for i in xrange(self.size)])
        V = sp.vstack([R[i*2+1,:] for i in xrange(self.size)])
        return [m,V]
    def draw(self,X_i,D_i,m):
        #make m draws at X_i Nd, X, D, R, m
        ns=X_i.shape[0]
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_i]
        R=sp.empty([m*self.size,ns])
        libGP.draw(self.s, cint(self.size), cint(ns), X_i.ctypes.data_as(ctpd), (cint*len(D))(*D),R.ctypes.data_as(ctpd),cint(m))
        return R
    def llk(self):
        R = sp.empty(self.size)
        libGP.llk(self.s, cint(self.size), R.ctypes.data_as(ctpd))
        return R
    
    def infer_LCB(self,X_i,D_i, p):
        ns=X_i.shape[0]
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_i]
        R=sp.empty([self.size,ns])
        libGP.infer_LCB(self.s, cint(self.size), ns,X_i.ctypes.data_as(ctpd),(cint*len(D))(*D), ct.c_double(p), R.ctypes.data_as(ctpd))
        
        return R
    
    def infer_EI(self,X_i,D_i):
        ns=X_i.shape[0]
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_i]
        R=sp.empty([self.size,ns])
        libGP.infer_EI(self.s, cint(self.size),ns,X_i.ctypes.data_as(ctpd),(cint*len(D))(*D), R.ctypes.data_as(ctpd))
        return R
#kf = gen_sqexp_k_d([1.,0.3])


SQUEXP = 0
LIN1 = 1
LINXSQUEXP = 2
LINSQUEXPXSQUEXP = 3
SQUEXP1SSQUEXP = 4
SSPS = 5
class kernel(object):
    def __init__(self,K,D,H):
        self.dim = D
        self.hyp = sp.array(H)
        self.Kindex = K
        #ihyp are derived from the hyperparameters for speed and will be 1/h^2 etc.
        self.ihyp = sp.empty(self.hyp.shape[0])
        libGP.hypconvert(self.hyp.ctypes.data_as(ctpd),cint(self.Kindex), cint(self.dim), self.ihyp.ctypes.data_as(ctpd))
        return
    
    def __call__(self,x1, x2, d1=[sp.NaN], d2=[sp.NaN]):
        D1 = 0 if sp.isnan(d1[0]) else int(sum([8**x for x in d1]))
        D2 = 0 if sp.isnan(d2[0]) else int(sum([8**x for x in d2]))
        r=libGP.k(x1.ctypes.data_as(ctpd),x2.ctypes.data_as(ctpd), cint(D1),cint(D2),cint(self.dim),self.ihyp.ctypes.data_as(ctpd),cint(self.Kindex))
        return r
    

class gen_sqexp_k_d():
    def __init__(self,theta):
        self.dim = len(theta)-1
        self.hyp = sp.array(theta)
        self.hypinv = sp.array([1./x**2 for x in theta])
        self.hypinv[0] = theta[0]**2
        self.Kindex = 0;
        return
    def __call__(self,x1, x2, d1=[sp.NaN], d2=[sp.NaN]):
        D1 = 0 if sp.isnan(d1[0]) else int(sum([8**x for x in d1]))
        D2 = 0 if sp.isnan(d2[0]) else int(sum([8**x for x in d2]))
        r=libGP.k(x1.ctypes.data_as(ctpd),x2.ctypes.data_as(ctpd), cint(D1),cint(D2),cint(self.dim),self.hypinv.ctypes.data_as(ctpd),cint(0))
        return r
    
class gen_lin1_k_d():
    def __init__(self,theta):
        self.hyp = sp.array(theta)
        self.Kindex = 1
        self.hypinv = sp.array(theta)
        self.hypinv[0] = self.hypinv[0]**2
        self.hypinv[2] = self.hypinv[2]**2
        
        return
    
    def __call__(self,x1, x2, d1=[sp.NaN], d2=[sp.NaN]):
        D1 = 0 if sp.isnan(d1[0]) else int(sum([8**x for x in d1]))
        D2 = 0 if sp.isnan(d2[0]) else int(sum([8**x for x in d2]))
        r=libGP.k(x1.ctypes.data_as(ctpd),x2.ctypes.data_as(ctpd), cint(D1),cint(D2),cint(-42),self.hypinv.ctypes.data_as(ctpd),cint(1))
        return r

def searchMLEhyp(X,Y,S,D,lb,ub, ki, mx=5000,fg=-1e9):
    libGP.SetHypSearchPara(cint(mx),ct.c_double(fg))
    ns=X.shape[0]
    dim = X.shape[1]
    Dx = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D]
    hy = sp.empty(libGP.numhyp(cint(ki),cint(dim)))
    
    lk = sp.empty(1)
    r = libGP.HypSearchMLE(cint(dim),cint(len(Dx)),X.ctypes.data_as(ctpd),Y.ctypes.data_as(ctpd),S.ctypes.data_as(ctpd),(cint*len(Dx))(*Dx),lb.ctypes.data_as(ctpd),ub.ctypes.data_as(ctpd),cint(ki), hy.ctypes.data_as(ctpd),lk.ctypes.data_as(ctpd))
    
    return hy


def searchMAPhyp(X,Y,S,D,m,s, ki, MAPmargin = 2.5, mx=5000,fg=-1e9):
    libGP.SetHypSearchPara(cint(mx),ct.c_double(fg))
    ns=X.shape[0]
    dim = X.shape[1]
    Dx = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D]
    hy = sp.empty(libGP.numhyp(cint(ki),cint(dim)))
    
    lk = sp.empty(1)
    r = libGP.HypSearchMAP(cint(dim),cint(len(Dx)),X.ctypes.data_as(ctpd),Y.ctypes.data_as(ctpd),S.ctypes.data_as(ctpd),(cint*len(Dx))(*Dx),m.ctypes.data_as(ctpd),s.ctypes.data_as(ctpd),ct.c_double(MAPmargin),cint(ki), hy.ctypes.data_as(ctpd),lk.ctypes.data_as(ctpd))
    
    return hy

#just for quickly making test draws
def buildKsym_d(kf,x,d):
        #x should be  column vector
        (l,_)=x.shape
        K=sp.matrix(sp.empty([l,l]))
        for i in range(l):
            K[i,i]=kf(x[i,:],x[i,:],d1=d[i],d2=d[i])+10**-10
            for j in range(i+1,l):
                K[i,j]=kf(x[i,:],x[j,:],d1=d[i],d2=d[j])
                
                K[j,i]=K[i,j]
                
        return K