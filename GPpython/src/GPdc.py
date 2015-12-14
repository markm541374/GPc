# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "mark"
__date__ = "$22-Nov-2015 21:27:19$"

import scipy as sp
import ctypes as ct
libGP = ct.cdll.LoadLibrary('../../dist/Release/GNU-Linux/libGPshared.so')
libGP.k.restype = ct.c_double

class GP_LKonly:
    def __init__(self, X_s, Y_s, S_s, D_s, kf):
        [n ,D] = X_s.shape
        R = ct.c_double()
        Dc = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_s]
        libGP.newGP_LKonly(ct.c_int(D),ct.c_int(n),X_s.ctypes.data_as(ct.POINTER(ct.c_double)),Y_s.ctypes.data_as(ct.POINTER(ct.c_double)),S_s.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*len(Dc))(*Dc), ct.c_int(kf.Kindex), kf.hyp.ctypes.data_as(ct.POINTER(ct.c_double)),ct.byref(R))
        self.l = R.value
        return
    
    def llk(self):
        return self.l

class GPcore:
    def __init__(self, X_s, Y_s, S_s, D_s, kf):
        
        [self.n ,self.D] = X_s.shape
        self.s = libGP.newGP(ct.c_int(self.D),ct.c_int(self.n),ct.c_int(kf.Kindex))
        self.Y_s=Y_s
        libGP.set_X(self.s,X_s.ctypes.data_as(ct.POINTER(ct.c_double)))
        libGP.set_Y(self.s,Y_s.ctypes.data_as(ct.POINTER(ct.c_double)))
        libGP.set_S(self.s,S_s.ctypes.data_as(ct.POINTER(ct.c_double)))
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_s]
        
        libGP.set_D(self.s,(ct.c_int*len(D))(*D))
        libGP.set_hyp(self.s,kf.hyp.ctypes.data_as(ct.POINTER(ct.c_double)))
        libGP.build_K(self.s)
        
        libGP.fac(self.s)
        libGP.presolv(self.s)
        
        return
    
    def __del__(self):
        libGP.killGP(self.s)
        return
    
    def printc(self):
        libGP.ping(self.s)
        return
    
    def infer_m(self,X_i,D_i):
        ns=X_i.shape[0]
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_i]
        R=sp.matrix(sp.empty(ns)).T
        libGP.infer_m(self.s,ns,X_i.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*len(D))(*D),R.ctypes.data_as(ct.POINTER(ct.c_double)))
        return R
    
    def infer_full(self,X_i,D_i):
        ns=X_i.shape[0]
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_i]
        R=sp.matrix(sp.empty([ns+1,ns]))
        libGP.infer_full(self.s,ns,X_i.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*len(D))(*D),R.ctypes.data_as(ct.POINTER(ct.c_double)))
        m = R[0,:].T
        V = R[1:,:]
        return [m,V]
    
    def infer_diag(self,X_i,D_i):
        ns=X_i.shape[0]
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_i]
        R=sp.matrix(sp.empty([2,ns]))
        libGP.infer_diag(self.s,ns,X_i.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*len(D))(*D),R.ctypes.data_as(ct.POINTER(ct.c_double)))
        m = R[0,:].T
        V = R[1,:].T
        return [m,V]
    def draw(self,X_i,D_i,m):
        #make m draws at X_i Nd, X, D, R, m
        ns=X_i.shape[0]
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_i]
        R=sp.matrix(sp.empty([ns,m]))
        libGP.draw(self.s, ct.c_int(ns), X_i.ctypes.data_as(ct.POINTER(ct.c_double)), (ct.c_int*len(D))(*D),R.ctypes.data_as(ct.POINTER(ct.c_double)),ct.c_int(m))
        return R
    
    def llk(self):
        R = ct.c_double()
        libGP.llk(self.s,ct.byref(R))
        return R.value
#kf = gen_sqexp_k_d([1.,0.3])

class GP_timing(GPcore):
    def __init__(self, X_s, Y_s, S_s, D_s, kf):
        
        [self.n ,self.D] = X_s.shape
        self.s = libGP.newGP_timing(ct.c_int(self.D),ct.c_int(self.n),ct.c_int(kf.Kindex))
        self.Y_s=Y_s
        libGP.set_X(self.s,X_s.ctypes.data_as(ct.POINTER(ct.c_double)))
        libGP.set_Y(self.s,Y_s.ctypes.data_as(ct.POINTER(ct.c_double)))
        libGP.set_S(self.s,S_s.ctypes.data_as(ct.POINTER(ct.c_double)))
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_s]
        
        libGP.set_D(self.s,(ct.c_int*len(D))(*D))
        libGP.set_hyp(self.s,kf.hyp.ctypes.data_as(ct.POINTER(ct.c_double)))
        libGP.build_K(self.s)
        libGP.fac(self.s)
        libGP.presolv(self.s)
        return
    
    def timing(self,c):
        T = sp.zeros(c)
        libGP.timing(self.s,  ct.c_int(c), T.ctypes.data_as(ct.POINTER(ct.c_double)))
        return T
    
    
class GP_EI_random(GPcore):
    def __init__(self, X_s, Y_s, S_s, D_s, kf):
        
        [self.n ,self.D] = X_s.shape
        self.s = libGP.newEI_random(ct.c_int(self.D),ct.c_int(self.n),ct.c_int(kf.Kindex))
        self.Y_s=Y_s
        libGP.set_X(self.s,X_s.ctypes.data_as(ct.POINTER(ct.c_double)))
        libGP.set_Y(self.s,Y_s.ctypes.data_as(ct.POINTER(ct.c_double)))
        libGP.set_S(self.s,S_s.ctypes.data_as(ct.POINTER(ct.c_double)))
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_s]
        
        libGP.set_D(self.s,(ct.c_int*len(D))(*D))
        libGP.set_hyp(self.s,kf.hyp.ctypes.data_as(ct.POINTER(ct.c_double)))
        libGP.build_K(self.s)
        libGP.fac(self.s)
        libGP.presolv(self.s)
        return
    
    def getnext(self,lb,ub,npts):
        xmin = sp.array([42.]*self.D)
        ymin = sp.array([42.])
        libGP.getnext(self.s,lb.ctypes.data_as(ct.POINTER(ct.c_double)),ub.ctypes.data_as(ct.POINTER(ct.c_double)),xmin.ctypes.data_as(ct.POINTER(ct.c_double)),ymin.ctypes.data_as(ct.POINTER(ct.c_double)),ct.c_int(npts))
        return [ymin,xmin]


class GP_EI_direct(GPcore):
    def __init__(self, X_s, Y_s, S_s, D_s, kf):
        
        [self.n ,self.D] = X_s.shape
        self.s = libGP.newEI_direct(ct.c_int(self.D),ct.c_int(self.n),ct.c_int(kf.Kindex))
        self.Y_s=Y_s
        libGP.set_X(self.s,X_s.ctypes.data_as(ct.POINTER(ct.c_double)))
        libGP.set_Y(self.s,Y_s.ctypes.data_as(ct.POINTER(ct.c_double)))
        libGP.set_S(self.s,S_s.ctypes.data_as(ct.POINTER(ct.c_double)))
        D = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D_s]
        
        libGP.set_D(self.s,(ct.c_int*len(D))(*D))
        libGP.set_hyp(self.s,kf.hyp.ctypes.data_as(ct.POINTER(ct.c_double)))
        libGP.build_K(self.s)
        libGP.fac(self.s)
        libGP.presolv(self.s)
        return
    
    def getnext(self,lb,ub,npts):
        xmin = sp.array([42.]*self.D)
        ymin = sp.array([42.])
        libGP.getnext(self.s,lb.ctypes.data_as(ct.POINTER(ct.c_double)),ub.ctypes.data_as(ct.POINTER(ct.c_double)),xmin.ctypes.data_as(ct.POINTER(ct.c_double)),ymin.ctypes.data_as(ct.POINTER(ct.c_double)),ct.c_int(npts))
        return [ymin,xmin]

SQUEXP = 0
LIN1 = 1
LINXSQUEXP = 2
LINSQUEXPXSQUEXP = 3
SQUEXP1SSQUEXP = 4
SSPS = 5
class kernel():
    def __init__(self,K,D,H):
        self.dim = D
        self.hyp = sp.array(H)
        self.Kindex = K
        #ihyp are derived from the hyperparameters for speed and will be 1/h^2 etc.
        self.ihyp = sp.empty(self.hyp.shape[0])
        libGP.hypconvert(self.hyp.ctypes.data_as(ct.POINTER(ct.c_double)),ct.c_int(self.Kindex), ct.c_int(self.dim), self.ihyp.ctypes.data_as(ct.POINTER(ct.c_double)))
        return
    
    def __call__(self,x1, x2, d1=[sp.NaN], d2=[sp.NaN]):
        D1 = 0 if sp.isnan(d1[0]) else int(sum([8**x for x in d1]))
        D2 = 0 if sp.isnan(d2[0]) else int(sum([8**x for x in d2]))
        r=libGP.k(x1.ctypes.data_as(ct.POINTER(ct.c_double)),x2.ctypes.data_as(ct.POINTER(ct.c_double)), ct.c_int(D1),ct.c_int(D2),ct.c_int(self.dim),self.ihyp.ctypes.data_as(ct.POINTER(ct.c_double)),ct.c_int(self.Kindex))
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
        r=libGP.k(x1.ctypes.data_as(ct.POINTER(ct.c_double)),x2.ctypes.data_as(ct.POINTER(ct.c_double)), ct.c_int(D1),ct.c_int(D2),ct.c_int(self.dim),self.hypinv.ctypes.data_as(ct.POINTER(ct.c_double)),ct.c_int(0))
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
        r=libGP.k(x1.ctypes.data_as(ct.POINTER(ct.c_double)),x2.ctypes.data_as(ct.POINTER(ct.c_double)), ct.c_int(D1),ct.c_int(D2),ct.c_int(-42),self.hypinv.ctypes.data_as(ct.POINTER(ct.c_double)),ct.c_int(1))
        return r

def searchMLEhyp(X,Y,S,D,lb,ub, ki, mx=5000,fg=-1e9):
    libGP.SetHypSearchPara(ct.c_int(mx),ct.c_double(fg))
    ns=X.shape[0]
    dim = X.shape[1]
    Dx = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D]
    hy = sp.empty(libGP.numhyp(ct.c_int(ki),ct.c_int(dim)))
    
    lk = sp.empty(1)
    r = libGP.HypSearchMLE(ct.c_int(dim),ct.c_int(len(Dx)),X.ctypes.data_as(ct.POINTER(ct.c_double)),Y.ctypes.data_as(ct.POINTER(ct.c_double)),S.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*len(Dx))(*Dx),lb.ctypes.data_as(ct.POINTER(ct.c_double)),ub.ctypes.data_as(ct.POINTER(ct.c_double)),ct.c_int(ki), hy.ctypes.data_as(ct.POINTER(ct.c_double)),lk.ctypes.data_as(ct.POINTER(ct.c_double)))
    
    return hy


def searchMAPhyp(X,Y,S,D,m,s, ki, MAPmargin = 2.5, mx=5000,fg=-1e9):
    libGP.SetHypSearchPara(ct.c_int(mx),ct.c_double(fg))
    ns=X.shape[0]
    dim = X.shape[1]
    Dx = [0 if sp.isnan(x[0]) else int(sum([8**i for i in x])) for x in D]
    hy = sp.empty(libGP.numhyp(ct.c_int(ki),ct.c_int(dim)))
    
    lk = sp.empty(1)
    r = libGP.HypSearchMAP(ct.c_int(dim),ct.c_int(len(Dx)),X.ctypes.data_as(ct.POINTER(ct.c_double)),Y.ctypes.data_as(ct.POINTER(ct.c_double)),S.ctypes.data_as(ct.POINTER(ct.c_double)),(ct.c_int*len(Dx))(*Dx),m.ctypes.data_as(ct.POINTER(ct.c_double)),s.ctypes.data_as(ct.POINTER(ct.c_double)),ct.c_double(MAPmargin),ct.c_int(ki), hy.ctypes.data_as(ct.POINTER(ct.c_double)),lk.ctypes.data_as(ct.POINTER(ct.c_double)))
    
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