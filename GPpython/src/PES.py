# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

#make a single posterior gp form data and take draws on this
import ESutils
import GPdc
import eprop
import scipy as sp
from scipy import stats as sps
import DIRECT

def makeG(X,Y,S,D,kindex,mprior,sprior,nh):
    #draw hyps based on plk
    H = ESutils.drawhyp_plk(X,Y,S,D,kindex,mprior,sprior,nh)
    #make a G with draws
    G = GPdc.GPcore(X,Y,S,D,[GPdc.kernel(kindex,X.shape[1],i) for i in H])
    return G

def drawmins(G,n,lb,ub,SUPPORT=300,SLICELCB_PARA=1.):
    #draw support points
    W = ESutils.draw_support(G, lb,ub,SUPPORT,SLICELCB_PARA)
    #draw in samples on the support
    R = ESutils.draw_min(G,W,n)
    return R

def drawmins_inplane(G,n,lb,ub,axis,value, SUPPORT=300,SLICELCB_PARA=1.):
    W = ESutils.draw_support_inplane(G, lb,ub,SUPPORT,SLICELCB_PARA,axis,value)
    #draw in samples on the support
    R = ESutils.draw_min(G,W,n)
    return R

OFFHESSZERO=0
OFFHESSINFER=1

def addmins(G,X,Y,S,D,xmin,mode=OFFHESSZERO, GRADNOISE=1e-9,EP_SOFTNESS=1e-9,EPROP_LOOPS=20):
    dim=X.shape[1]
    #grad elements are zero
    Xg = sp.vstack([xmin]*dim)
    Yg = sp.zeros([dim,1])
    Sg = sp.ones([dim,1])*GRADNOISE
    Dg = [[i] for i in range(dim)]
    
    #offdiag hessian elements
    nh = ((dim-1)*dim)/2
    Xh = sp.vstack([sp.empty([0,dim])]+[xmin]*nh)
    Dh=[]
    for i in range(dim):
        for j in range(i):
            Dh.append([i,j])
    class MJMError(Exception):
        pass
    if mode==OFFHESSZERO:
        Yh = sp.zeros([nh,1])
        Sh = sp.ones([nh,1])*GRADNOISE
    elif mode==OFFHESSINFER:
        raise MJMError("addmins with mode offhessinfer not implemented yet")
    else:
        raise MJMError("invalid mode in addmins")
        
    #diag hessian and min
    Xd = sp.vstack([xmin]*(dim+1))
    Dd = [[sp.NaN]]+[[i,i] for i in xrange(dim)]
    [m,V] = G.infer_full_post(Xd,Dd)
    yminarg = sp.argmin(Y)
    Y_ = sp.array([Y[yminarg,0]]+[0.]*dim)
    Z = sp.array([-1]+[1.]*dim)
    F = sp.array([S[yminarg,0]]+[EP_SOFTNESS]*dim)
    [Yd,Stmp] = eprop.expectation_prop(m,V,Y_,Z,F,EPROP_LOOPS)
    Sd = sp.diagonal(Stmp).flatten()
    Sd.resize([dim+1,1])
    Yd.resize([dim+1,1])
    #concat the obs
    Xo = sp.vstack([X,Xg,Xd,Xh])
    Yo = sp.vstack([Y,Yg,Yd,Yh])
    So = sp.vstack([S,Sg,Sd,Sh])
    Do = D+Dg+Dd+Dh
    
    return [Xo,Yo,So,Do]

NOMIN=0
def addmins_inplane(G,X,Y,S,D,xmin,axis,value,mode=OFFHESSZERO, GRADNOISE=1e-9,EP_SOFTNESS=1e-9,EPROP_LOOPS=20,MINPOLICY=NOMIN):
    dim=X.shape[1]
    #grad elements are zero
    Xg = sp.vstack([xmin]*(dim-1))
    Yg = sp.zeros([dim-1,1])
    Sg = sp.ones([dim-1,1])*GRADNOISE
    Dg = [[i] for i in range(dim) if i!=axis]
    #offdiag hessian elements
    nh = ((dim-1)*(dim-2))/2
    Xh = sp.vstack([sp.empty([0,dim])]+[xmin]*nh)
    Dh=[]
    for i in range(dim):
        for j in range(i):
            if i!=axis and j!=axis:
                Dh.append([i,j])
    class MJMError(Exception):
        pass
    if mode==OFFHESSZERO:
        Yh = sp.zeros([nh,1])
        Sh = sp.ones([nh,1])*GRADNOISE
    elif mode==OFFHESSINFER:
        raise MJMError("addmins with mode offhessinfer not implemented yet")
    else:
        raise MJMError("invalid mode in addmins")
    #diag hessian and min
    if MINPOLICY==NOMIN:
        Xd = sp.vstack([xmin]*(dim-1))
        Dd = [[i,i] for i in xrange(dim) if i !=axis]
        [m,V] = G.infer_full_post(Xd,Dd)
        #yminarg = sp.argmin(Y)
    
        Y_ = sp.array([0.]*dim)#!!!!!!!!!!!!!!!!
        Z = sp.array([1.]*dim)
        F = sp.array([EP_SOFTNESS]*dim)#!!!!!!!!!!!!

        [Yd,Stmp] = eprop.expectation_prop(m,V,Y_,Z,F,EPROP_LOOPS)
        Sd = sp.diagonal(Stmp).flatten()
        Sd.resize([dim-1,1])
        Yd.resize([dim-1,1])
    else:
        print "this isn't a valid approach!!!!!!!!!"
        Xd = sp.vstack([xmin]*(dim-1+1))
        Dd = [[sp.NaN]]+[[i,i] for i in xrange(dim) if i !=axis]
        [m,V] = G.infer_full_post(Xd,Dd)
        yminarg = sp.argmin(Y)
    
        Y_ = sp.array([Y[yminarg,0]]+[0.]*dim)#!!!!!!!!!!!!!!!!
        Z = sp.array([-1]+[1.]*dim)
        F = sp.array([S[yminarg,0]]+[EP_SOFTNESS]*dim)#!!!!!!!!!!!!

        [Yd,Stmp] = eprop.expectation_prop(m,V,Y_,Z,F,EPROP_LOOPS)
        Sd = sp.diagonal(Stmp).flatten()
        Sd.resize([dim+1-1,1])
        Yd.resize([dim+1-1,1])
    #concat the obs
    Xo = sp.vstack([X,Xg,Xd,Xh])
    Yo = sp.vstack([Y,Yg,Yd,Yh])
    So = sp.vstack([S,Sg,Sd,Sh])
    Do = D+Dg+Dd+Dh
    
    return [Xo,Yo,So,Do]

def PESgain(g0,G1,Z,X,D,s):
    
    H = sp.zeros(len(D))
    [m0,v0] = g0.infer_diag_post(X,D)
    
    
    for j in xrange(X.shape[0]):
        H[j]-= len(G1)*0.5*sp.log(2*sp.pi*sp.e*(v0[0,j]+s[j]))
        for i,g1 in enumerate(G1):
            
            Xi = sp.vstack([X[j,:],Z[i,:]])
            Di = [D[j]]+[[sp.NaN]]
            
            [mi,Vi] = g1.infer_full_post(Xi,Di)
            #print [mi,Vi]
            v1 = Vadj(mi,Vi)
        
            h1 = 0.5*sp.log(2*sp.pi*sp.e*(v1+s[j]))
            H[j]+=h1
    
    return -H/float(len(G1))

def Vadj(m,V):
    s = V[0,0]+V[1,1]-2*V[0,1]
    mu = m[0,1]-m[0,0]
    alpha = -mu/sp.sqrt(s) #sign difference for minimizing
    beta = sp.exp(sps.norm.logpdf(alpha) - sps.norm.logcdf(alpha))
    #print [s,mu,alpha,beta]
    vadj = V[0,0]-beta*(beta+alpha)*(1./s)*(V[0, 0]-V[0, 1])**2
    return vadj

class PES:
    def __init__(self,X,Y,S,D,lb,ub,kindex,mprior,sprior,DH_SAMPLES=8,DM_SAMPLES=8, DM_SUPPORT=400,DM_SLICELCBPARA=1.):
        self.lb=lb
        self.ub=ub
        self.G = makeG(X,Y,S,D,kindex,mprior,sprior,DH_SAMPLES)
        self.Z = drawmins(self.G,DM_SAMPLES,lb,ub,SUPPORT=DM_SUPPORT,SLICELCB_PARA=DM_SLICELCBPARA)
        self.Ga = [GPdc.GPcore(*addmins(self.G,X,Y,S,D,self.Z[i,:])+[self.G.kf]) for i in xrange(DM_SAMPLES)]
        
    def query_pes(self,Xq,Sq,Dq):
        a = PESgain(self.G,self.Ga,self.Z,Xq,Dq,Sq)
        return a
    
    def query_acq(self,Xq,Sq,Dq,costfn):
        a = PESgain(self.G,self.Ga,self.Z,Xq,Dq,Sq)
        for i in xrange(Xq.shape[0]):
            a[i] = a[i]/costfn(Xq[i,:].flatten(),Sq[i,:].flatten())
        return a
    
    def search_pes(self,s,maxf=2500,dv=[[sp.NaN]]):
        def directwrap(Q,extra):
            x = sp.array([Q])
            acq = PESgain(self.G,self.Ga,self.Z,x,dv,[s])
            R = -acq
            return (R,0)
        
        [xmin, ymin, ierror] = DIRECT.solve(directwrap,self.lb,self.ub,user_data=[], algmethod=1, maxf=maxf, logfilename='/dev/null')
        return [xmin,ymin,ierror]
    
    def search_acq(self,cfn,logsl,logsu,maxf=2500,dv=[[sp.NaN]]):
        def directwrap(Q,extra):
            x = sp.array([Q[:-1]])
            s = 10**Q[-1]
            acq = PESgain(self.G,self.Ga,self.Z,x,dv,[s])
            R = -acq/cfn(x,s)
            return (R,0)
        
        [xmin, ymin, ierror] = DIRECT.solve(directwrap,sp.hstack([self.lb,logsl]),sp.hstack([self.ub,logsu]),user_data=[], algmethod=1, maxf=maxf, logfilename='/dev/null')
        return [xmin,ymin,ierror]
    
class PES_inplane:
    def __init__(self,X,Y,S,D,lb,ub,kindex,mprior,sprior,axis,value,DH_SAMPLES=8,DM_SAMPLES=8, DM_SUPPORT=400,DM_SLICELCBPARA=1.,AM_POLICY=NOMIN):
        self.lb=lb
        self.ub=ub
        self.G = makeG(X,Y,S,D,kindex,mprior,sprior,DH_SAMPLES)
        self.Z = drawmins_inplane(self.G,DM_SAMPLES,lb,ub,axis=axis,value=value,SUPPORT=DM_SUPPORT,SLICELCB_PARA=DM_SLICELCBPARA)
        self.Ga = [GPdc.GPcore(*addmins_inplane(self.G,X,Y,S,D,self.Z[i,:],axis=axis,value=value,MINPOLICY=AM_POLICY)+[self.G.kf]) for i in xrange(DM_SAMPLES)]
        
    def query_pes(self,Xq,Sq,Dq):
        a = PESgain(self.G,self.Ga,self.Z,Xq,Dq,Sq)
        return a
    
    def query_acq(self,Xq,Sq,Dq,costfn):
        a = PESgain(self.G,self.Ga,self.Z,Xq,Dq,Sq)
        for i in xrange(Xq.shape[0]):
            a[i] = a[i]/costfn(Xq[i,:].flatten())
        return a
    
    def search_acq(self,cfn,sfn,maxf=2500,dv=[[sp.NaN]]):
        def directwrap(Q,extra):
            x = sp.array([Q])
            s = sfn(x)
            acq = PESgain(self.G,self.Ga,self.Z,x,dv,[s])
            R = -acq/cfn(x)
            return (R,0)
        
        [xmin, ymin, ierror] = DIRECT.solve(directwrap,self.lb,self.ub,user_data=[], algmethod=1, maxf=maxf, logfilename='/dev/null')
        return [xmin,ymin,ierror]