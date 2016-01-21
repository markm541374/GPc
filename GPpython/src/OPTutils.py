# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import scipy as sp
from scipy import linalg as spl
import time
import GPdc
from matplotlib import pyplot as plt
import DIRECT
import PES
import ESutils
def cosines(x,s,d):
    x.resize([1,x.size])
    assert(d==[sp.NaN])
    
    u = 1.6*x[0,0]-0.5
    v = 1.6*x[0,1]-0.5
    f = 1.-(u**2 + v**2 -0.3*sp.cos(3*sp.pi*u)-0.3*sp.cos(3*sp.pi*v)+0.7)
    
    if s==0.:
        noise = 0.
    else:
        noise = sp.random.normal(scale=sp.sqrt(s))
    return [f +noise,1.]

def quad(x,s,d):
    assert(d==[sp.NaN])
    f = sum((x.flatten()-0.1)**2)
    if s==0.:
        noise = 0.
    else:
        noise = sp.random.normal(scale=sp.sqrt(s))
    return [f +noise,1.]
bananamin = sp.array([0.2,0.2])
def banana(x,s,d):
    assert(d==[sp.NaN])
    x.resize([1,x.size])
    u = 5.*x[0,0]
    v = 5.*x[0,1]
    a=1.
    b=100.
    f = 1e-3*((a-u)**2 + b*(v-u**2)**2)
    if s==0.:
        noise = 0.
    else:
        noise = sp.random.normal(scale=sp.sqrt(s))
    return [f+noise,1.]


def gensquexpdraw(d,lb,ub,ignores=-1):
    nt=14
    [X,Y,S,D] = ESutils.gen_dataset(nt,d,lb,ub,GPdc.SQUEXP,sp.array([1.5]+[0.30]*d))
    G = GPdc.GPcore(X,Y,S,D,GPdc.kernel(GPdc.SQUEXP,d,sp.array([1.5]+[0.30]*d)))
    def obj(x,s,d):
        #print [x,s,d]
        if ignores>0:
            s=ignores
        if s==0.:
            noise = 0.
        else:
            noise = sp.random.normal(scale=sp.sqrt(s))
        return [G.infer_m(x,[d])[0,0]+noise,1.]
    def dirwrap(x,y):
        z = obj(x,0.,[sp.NaN])
        return (z,0)
    [xmin,ymin,ierror] = DIRECT.solve(dirwrap,lb,ub,user_data=[], algmethod=1, maxf=10000, logfilename='/dev/null')
    
    return [obj,xmin]

def gensquexpIPdraw(d,lb,ub,sl,su,sfn,sls):
    #axis = 0 value = sl
    #d dimensional objective +1 for s
    nt=25
    print sp.hstack([sp.array([[sl]]),lb])
    print sp.hstack([sp.array([[su]]),ub])
    [X,Y,S,D] = ESutils.gen_dataset(nt,d+1,sp.hstack([sp.array([[sl]]),lb]).flatten(),sp.hstack([sp.array([[su]]),ub]).flatten(),GPdc.SQUEXP,sp.array([1.5]+[sls]+[0.30]*d))
    G = GPdc.GPcore(X,Y,S,D,GPdc.kernel(GPdc.SQUEXP,d+1,sp.array([1.5]+[sls]+[0.30]*d)))
    def obj(x,s,d):
        x = x.flatten()
        if sfn(x)==0.:
            noise = 0.
        else:
            noise = sp.random.normal(scale=sp.sqrt(sfn(x)))
        
        return [G.infer_m(x,[d])[0,0]+noise,1.]
    def dirwrap(x,y):
        z = obj(sp.array([[sl]+[i for i in x]]),sl,[sp.NaN])
        return (z,0)
    [xmin,ymin,ierror] = DIRECT.solve(dirwrap,lb,ub,user_data=[], algmethod=1, maxf=10000, logfilename='/dev/null')
    
    return [obj,xmin]




class opt(object):
    def __init__(self,objective,lb,ub,para=None):
        self.ojf = objective
        self.lb = lb
        self.ub = ub
        self.d = lb.shape[1]
        try:
            self.d = para['d']
        except:
            pass
        self.X = sp.empty([0,self.d])
        self.Y = sp.empty([0,1])
        self.S = sp.empty([0,1])
        self.D = []
        
        self.R = sp.empty([0,self.d])
        self.C = []
        self.T = []
        self.Tr = []
        self.Ymin = []
        self.sdefault = 1e-6
        self.init_search(para)
        return
    
    def init_search(self,para):
        self.method = "Default: RandomSearch"
        return
    
    def run_search(self):
        return self.run_search_random()
    
    def run_search_random(self):
        xnext = sp.random.uniform(size=self.d)*(self.ub-self.lb)+self.lb
        snext = self.sdefault
        dnext = [sp.NaN]
        return [xnext,snext,dnext]
    
    def reccomend(self):
        return self.reccomend_random()
    
    def reccomend_random(self):
        i = sp.argmin(self.Y)
        return self.X[i,:]
    
    def query_ojf(self,x,s,d):
        return self.ojf(x,s,d)
    
    def step(self,random=False):
        t0=time.time()
        if random:
            [x,s,d] = self.run_search_random()
            print "random selected: "+str([x,s,d])
        else:
            [x,s,d] = self.run_search()
            print "search found: "+str([x,s,d])
        t1=time.time()
        if self.X.shape[0]==0:
            xr = (self.lb+self.ub)/2.
        elif random:
            xr = self.reccomend_random()
        else:
            xr = self.reccomend()
        print "reccomend point "+str(xr)
        t2=time.time()
        
        [y,c] = self.query_ojf(x,s,d)
        self.X = sp.vstack([self.X,x])
        self.Y = sp.vstack([self.Y,y])
        self.S = sp.vstack([self.S,s])
        self.D.append(d)
        
        self.R = sp.vstack([self.R,xr])
        self.Tr.append(t2-t1)
        self.C.append(c)
        self.T.append(t1-t0)
        self.Ymin.append(sp.amin(self.Y))
        return

    def compX(self,xtrue):
        n = self.X.shape[0]
        R = sp.empty([2,n])
        for i in xrange(n):
            R[0,i] = spl.norm(self.X[i,:]-xtrue)
            R[1,i] = sp.amin(R[0,:i+1])
        return R
    def plot(self,truex,ax,c):
        ax[0].plot(self.Ymin,c)
        ax[0].set_ylabel('Ymin')
        n=self.X.shape[0]
        M = sp.empty(n)
        V = sp.empty(n)
        for i in xrange(n):
            M[i] = spl.norm(self.X[i,:]-truex)
            V[i] = spl.norm(self.R[i,:]-truex)
        ax[1].semilogy(M,c,label=str(type(self)))
        ax[1].set_ylabel('Xeval')
        ax[1].legend(loc='upper center',ncol=2).draggable()
        ax[2].semilogy(V,c)
        ax[2].set_ylabel('Xrecc')
        
        ax[3].plot([sum(self.C[:i]) for i in xrange(n)],c)
        ax[3].set_ylabel('cost')
        
        ax[4].semilogy([sum(self.C[:i]) for i in xrange(n)],V,c)
        ax[4].set_ylabel('Xrecc/cost')
        
        ax[5].plot(self.T,c)
        ax[5].set_ylabel('Stime')
        ax[6].plot(self.Tr,c)
        ax[6].set_ylabel('Rtime')
        return
    
class LCBMLE(opt):
    def init_search(self,para):
        self.kindex = para[0]
        self.mprior = para[1]
        self.sprior = para[2]
        self.maxf = para[3]
        self.s = para[4]
        self.sdefault = para[4]
        ninit=para[5]
        for i in xrange(ninit):
            self.step(random=True)
        return
    
    def run_search(self):
        
        MAP = GPdc.searchMAPhyp(self.X,self.Y,self.S,self.D,self.mprior,self.sprior, self.kindex)
        self.G = GPdc.GPcore(self.X,self.Y,self.S,self.D,GPdc.kernel(self.kindex,self.d,MAP))
        def directwrap(x,y):
            x.resize([1,self.d])
            
            a = self.G.infer_LCB(x,[[sp.NaN]],1.)[0,0]
            return (a,0)
        [xmin,ymin,ierror] = DIRECT.solve(directwrap,self.lb,self.ub,user_data=[], algmethod=1, maxf=self.maxf, logfilename='/dev/null')
        return [xmin,self.s,[sp.NaN]]
    def reccomend(self):
        def dirwrap(x,y):
            m  =self.G.infer_m(x,[[sp.NaN]])[0,0]
            return (m,0)
        [xmin,ymin,ierror] = DIRECT.solve(dirwrap,self.lb,self.ub,user_data=[], algmethod=1, maxf=self.maxf, logfilename='/dev/null')
        return xmin
    
class EIMLE(opt):
    def init_search(self,para):
        self.kindex = para[0]
        self.mprior = para[1]
        self.sprior = para[2]
        self.maxf = para[3]
        self.s = para[4]
        self.sdefault = para[4]
        ninit=para[5]
        for i in xrange(ninit):
            self.step(random=True)
        return
    
    def run_search(self):
        
        MAP = GPdc.searchMAPhyp(self.X,self.Y,self.S,self.D,self.mprior,self.sprior, self.kindex)
        self.G = GPdc.GPcore(self.X,self.Y,self.S,self.D,GPdc.kernel(self.kindex,self.d,MAP))
        def directwrap(x,y):
            x.resize([1,self.d])
            
            a = self.G.infer_EI(x,[[sp.NaN]])
            #print [x,a]
            #print G.infer_diag_post(x,[[sp.NaN]])
            return (-a[0,0],0)
        [xmin,ymin,ierror] = DIRECT.solve(directwrap,self.lb,self.ub,user_data=[], algmethod=1, maxf=self.maxf, logfilename='/dev/null')
        return [xmin,self.s,[sp.NaN]]
    
    def reccomend(self):
        def dirwrap(x,y):
            m  =self.G.infer_m(x,[[sp.NaN]])[0,0]
            return (m,0)
        [xmin,ymin,ierror] = DIRECT.solve(dirwrap,self.lb,self.ub,user_data=[], algmethod=1, maxf=self.maxf, logfilename='/dev/null')
        return xmin
    
class PESFS(opt):
    def init_search(self,para):
        self.para=para
        self.sdefault = para['s']
        
        for i in xrange(para['ninit']):
            self.step(random=True)
        return
    
    def run_search(self):
        print "begin PESFS:"
        self.pesobj = PES.PES(self.X,self.Y,self.S,self.D,self.lb.flatten(),self.ub.flatten(),self.para['kindex'],self.para['mprior'],self.para['sprior'],DH_SAMPLES=self.para['DH_SAMPLES'], DM_SAMPLES=self.para['DM_SAMPLES'], DM_SUPPORT=self.para['DM_SUPPORT'],DM_SLICELCBPARA=self.para['DM_SLICELCBPARA'],mode=self.para['SUPPORT_MODE'])
        [xmin,ymin,ierror] = self.pesobj.search_pes(self.sdefault,maxf=self.para['maxf'])
        return [xmin,self.para['s'],[sp.NaN]]
    
    def reccomend(self):
        def dirwrap(x,y):
            m  =self.pesobj.G.infer_m(x,[[sp.NaN]])[0,0]
            return (m,0)
        [xmin,ymin,ierror] = DIRECT.solve(dirwrap,self.lb,self.ub,user_data=[], algmethod=1, maxf=self.para['maxf'], logfilename='/dev/null')
        return xmin

class PESIS(PESFS):
    def init_search(self,para):
        self.para=para
        self.sdefault = -1
        for i in xrange(para['ninit']):
            self.step(random=True)
        return
    
    def run_search(self):
        print "begin PESIS:"
        self.pesobj = PES.PES(self.X,self.Y,self.S,self.D,self.lb.flatten(),self.ub.flatten(),self.para['kindex'],self.para['mprior'],self.para['sprior'],DH_SAMPLES=self.para['DH_SAMPLES'], DM_SAMPLES=self.para['DM_SAMPLES'], DM_SUPPORT=self.para['DM_SUPPORT'],DM_SLICELCBPARA=self.para['DM_SLICELCBPARA'],mode=self.para['SUPPORT_MODE'],noS=True)
        [xmin,ymin,ierror] = self.pesobj.search_pes(-1,maxf=self.para['maxf'])
        return [xmin,0.,[sp.NaN]]


class PESVS(opt):
    def init_search(self,para):
        self.para=para
        self.sdefault = para['s']
        
        for i in xrange(para['ninit']):
            self.step(random=True)
        return
    
    def run_search(self):
        print "begin PES:"
        self.pesobj = PES.PES(self.X,self.Y,self.S,self.D,self.lb.flatten(),self.ub.flatten(),self.para['kindex'],self.para['mprior'],self.para['sprior'],DH_SAMPLES=self.para['DH_SAMPLES'], DM_SAMPLES=self.para['DM_SAMPLES'], DM_SUPPORT=self.para['DM_SUPPORT'],DM_SLICELCBPARA=self.para['DM_SLICELCBPARA'],mode=self.para['SUPPORT_MODE'])
        [Qmin,ymin,ierror] = self.pesobj.search_acq(self.para['cfn'],self.para['logsl'],self.para['logsu'],maxf=self.para['maxf'])
        return [Qmin[:-1],10**Qmin[-1],[sp.NaN]]
    
    def reccomend(self):
        def dirwrap(x,y):
            m  =self.pesobj.G.infer_m(x,[[sp.NaN]])[0,0]
            return (m,0)
        [xmin,ymin,ierror] = DIRECT.solve(dirwrap,self.lb,self.ub,user_data=[], algmethod=1, maxf=self.para['maxf'], logfilename='/dev/null')
        return xmin
    
    def query_ojf(self,x,s,d):
        [y,c0] = self.ojf(x,s,d)
        c = self.para['cfn'](x,s)
        return [y,c]


class PESIP(opt):
    def init_search(self,para):
        self.para=para
        self.sdefault = para['s']
        self.lb = sp.hstack([sp.array([[para['sl']]]),self.lb])
        self.ub = sp.hstack([sp.array([[para['su']]]),self.ub])
        print self.lb
        print self.ub
        for i in xrange(para['ninit']):
            self.step(random=True)
        return
    
    def run_search(self):
        print "begin PES:"
        
        self.pesobj = PES.PES_inplane(self.X,self.Y,self.S,self.D,self.lb.flatten(),self.ub.flatten(),self.para['kindex'],self.para['mprior'],self.para['sprior'],self.para['axis'],self.para['value'],DH_SAMPLES=self.para['DH_SAMPLES'], DM_SAMPLES=self.para['DM_SAMPLES'], DM_SUPPORT=self.para['DM_SUPPORT'],DM_SLICELCBPARA=self.para['DM_SLICELCBPARA'],mode=self.para['SUPPORT_MODE'])
        
        [Qmin,ymin,ierror] = self.pesobj.search_acq(self.para['cfn'],self.para['sfn'],maxf=self.para['maxf'])
        return [Qmin,self.para['sfn'](Qmin),[sp.NaN]]
    
    def reccomend(self):
        def dirwrap(x,y):
            m  =self.pesobj.G.infer_m(sp.hstack([self.para['sl'],x]),[[sp.NaN]])[0,0]
            return (m,0)
        [xmin,ymin,ierror] = DIRECT.solve(dirwrap,self.lb[:,1:],self.ub[:,1:],user_data=[], algmethod=1, maxf=self.para['maxf'], logfilename='/dev/null')
        return sp.hstack([sp.array(self.para['sl']),xmin])
    def reccomend_random(self):
        i = sp.argmin(self.Y)
        
        return sp.hstack([sp.array([self.para['sl']]),self.X[i,1:]])
    
    def query_ojf(self,x,s,d):
        [y,c0] = self.ojf(x,s,d)
        c = self.para['cfn'](x,s)
        return [y,c]

    def plotstate(self,a):
        self.pesobj.query_acq(x,1e-8,[sp.NaN])
        