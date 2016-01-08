# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import scipy as sp
import time
import GPdc
from matplotlib import pyplot as plt
import DIRECT
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


class opt:
    def __init__(self,objective,lb,ub,para=None):
        self.ojf = objective
        self.lb = lb
        self.ub = ub
        self.d = lb.shape[1]
        
        self.X = sp.empty([0,self.d])
        self.Y = sp.empty([0,1])
        self.S = sp.empty([0,1])
        self.D = []
        
        self.C = []
        self.T = []
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
    
    def step(self,random=False):
        t0=time.time()
        if random:
            [x,s,d] = self.run_search_random()
        else:
            [x,s,d] = self.run_search()
            print "search found: "+str([x,s,d])
        t1=time.time()
        
        [y,c] = self.ojf(x,s,d)
        self.X = sp.vstack([self.X,x])
        self.Y = sp.vstack([self.Y,y])
        self.S = sp.vstack([self.S,s])
        self.D.append(d)
        
        self.C.append(c)
        self.T.append(t1-t0)
        self.Ymin.append(sp.amin(self.Y))
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
        G = GPdc.GPcore(self.X,self.Y,self.S,self.D,GPdc.kernel(self.kindex,self.d,MAP))
        def directwrap(x,y):
            x.resize([1,self.d])
            
            a = G.infer_LCB(x,[[sp.NaN]],1.)[0,0]
            return (a,0)
        [xmin,ymin,ierror] = DIRECT.solve(directwrap,self.lb,self.ub,user_data=[], algmethod=1, maxf=self.maxf, logfilename='/dev/null')
        return [xmin,self.s,[sp.NaN]]