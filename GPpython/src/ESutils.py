# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import GPdc
import slice
import scipy as sp
from scipy import linalg as spl
from scipy import stats as sps
from matplotlib import pyplot as plt
from scipy.optimize import minimize as spomin
from scipy.stats import multivariate_normal as mnv


SUPPORT_UNIFORM = 0
SUPPORT_SLICELCB = 1
SUPPORT_SLICEEI = 2
SUPPORT_SLICEPM = 3
SUPPORT_LAPAPR = 4
#drawing points between lb and ub using specified method
def draw_support(g, lb, ub, n, method, para=1.):
    #para is the std confidence bound
    if (type(g) is int):
        d=g
    else:
        d=g.D
    if method==SUPPORT_UNIFORM:
        print "Drawing support using uniform:"
        X=sp.random.uniform(size=[n,d])
        for i in xrange(d):
            X[:,i] *= ub[i]-lb[i]
            X[:,i] += lb[i]
    elif method==SUPPORT_LAPAPR:
        print "Drawing support using lapapr:"
        #start with 4 times as many points as needed
        #print 'a'
        para = int(para)
        over = 4
        Xsto=sp.random.uniform(size=[over*para,d])
        for i in xrange(d):
            Xsto[:,i] *= ub[i]-lb[i]
            Xsto[:,i] += lb[i]
        #eval mean at the points
        #print 'b'
        fs = sp.empty(para*over)
        for i in xrange(para*over):
            fs[i] = g.infer_m_post(Xsto[i,:],[[sp.NaN]])[0,0]
        Xst = sp.empty([2*para,d])
        #keep the lowest
        #print 'c'
        for i in xrange(para):
            j = fs.argmin()
            Xst[i,:] = Xsto[j,:]
            fs[j]=1e99
        #minimize the posterior mean from each start
        #print 'd'
        def f(x):
            y= g.infer_m_post(sp.array(x),[[sp.NaN]])[0,0]
            bound=0
            r = max(abs(x))
            if r>1:
                bound=1e3*(r-1)
            return y+bound
        for i in xrange(para):
            res = spomin(f,Xst[i,:],method='Nelder-Mead',options={'xtol':0.0001})
            if not res.success:
                class MJMError(Exception):
                    pass
                print res.message
                raise MJMError('failed in opt in support lapapr')
            Xst[i+int(para),:] = res.x
            
        
        
        #find endpoints that are unique
        #print 'e'
        Xst
        unq = [Xst[0+para,:]]
        for i in xrange(para):
            tmp=[]
            for xm in unq:
                tmp.append(abs((xm-Xst[i+para,:])).max())
            
            if min(tmp)>0.0002:
                unq.append(Xst[i+para,:])
        #get the alligned gaussian approx of pmin
        #print 'f'
        print unq
        cls = []
        for xm in unq:
            ls = []
            for i in xrange(d):
                
                vg = g.infer_diag_post(xm,[[i]])[1][0,0]
                gg = g.infer_diag_post(xm,[[i,i]])[0][0,0]
                ls.append(sp.sqrt(vg)/gg)
                
            cls.append(ls)
        #print 'g'
        X=mnv.rvs(size=n,mean=[0.]*d)
        if d==1:
            X.resize([n,1])
        neach = int(n/len(unq))
        for i in xrange(len(unq)):
            
            for j in xrange(d):
                X[i*neach:(i+1)*neach,j]*=cls[i][j]
                X[i*neach:(i+1)*neach,j]+=unq[i][j]
            
        
        if False:
            print 'inits'
            print Xst
            
            print 'cls'
            print cls
            plt.figure()
            np = para
            for i in xrange(np):
                plt.plot([Xst[i,0],Xst[i+np,0]],[Xst[i,1],Xst[i+np,1]],'b.-')
            for j in xrange(len(unq)):
                x = unq[j]
                plt.plot(x[0],x[1],'ro')
                xp = [x[0]+cls[j][0],x[0],x[0]-cls[j][0],x[0],x[0]+cls[j][0]]
                yp = [x[1],x[1]+cls[j][1],x[1],x[1]-cls[j][1],x[1]]
                plt.plot(xp,yp,'r-')
            plt.show()
    elif method==SUPPORT_SLICELCB:
        def f(x):
            if all(x>lb) and all(x<ub):
                try:
                    return -g.infer_LCB_post(sp.array(x),[[sp.NaN]],para)[0,0]
                except:
                    g.infer_LCB_post(sp.array(x),[[sp.NaN]],para)[0,0]
                    g.printc()
                    raise
            else:
                return -1e99
        print "Drawing support using slice sample over LCB:"
        X = slice.slice_sample(f,0.5*(ub+lb),n,0.1*(ub-lb))
    
    elif method==SUPPORT_SLICEEI:
        def f(x):
            if all(x>lb) and all(x<ub):
                try:
                    ei=g.infer_EI(sp.array(x),[[sp.NaN]])[0,0]
                    #print ei
                    return sp.log(ei)
                except:
                    #ei=g.infer_EI(sp.array(x),[[sp.NaN]])[0,0]
                    g.printc()
                    raise
            else:
                return -1e99
        print "Drawing support using slice sample over EI:"
        X = slice.slice_sample(f,0.5*(ub+lb),n,0.1*(ub-lb))
    
    elif method==SUPPORT_SLICEPM:
        def f(x):
            if all(x>lb) and all(x<ub):
                [m,v] = g.infer_diag_post(sp.vstack([sp.array(x)]*d),[[i] for i in xrange(d)])
                p = 0.
                for i in xrange(d):
                    p+= -0.5*(m[0,i]**2)/v[0,i]
                ym = g.infer_m_post(sp.array(x),[[sp.NaN]])[0,0]
                if not sp.isfinite(p):
                    print [m,V,p]
                    #raise ValueError
                return -10*ym+0.01*p
            else:
                return -1e99
        if False:
            A = sp.empty([100,100])
            sup = sp.linspace(-0.999,0.999,100)
            for i in xrange(100):
                for j in xrange(100):
                    print sp.array([sup[i],sup[j]])
                    A[99-j,i] = f([sup[i],sup[j]])
                    print A[99-j,i]
            print A
            plt.figure()
            plt.imshow(A)
            plt.figure()
        print "Drawing support using slice sample over PM:"
        X = slice.slice_sample(f,0.5*(ub+lb),n,0.1*(ub-lb))
    else:
        raise RuntimeError("draw_support method invalid")
    return X


#return the min loc of draws on given support
def draw_min(g,support,n):
    Z = g.draw_post(support, [[sp.NaN]]*support.shape[0],n)
    
    R = sp.empty([Z.shape[0],support.shape[1]])
    args = []
    for i in xrange(Z.shape[0]):
        a = sp.argmin(Z[i,:])
        args.append(a)
        R[i,:] = support[a,:]
    from itertools import groupby
    amins = [len(list(group)) for key, group in groupby(sorted(args))]
    print "In drawmin with {} support drew {} unique mins. Most freqent min chosen {}%".format(support.shape[0],len(amins),100.*max(amins)/float(n))
    return R

#fake gp class that 9looks like a d-1 gp becuase an extra vaue is added before callind
class gpfake():
    def __init__(self,g,axis,value):
        self.g=g
        self.D = g.D-1
        self.axis=axis
        self.value=value
        return
    
    def augx(self,x):
        return sp.hstack([x[:self.axis],sp.array([self.value]*x.shape[0]).T,x[self.axis:]])
    
    def infer_m_post(self,x,d):
        return self.g.infer_m_post(self.augx(x),d)
    
    def infer_diag_post(self,x,d):
        return self.g.infer_diag_post(self.augx(x),d)
    
    def infer_EI(self,x,d):
        return self.g.infer_EI(self.augx(x),d)
    
    def infer_LCB_post(self,x,d,p):
        return self.g.infer_LCB_post(self.augx(x),d,p)
    
#ub and lb are still for the full space but the values in the chosen axis do not determine the outcome
def draw_support_inplane(g,lb,ub,n,method,axis,value,para=1.):
    if (type(g) is int):
        gf = g-1
    else:
        gf = gpfake(g,axis,value)
    lb_red = sp.hstack([lb[:axis],lb[axis+1:]])
    ub_red = sp.hstack([ub[:axis],ub[axis+1:]])
    X = draw_support(gf,lb_red,ub_red,n,method,para=para)
    return sp.hstack([X[:,:axis],sp.ones([n,1])*value,X[:,axis:]])
    

def plot_gp(g,axis,x,d):
    [m,v] = g.infer_diag(x,d)
    s = sp.sqrt(v)
    axis.fill_between(x,sp.array(m-2.*s).flatten(),sp.array(m+2.*s).flatten(),facecolor='lightblue',edgecolor='lightblue')
    axis.plot(x,m.flatten(),'b')
    return 0

#draw hyperparameters given data from posterior likelihood
def drawhyp_plk(X,Y,S,D,ki,hm,hs,n,burn=80,subsam=5):
    def f(loghyp):
        ub = hm+1.8*hs
        lb = hm-1.8*hs
        if all(loghyp<ub) and all(loghyp>lb):
            r=GPdc.GP_LKonly(X,Y,S,D,GPdc.kernel(ki,X.shape[1],[10**i for i in loghyp])).plk(hm,hs)
            if sp.isnan(r):
                class MJMError(Exception):
                    pass
                print 'nan from GPLKonly with input'
                print [X,Y,S,D,ki,hm,hs,n,burn,subsam]
                raise MJMError('nan from GPLKonly with input')
        else:
            r=-1e99
        #print [loghyp, r]
        return r
    X = slice.slice_sample(f,hm,n,0.05*hs,burn=burn,subsam=subsam)
    return 10**X

#take a random draw of X points and draw Y from the specified kernel
def gen_dataset(nt,d,lb,ub,kindex,hyp,s=1e-9):
    X = draw_support(d, lb,ub,nt,SUPPORT_UNIFORM)
    D = [[sp.NaN]]*(nt)
    kf = GPdc.kernel(kindex,d,hyp)
    Kxx = GPdc.buildKsym_d(kf,X,D)
    Y = spl.cholesky(Kxx,lower=True)*sp.matrix(sps.norm.rvs(0,1.,nt)).T+sp.matrix(sps.norm.rvs(0,sp.sqrt(s),nt)).T
    S = sp.matrix([s]*nt).T
    return [X,Y,S,D]