# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import GPdc
import slice
import scipy as sp

SUPPORT_UNIFORM = 0
SUPPORT_SLICELCB = 1

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
    elif method==SUPPORT_SLICELCB:
        def f(x):
            if all(x>lb) and all(x<ub):
                return -g.infer_LCB(sp.array(x),[[sp.NaN]],para)[0,0]
            else:
                return -1e99
        print "Drawing support using slice sample over LCB:"
        X = slice.slice_sample(f,0.5*(ub+lb),n,0.05*(ub-lb))
    else:
        raise RuntimeError("draw_support method invalid")
    return X


#return the min loc of draws on given support
def draw_min(g,support,n):
    Z = g.draw(support, [[sp.NaN]]*support.shape[0],n)
    
    R = sp.empty([n,support.shape[1]])
    for i in xrange(n):
        R[i,:] = support[sp.argmin(Z[:,i]),:]
    return R

#ub and lb are still for the full space but the values in the chosen axis do not determine the outcome
def draw_support_inplane(g,lb,ub,n,method,axis,value,para=1.):
    lb[axis]=value-1.
    ub[axis]=value+1.
    d = g.D
    if method==SUPPORT_UNIFORM:
        #still draws in the full space then overwrites in the fixed axis
        print "Drawing support using uniform:"
        X=sp.random.uniform(size=[n,g.D])
        for i in xrange(g.D):
            X[:,i] *= ub[i]-lb[i]
            X[:,i] += lb[i]
        X[:,axis] *= 0.
        X[:,axis] += value
        return X
    elif method==SUPPORT_SLICELCB:
        def f(x):
            y = sp.hstack([x[:axis],value,x[axis:]])
            if all(y>lb) and all(y<ub):
                return -g.infer_LCB(sp.array(y),[[sp.NaN]],para)[0,0]
            else:
                return -1e99
        print "Drawing support using slice sample over LCB:"
        lb_red = sp.hstack([lb[:axis],lb[axis+1:]])
        ub_red = sp.hstack([ub[:axis],ub[axis+1:]])
        X = slice.slice_sample(f,0.5*(ub_red+lb_red),n,0.05*(ub_red-lb_red))
        return sp.hstack([X[:,:axis],sp.ones([n,1])*value,X[:,axis:]])
    else:
        raise RuntimeError("draw_support method invalid")
    return -1

def plot_gp(g,axis,x,d):
    [m,v] = g.infer_diag(x,d)
    s = sp.sqrt(v)
    axis.fill_between(x,sp.array(m-2.*s).flatten(),sp.array(m+2.*s).flatten(),facecolor='lightblue',edgecolor='lightblue')
    axis.plot(x,m.flatten(),'b')
    return 0

def drawhyp_plk(X,Y,S,D,ki,hm,hs,n,burn=80,subsam=5):
    def f(loghyp):
        r=GPdc.GP_LKonly(X,Y,S,D,GPdc.kernel(ki,X.shape[1],[10**i for i in loghyp])).plk(hm,hs)
        
        return r
    X = slice.slice_sample(f,hm,n,0.05*hs,burn=burn,subsam=subsam)
    return 10**X