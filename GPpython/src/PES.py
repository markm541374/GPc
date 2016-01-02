# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

#make a single posterior gp form data and take draws on this
import ESutils
import GPdc
import eprop
import scipy as sp


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

def addmins(G,X,Y,S,D,xmin,GRADNOISE=1e-9,EP_SOFTNESS=1e-9,EPROP_LOOPS=20):
    dim=X.shape[1]
    #grad elements are zero
    Xg = sp.vstack([xmin]*dim)
    Yg = sp.zeros([dim,1])
    Sg = sp.ones([dim,1])*GRADNOISE
    Dg = [[i] for i in range(dim)]
    
    #offdiag hessian elements
    
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
    Xo = sp.vstack([X,Xg,Xd])
    Yo = sp.vstack([Y,Yg,Yd])
    So = sp.vstack([S,Sg,Sd])
    Do = D+Dg+Dd
    
    return [Xo,Yo,So,Do]