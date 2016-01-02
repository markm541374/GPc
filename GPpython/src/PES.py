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

def addmins(X,Y,S,D,kfs,xmin,GRADNOISE=1e-9):
    dim=X.shape[1]
    #grad elements are zero
    Xg = sp.vstack([xmin]*dim)
    Yg = sp.zeros([dim,1])
    Sg = sp.ones([dim,1])*GRADNOISE
    Dg = [[i] for i in range(dim)]
    
    #offdiag hessian elements
    
    #diag hessian and min
    
    #concat the obs
    Xo = sp.vstack([X,Xg])
    Yo = sp.vstack([Y,Yg])
    So = sp.vstack([S,Sg])
    Do = D+Dg
    
    
    return [Xo,Yo,So,Do]