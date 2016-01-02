# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

#make a single posterior gp form data and take draws on this
import ESutils
import GPdc
"""
NUMINPLANESUPPORT = 300
SLICELCB_PARA = 1.
def method1(G):
    # this assumes
    dim = X.shape[1]
    lb=0
    ub=0
    Z = ESutils.draw_support_inplane(G, lb,ub,NUMINPLANESUPPORT,ESutils.SUPPORT_SLICELCB,dim-1,0., para=SLICELCB_PARA)
    M = ESutils.draw_min(G,Z,100)
    return
"""

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