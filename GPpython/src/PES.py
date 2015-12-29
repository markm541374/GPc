# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

#make a single posterior gp form data and take draws on this
NUMINPLANESUPPORT = 300
SLICELCB_PARA = 1.
def method1(G,mprior,sprior):
    # this assumes
    dim = X.shape[1]
    Z = ESutils.draw_support_inplane(G, dim, mprior,sprior,NUMINPLANESUPPORT,ESutils.SUPPORT_SLICELCB,dim-1,0., para=SLICELCB_PARA)
    M = ESutils.draw_min(G,Z,100)
    return

