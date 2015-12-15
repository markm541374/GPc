# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import GPdc
import slice
import scipy as sp

SUPPORT_UNIFORM = 0
SUPPORT_SLICELCB = 1
def draw_support(g, lb, ub, n, method, para=1.):
    #para is the std confidence bound
    d = g.D
    if method==SUPPORT_UNIFORM:
        
        X=sp.random.uniform(size=[n,g.D])
        for i in xrange(g.D):
            X[:,i] *= ub[i]-lb[i]
            X[:,i] += lb[i]
    elif method==SUPPORT_SLICELCB:
        pass
    else:
        raise RuntimeError("draw_support method invalid")
    return X
    