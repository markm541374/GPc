# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.


import ctypes as ct
import scipy as sp
from scipy import linalg as spl
from matplotlib import pyplot as plt
libGP = ct.cdll.LoadLibrary('../../dist/Release/GNU-Linux/libGPshared.so')

ctpd = ct.POINTER(ct.c_double)

import GPdc

ni = 50
kf = GPdc.kernel(GPdc.SQUEXP,1,sp.array([1.3,0.3]))
X = sp.matrix(sp.linspace(-1,1,ni)).T
D = [[sp.NaN]]*ni
Kxx = GPdc.buildKsym_d(kf,X,D)

tmp = spl.cholesky(Kxx,lower=True)
Ch = sp.vstack([tmp[i,:] for i in xrange(ni)]) #hack/force row major storage


z = 5

b = sp.empty([ni,z])


libGP.drawcov(Ch.ctypes.data_as(ctpd),ct.c_int(ni),b.ctypes.data_as(ctpd),ct.c_int(z))

#print b
for i in xrange(z):
    plt.plot(X[:,0],b[:,i])

plt.show()