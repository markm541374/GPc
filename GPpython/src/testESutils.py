# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import ESutils
import scipy as sp
from matplotlib import pyplot as plt

class bob:
    def __init__(self,D):
        self.D=D
        return


X = ESutils.draw_support(bob(2), sp.array([-2,-1]),sp.array([0,3]),500,ESutils.SUPPORT_UNIFORM)
for i in xrange(X.shape[0]):
    plt.plot(X[i,0],X[i,1],'r.')
plt.axis([-5,5,-5,5])
plt.show()