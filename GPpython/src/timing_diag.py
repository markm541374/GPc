# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "mark"
__date__ = "$02-Dec-2015 21:11:58$"

from scipy import stats as sps
from scipy import linalg as spl
import scipy as sp
from matplotlib import pyplot as plt

import GPdc
D = 2
nt = 22
X = sp.random.uniform(-1,1,size=[nt,D])
Y = sp.random.normal(size=[nt,1])
Dx = [[sp.NaN]]*nt
S = sp.array([[0.0001]]*nt)

MLEH = sp.array([1.]+[0.1]*D)

g = GPdc.GP_timing(X,Y,S,Dx,GPdc.gen_sqexp_k_d(MLEH))
T = g.timing(100)
f0 = plt.figure()
plt.plot(range(1,100),T[1:])
plt.plot([1,100],[T[0]/100.,T[0]])
#plt.axis([0,100,0,100000])

f0.savefig("../figs/timingdiag.png")
plt.show()
