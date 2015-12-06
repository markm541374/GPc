# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from scipy import stats as sps
from scipy import linalg as spl
import scipy as sp
from matplotlib import pyplot as plt

import GPdc

hyp = [1.,0.4,0.1]
kern = GPdc.gen_lin1_k_d(hyp)

X = sp.matrix([0.,0.2,0.4,0.6,0.8]).T
Y = sp.matrix([-0.2,-0.1,0.,0.15,0.3]).T

D = [[sp.NaN]]*(X.shape[0])
S = sp.matrix([1e-2]*X.shape[0]).T
f0 = plt.figure()
a0 = plt.subplot(111)
a0.plot(sp.array(X).flatten(),Y,'g.')

G = GPdc.GPcore(X,Y,S,D,GPdc.gen_lin1_k_d(hyp))

np=180
sup = sp.linspace(0,1,np)
Dp = [[sp.NaN]]*np
Xp = sp.matrix(sup).T

[m,v] = G.infer_diag(Xp,Dp)
a0.plot(sup,m)
sq = sp.sqrt(v)

a0.fill_between(sup, sp.array(m-1.*sq).flatten(), sp.array(m+1.*sq).flatten(), facecolor='lightblue',edgecolor='lightblue')
plt.show()