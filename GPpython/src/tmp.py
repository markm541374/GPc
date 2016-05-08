import wrappingLogistic
import time
from matplotlib import pyplot as plt
import scipy as sp

F=[]
T=[]
for i in sp.linspace(0.01,1,50):
    print "subsample: {}".format(i)
    ds=i
    t0 = time.clock()
    f = wrappingLogistic.main({'lrate':0.1,'l2_reg':0.2,'batchsize':800,'n_epochs':80},fold=1,folds=1,downsize=ds)
    t1 = time.clock()
    F.append(f)
    T.append(t1-t0)
print F
print T
f,a=plt.subplots(3)
a[0].plot(F)
a[1].plot(T)
plt.show()