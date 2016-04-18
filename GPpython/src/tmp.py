import wrappingLogistic
import time
ds=1
t0 = time.clock()
f = wrappingLogistic.main({'lrate':0.1,'l2_reg':0.2,'batchsize':800,'n_epochs':2},fold=1,folds=1,downsize=ds)
t1 = time.clock()

print t1-t0