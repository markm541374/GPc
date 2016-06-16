# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

#random search on a test function
import acquisitions
import reccomenders
import objectives
import optimize

import scipy as sp
import os

path = os.path.join(os.path.expanduser('~'),'Dropbox/workspace/GPshared/results/gridsh')
aqfn,aqpara = acquisitions.brute
aqpara['lb']=[-1.,-1.]
aqpara['ub']=[1.,1.]

stoppara= {'nmax':8000}
stopfn = optimize.nstopfn

reccfn,reccpara = reccomenders.argmin
reccpara['lb']=aqpara['lb']
reccpara['ub']=aqpara['ub']
reccpara['everyn']=4
reccpara['onlyafter']=0

cfn = objectives.cf42

ojf = objectives.rosenojf
ojfchar = {'dx':len(aqpara['lb']),'dev':len(aqpara['ev'])}