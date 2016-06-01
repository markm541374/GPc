#!/usr/bin/env python2
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.


import optimize
import acquisitions
import reccomenders
import scipy as sp
import os
import logging
logging.basicConfig(level=logging.DEBUG)


try:
    os.mkdir('test')
except OSError:
    pass
path = os.path.join(os.getcwd(),'test')

aqfn,aqpara = acquisitions.PESfs
aqpara['lb']=[-1.,-1.]
aqpara['ub']=[1.,1.]



stoppara= {'nmax':20}
stopfn = optimize.nstopfn


reccfn,reccpara = reccomenders.gpmap
reccpara['lb']=aqpara['lb']
reccpara['ub']=aqpara['ub']

ojfn = optimize.trivialojf
ojfchar = {'dx':len(aqpara['lb']),'dev':len(aqpara['ev'])}

O = optimize.optimizer(path,aqpara,aqfn,stoppara,stopfn,reccpara,reccfn,ojfn,ojfchar)

O.run()

