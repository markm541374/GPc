# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import pickle
import scipy as sp
import os
import time
import logging
import copy


logger = logging.getLogger(__name__)

class optstate:
    def __init__(self):
        self.x = []
        self.ev = []
        self.y = []
        self.c = []
        self.C = 0
        self.n = 0
        return
    
    def update(self,x,ev,y,c):
        #print 'updata {}'.format(self.ev)
        self.x.append(x)
        self.ev.append(copy.copy(ev))
        self.y.append(y)
        self.c.append(c)
        self.C +=c
        self.n+=1
        return 


class optimizer:
    def __init__(self,dirpath,aqpara,aqfn,stoppara,stopfn,reccpara,reccfn,ojf,ojfchar,checkrecc=False):
        print aqpara
        self.checkrecc=checkrecc
        self.dirpath = dirpath
        self.setaq(aqpara,aqfn)
        self.setstopcon(stoppara,stopfn)
        self.setojf(ojf)
        self.setrecc(reccpara,reccfn)
        
        self.ojfchar = ojfchar
        self.dx = ojfchar['dx']
        self.dev = ojfchar['dev']
        return
    
    def setaq(self,aqpara,aqfn):
        self.aqfn = aqfn
        self.aqpara = aqpara
        self.aqpersist = None
        return
    
    def setrecc(self,reccpara,reccfn):
        self.reccpara = reccpara
        self.reccfn = reccfn
        return
    
    def setstopcon(self,stoppara,stopfn):
        self.stoppara = stoppara
        self.stopfn=stopfn
        return
    
    def setojf(self,ojf):
        self.ojf = ojf
        return
    
    def run(self):
        logger.info('startopt:')
        
        lf = open(os.path.join(self.dirpath,'trace.csv'),'wb',0)
        lf.write(''.join(['n, ']+['x'+str(i)+', ' for i in xrange(self.dx)]+[i+', ' for i in self.aqpara['ev'].keys()]+['y, c, ']+['rx'+str(i)+', ' for i in xrange(self.dx)]+['truey at xrecc, taq, tev, trc, realtime'])+'\n')
        self.state = optstate()
        stepn=0
        while not self.stopfn(self.state,**self.stoppara):
            stepn+=1
            logger.info("---------------------\nstep {}\naquisition:".format(stepn))
            
            t0 = time.time()
            x,ev,self.aqpersist,aqaux = self.aqfn(self.state,self.aqpersist,**self.aqpara)
            t1 = time.time()
            logger.info("{} : {}    aqtime: {}\nevaluate:".format(x,ev,t1-t0))
            
            y,c,ojaux  = self.ojf(x,**ev)
            t2 = time.time()
            self.state.update(x,ev,y,c)
            logger.info("{} : {}     evaltime: {}\nreccomend:".format(y,c,t2-t1))
            rx,reaux = self.reccfn(self.state,**self.reccpara)
            t3 = time.time()
            
            if self.checkrecc:
                logger.info("checkin {} : {}".format(rx,self.aqpara['ev']))
                checky,checkc,checkojaux  = self.ojf(rx,**self.aqpara['ev'])
                logger.info("checkout {} : {} : {}".format(checky,checkc,checkojaux))
            else:
                checky=sp.NaN

            
            logger.info("{}     recctime: {}\n".format(rx,t3-t2))
            
            logstr = ''.join([str(stepn)+', ']+[str(xi)+', ' for xi in x]+[str(evi[1])+', ' for evi in ev.items()]+[str(y)+', ']+[str(c)+', ']+[str(ri)+', ' for ri in rx]+[str(checky)+',']+[str(i)+', ' for i in [t1-t0,t2-t1,t3-t2]]+[time.strftime('%H:%M:%S  %d-%m-%y')])+'\n'
            lf.write(logstr)
            
            pobj = [x,ev,self.aqpersist,aqaux,y,c,ojaux,rx,reaux,t1-t0,t2-t1,t3-t2,time.strftime('%H:%M:%S  %d-%m-%y')]
            
            pickle.dump(pobj,open(os.path.join(self.dirpath,'step{}.p'.format(stepn)),'wb'))
        pickle.dump(self.state,open(os.path.join(self.dirpath,'state.p'),'wb'))
        logger.info('endopt')
        return
    

def nstopfn(optstate,nmax = 1):
    return optstate.n >= nmax

def cstopfn(optstate,cmax = 1):
    return optstate.C >= cmax







    