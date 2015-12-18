# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import scipy as sp
from scipy import linalg as spl
from scipy import stats as sps

def PhiR(x):
    # return sp.exp(sps.norm.logpdf(x) - sps.norm.logcdf(x))
    return sps.norm.pdf(x)/sps.norm.cdf(x)

def expectation_prop(m0,V0,Y,Z,F,z):
    #expectation propagation on multivariate gaussian for soft inequality constraint
    #m0,v0 are mean vector , covariance before EP
    #Y is inequality value, Z is sign, 1 for geq, -1 for leq, F is softness ??
    #z is number of ep rounds to run
    #returns mt, Vt the value and variance for observations created by ep
    n = V0.shape[0]
    mt =sp.zeros(n)
    Vt= sp.eye(n)*1e-2
    m = sp.empty(n)
    V = sp.empty([n,n])
    for i in xrange(z):
        #compute the m V give ep obs
        V = V0.dot(spl.solve(V0+Vt,Vt))
        m = V.dot((spl.solve(V0,m0)+spl.solve(Vt,mt)).T)
        for j in xrange(n):
            #the cavity dist at index j
            tmp = 1./(Vt[j,j]-V[j,j])
            v_ = (V[j,j]*Vt[j,j])*tmp
            m_ = tmp*(m[j]*Vt[j, j]-mt[j]*V[j, j])
            
            alpha = (m_-Y[j]) / (sp.sqrt(v_+F[j]**2))
            pr = PhiR(alpha)
            if sp.isnan(pr):
                pr = -alpha
            beta = pr*(pr+alpha)/(v_+F[j]**2)
            kappa = (pr+alpha) / (sp.sqrt(v_+F[j]**2))
            #print [alpha,beta,kappa,pr]
            mt[j] = m_+1./kappa
            Vt[j,j] = 1./beta - v_
    V = V0.dot(spl.solve(V0+Vt,Vt))
    m = V.dot((spl.solve(V0,m0)+spl.solve(Vt,mt)).T)
    return m, V
"""
def runEP(self, plot='none'):
        g = GPd.GPcore(self.X_c, self.Y_c, self.S_c, self.D_c, self.kf)
        # start by making the full inference at the inequality locations
        [m0, V0] = g.infer_full(self.X_z, self.D_z)
        V0Im0 = spl.cho_solve(spl.cho_factor(V0), m0)
        V0I = V0.I  # ------------------------------explicit inverse, not good
        # create the ep observations
        yt = sp.matrix(sp.zeros([self.n_z, 1]))
        St = 10**10*sp.matrix(sp.ones([self.n_z, 1]))

        ytR = yt.copy()
        StR = St.copy()
        m_R = [[]]*self.n_z
        v_R = [[]]*self.n_z
        for it in xrange(self.nloops):
            for i in xrange(self.n_z):
                # update the inference at z with the ep observations
                StIyt = sp.matrix(sp.zeros([self.n_z, 1]))
                StI = sp.matrix(sp.zeros([self.n_z, self.n_z]))
                VtIV0I = V0I.copy()
                # print V0I
                for j in xrange(self.n_z):
                    StIyt[j, 0] = yt[j, 0]/float(St[j, 0])
                    VtIV0I[j, j] += 1./float(St[j, 0])

                Vp = VtIV0I.I.copy()
                mp = Vp*(StIyt+V0Im0)
                # get the inference at the specific inequality currently being updates
                v_ = 1/(1/(Vp[i, i]) - 1/(St[i, 0]))
                m_ = v_*(mp[i, 0]/Vp[i, i]-yt[i, 0]/St[i, 0])

                m_R[i].append(m_)
                v_R[i].append(v_)
                # find the new ep obs

                alpha = (m_+((-1)**self.G_z[i])*self.I_z[0,i]) / (sp.sqrt(v_+self.N_z[i]))

                pr = PhiR(alpha)
                if sp.isnan(pr):
                    pr = -alpha
                beta = pr*(pr+alpha)/(v_+self.N_z[i]**2)
                kappa = ((-1)**self.G_z[i])*(pr+alpha) / (sp.sqrt(v_+self.N_z[i]**2))

                # replace with the new ep obs
                yt[i, 0] = m_-1./kappa
                St[i, 0] = 1./beta - v_
                if (1./beta) == sp.inf:
                    yt[i, 0] = m_
                    St[i, 0] = 10**100

            ytR = sp.hstack([ytR, yt])
            StR = sp.hstack([StR, St])
        Y_z = yt.copy()
        
        S_z = St.copy()

        [Xep, Yep, Sep, Dep] = GPd.catObs([[self.X_c, self.Y_c, self.S_c, self.D_c], [self.X_z, Y_z, S_z, self.D_z]])
        self.gep = GPd.GPcore(Xep, Yep, Sep, Dep, self.kf)

        self.invalidflag = False
        return
"""