<h1>GPshared</h1>

<p>This project is a c/c++ shared library and python wrapper to be used for Gaussian Process Global Optimisation. The GPpython folder contains a separate python project for Bayesian Optimization</p>
<p>The cpp library and python wrapper implement Gaussian processes with observations and inference up to second derivatives in multiple axes, and posterior inference over sets of equally weighted hyperparameter draws</p>
<h3>Files:</h3>
<p>
    In c lib
<UL>
    <LI> libGP.cpp: main file for cpp library, maintains a vector of GPs and handles calls on them, also a function to return log likelihood only
    <LI> GPsimple.cpp: GP implementation, methods for infering mean only, diagonal variance or full covariance
    <LI> direct.cpp: direct from external source
    <LI> bayesutils.cpp: EI, LCB and logEI functions
    <LI> hypsearch.cpp: fn to run direct to find MLE or MAP hyperparameters given data (and prior for MAP)
    <LI> kernel.cpp: has or imports all the kernel functions, taking numnbers correspoinding to derivatives in each axis of each x1 and x2, converters for hyperparameters from  natural (length) to hte form used in kernel (1/l**2) and to log space for searching
    <LI> matern.cpp: the matern 5.2 kernel
    <LI> misctool.cpp: normal pdf and cdf, EI and logEI draw samples from a covariance matrix or cholesky decomp.
    <LI> simplekernels.cpp: some sums and productts of other kernels, derivatives not implemented.
   </UL>
    In python lib
<UL>
    <LI>   GPdc.py: imports the c library and provides interface
    <LI>   slice.py: slice samples adapted from 3rd party
    <LI>   search.py: wrapper for search algorithms, sets up parameters and runs the search
    <LI>   eprop.py: expectation propagation for hard and soft inequalities on multivatiate normals
    <LI>   optutils.py: selection of synthetic objectives and implementation of all the searches, mostly inheriting from a base
    <LI>   ESutils.py: draws support points and mins from support given a GP, draws hyperparameters from data given prior
    <LI>   PES.py: implementation of PESaquisition function, regular and env variable versions.
    <LI>   test* : tests for all the above
    <LI>   exp*  : experiments according to comments in header
    </UL>
    
    </p>

<h3>Using External Code:</h3>
<p>DIRECT implementation<BR> Jones, D.R., Perttunen, C.D., Stuckman, B.E.: Lipschitzian optimization without 
!    the Lipschitz constant. J. Optim. Theory Appl. 79(1), 157–181 (1993)</p>