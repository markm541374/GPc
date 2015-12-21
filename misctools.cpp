/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <stdio.h>
#include <cmath>
#include <cblas.h>
#include <random>
const double PI = 3.141592653589793238463;
const double SQRT_1_2PI = 1/sqrt(2*PI);
const double SQRT_1_2 =  1/sqrt(2);


double pdf(double z){
    //if (-0.5*pow(z,2)<-704.8){printf(" pdf underflow");}
    return SQRT_1_2PI*exp(-0.5*pow(z,2));
}

double cdf(double z){
    //if (-SQRT_1_2*z>26.55){printf(" cdf underflow");}
    return 0.5 * erfc(-SQRT_1_2*z);
}

extern "C" int EI(double* m, double* s, double y, int N, double* R){
    double S;
    
    for (int i=0; i<N; i++){
        S = (y-m[i])/s[i];
        R[i] = (y-m[i])*cdf(S)+s[i]*pdf(S);
        //TODO fix EI for values that make pdf and cdf zero
        //if (R[i]==0.){printf("ei=0 %e %e %e_",S,pdf(S),cdf(S));}
    }
    return 0;
}

//make m draws from a multivariate Gaussian of size n where the cholesky deom of the covariance is the lower triangular matrix K
//matrix mult means the draws are down the columns of R
extern "C" int drawcov(double* K, int n, double* R, int m){
    std::random_device rand_dev;          // Use random_device to get a random seed.
    std::mt19937 rand_engine(rand_dev()); //seed merseene twister with random number
    std::normal_distribution<double> nmdist(0., 1.); //normal unit dist
    for (int i=0;i<n*m;i++){
        R[i] = nmdist(rand_engine);
        
    }
    cblas_dtrmm(CblasRowMajor,CblasLeft,CblasLower,CblasNoTrans,CblasNonUnit,n,m,1.,K,n,R,m);
    return 0;
    
}