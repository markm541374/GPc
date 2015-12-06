/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <stdio.h>
#include <cmath>

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
