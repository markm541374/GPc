/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "GPsimple.h"
#include "misctools.h"
#include <stdio.h>
#include "direct.h"
#include <random>

#include <chrono>
#include <boost/random.hpp>

typedef std::chrono::high_resolution_clock Clock;

class EI_random : public GP{
    public:
        using GP::GP;
        int getnext(double* lb, double* ub, double* argmin, double* min, int npts);
        double acq(double* x);
        double ymin;
};
int EI_random::getnext(double* lb, double* ub, double* argmin, double* min, int npts){
    auto tstart=Clock::now();
    ymin=10e10;
    for (int i; i<this->N; i++){
        if (this->Y[i]<ymin){
            ymin = this->Y[i];
        }
    }
    int block = 8000;
    min[0]=-1.;
    double maxEI=0.;
    std::vector<double> R = std::vector<double>(2*block);
    std::vector<int> Ds = std::vector<int>(block,0);
    std::vector<double> ei = std::vector<double>(block);
    std::vector<double> x = std::vector<double>(block*D);

    std::uniform_real_distribution<double> unif(-1., 1.);

    std::random_device rand_dev;          // Use random_device to get a random seed.

    std::mt19937 rand_engine(12345);
    int nthis;
    
    for (int i=0; i<npts;i+=block){
        
        if (block<npts-i){
            nthis=block;
            
        }
        else{
            nthis = npts-i;
        }
       
        //printf("%d",nthis);
        for (int j=0;j<D*nthis;j++){
            x[j] = unif(rand_engine);
            
            
        }  
        
        infer_diag(nthis,&x[0],&Ds[0],&R[0]);
        for (int j=0; j<nthis; j++){
            R[nthis+j]=sqrt(R[nthis+j]);
        }
        
        EI(&R[0], &R[nthis], ymin, nthis, &ei[0]);
        //printf("%f ",min[0]);
        for (int j=0; j<nthis; j++){
            //printf("%f_",ei[j]);
            if (ei[j]>min[0]){
                //printf("%f ",ei[j]);
                min[0]=ei[j];
                for (int k=0;k<D;k++){argmin[k]=x[D*j+k];}
                                      
            }
        }
    }
    auto tend=Clock::now();
    int stime = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
    printf("random evaluated %d points in %d microseconds\n",npts,stime);
    
    return 0;
}

double EI_random::acq(double* x){
    
    std::vector<double> R = std::vector<double>(2);
    std::vector<int> Ds = std::vector<int>(2);
    Ds[0]=0;Ds[1]=0;
    
    infer_diag(1,x,&Ds[0],&R[0]);
    
    double ei=-1.;
    R[1] = sqrt(R[1]);
    EI(&R[0], &R[1], ymin, 1, &ei);
    //printf("\n[%f %f %f %f]",ei,R[0],R[1],ymin);
    return ei;
}


class EI_direct : public GP{
    public:
        using GP::GP;
        //EIdirect(int d, int n, int kindex) : GP(int d, int n, int kindex){}
        int getnext(double* lb, double* ub, double* argmin, double* min, int npts);
        double acq(double* x);
        double ymin;
};

EI_direct* currentGP;
double directwrap(int dim, double* X){
    //printf("[%d_%f_%f]",dim,X[0],X[1]);
    double ei = currentGP->acq(X);
    return -ei;
}


double EI_direct::acq(double* x){
    
    std::vector<double> R = std::vector<double>(2);
    std::vector<int> Ds = std::vector<int>(2);
    Ds[0]=0;Ds[1]=0;
    
    infer_diag(1,x,&Ds[0],&R[0]);
    
    double ei=-1.;
    R[1] = sqrt(R[1]);
    EI(&R[0], &R[1], ymin, 1, &ei);
    //printf("\n[%f %f %f %f]",ei,R[0],R[1],ymin);
    return ei;
}
int EI_direct::getnext(double* lb, double* ub, double* argmin, double* min, int npts){
    auto tstart=Clock::now();
    currentGP = this;
    ymin=10e10;
    for (int i; i<this->N; i++){
        if (this->Y[i]<ymin){
            ymin = this->Y[i];
        }
    }
    
    direct(D, lb, ub, npts, -1e8, argmin, min, directwrap);
    min[0]= -min[0];
    auto tend=Clock::now();
    int stime = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
    printf("direct evaluated %d points in %d microseconds\n",npts,stime);
    return 0;
}



