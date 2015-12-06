/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "GPsimple.h"
#include <chrono>
#include <stdio.h>
#include <vector>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

class GP_timing : public GP{
    public:
        using GP::GP;
        int timing(int, double*);
};

int GP_timing::timing(int c,double* T){
    printf("timing %d",c);
    std::vector<double> R = std::vector<double>(2*c);
    std::vector<int> Ds = std::vector<int>(c,0);
    std::vector<double> x = std::vector<double>(c*this->D,0);
    for (int i=1;i<c;i++){
        for (int j=0;j<100;j++){
            auto tstart=Clock::now();
            infer_diag(i,&x[0],&Ds[0],&R[0]);
            auto tend=Clock::now();
            int stime = std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();
            //printf("inferdiag evaluated %d points in %d nanoseconds\n",i,stime);
        T[i] += stime;
        }
        T[i] = T[i]/100.;
    }
    for (int i=1;i<100;i++){
        auto tstart=Clock::now();
        infer_diag(1,&x[0],&Ds[0],&R[0]);
        auto tend=Clock::now();
        int stime = std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();
        //printf("inferdiag evaluated %d points in %d nanoseconds\n",i,stime);
        T[0] += stime;
    }
    
    
    
    return 0;
}