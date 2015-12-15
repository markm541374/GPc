/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "GPsimple.h"
#include <cmath>
#include <vector>

extern "C" int LCB(GP* g, int n, double* X, int* D, double p, double* R){
    std::vector<double> U = std::vector<double>(2*n);
    g->infer_diag(n,X,D,&U[0]);
    for (int i=0; i<n; i++){
        R[i] = U[i] - p*sqrt(U[i+n]);
    }
    return 0;
}