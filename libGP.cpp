/*
 * libtest.cpp
 *
 *  Created on: 3 Oct 2015
 *      Author: mark
 */

#include <vector>
#include <cmath>
#include <cblas.h>
#include <lapacke.h>
#include <stdio.h>
#include <stdlib.h>



#ifdef __cplusplus
extern "C"
{
#endif

#include "direct.h"
#include "kernel.h"
#include "hypsearch.h"
#include "misctools.h"
#include "GPsimple.h"

#include "bayesutils.h"
    
std::vector<GP*> SS;

int newGP(int D, int N, int kindex, double* X, double* Y, double* Sx, int* Dx, double* h){
	GP *p = new GP(D,N,kindex);
	SS.push_back(p);
        int k = SS.size()-1;
        SS[k]->set_X(X);
        SS[k]->set_Y(Y);
        SS[k]->set_S(Sx);
        SS[k]->set_D(Dx);
        SS[k]->set_hyp(h);
	return SS.size()-1;
}

int newGP_LKonly(int D, int N, double* Xin, double* Yin, double* Sin, int* Din, int kindex, double* hyp, double* R){
	GP_LKonly *p =  new GP_LKonly(D,N, Xin, Yin, Sin, Din, kindex, hyp, R);
	p->~GP_LKonly();
	return 0;
}


void killGP(int k){
	SS[k]->~GP();
	SS[k] = 0;
}
int ping(int k){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
	SS[k]->ping();
	return 0;
}

int set_Data(int k, double* X, double* Y, double* Sx, int* D){
    if (SS[k]==0){
	printf("trying to use deleted GP\n");
	return -1;
    };
    SS[k]->set_X(X);
    SS[k]->set_Y(Y);
    SS[k]->set_S(Sx);
    SS[k]->set_D(D);
    return 0;
}

int set_hyp(int k, double* h){
    if (SS[k]==0){
	printf("trying to use deleted GP\n");
    return -1;
    };

    SS[k]->set_hyp(h);
    return 0;
}

int presolv(int k){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
	return SS[k]->presolv();
}
int infer_diag(int k,int Ns,double* Xs, int* Ds, double* R){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
	return SS[k]->infer_diag(Ns, Xs,Ds,R);
}
int infer_m(int k,int Ns,double* Xs, int* Ds, double* R){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
	return SS[k]->infer_m(Ns, Xs,Ds,R);
}
int llk(int k, double* R){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
	return SS[k]->llk(R);
}
int infer_full(int k,int Ns,double* Xs, int* Ds, double* R){

	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
	return SS[k]->infer_full(Ns, Xs,Ds,R);
}
int draw(int k, int Nd, double* X, int* D, double* R, int m){
    if (SS[k]==0){
	printf("trying to use deleted GP\n");
	return -1;
    };
    return SS[k]->draw(Nd, X, D, R, m);
}

int infer_LCB(int k, int n, double* X, int* D, double p, double* R){
    if (SS[k]==0){
	printf("trying to use deleted GP\n");
	return -1;
    };
    return LCB(SS[k], n, X, D, p, R);
}

int infer_EI(int k, int n, double* X, int* D, double* R){
    if (SS[k]==0){
	printf("trying to use deleted GP\n");
	return -1;
    };
    return EI_gp(SS[k], n, X, D, R);
}
#ifdef __cplusplus
}
#endif