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

int newGP_hypset(int D, int N, int kindex, double* X, double* Y, double* Sx, int* Dx, double* h, int s){
    int base = SS.size();
    for (int i=0; i<s; i++){
        newGP(D,N,kindex,X,Y,Sx,Dx,&h[i*numhyp(kindex,D)]);
    }
    return base;
}

void killGP(int k){
	SS[k]->~GP();
	SS[k] = 0;
}
int ping(int k, int s){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
        for (int i=0; i<s; i++){
            SS[k+i]->ping();
        }
	
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

int presolv(int k, int s){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
        for (int i=0; i<s; i++){
            SS[k+i]->presolv();
        }
	return 0;
}
int infer_diag(int k, int s, int Ns,double* Xs, int* Ds, double* R){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
        for (int i=0; i<s; i++){
            SS[k+i]->infer_diag(Ns, Xs,Ds,&R[2*i*Ns]);
        }
	return 0;
}
int infer_m(int k, int s, int Ns,double* Xs, int* Ds, double* R){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
        for (int i=0; i<s; i++){
            SS[k+i]->infer_m(Ns, Xs,Ds,&R[Ns*i]);
        }
	return 0;
}
int llk(int k, double* R){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
	return SS[k]->llk(R);
}
int infer_full(int k, int s, int Ns,double* Xs, int* Ds, double* R){

	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
        for (int i=0; i<s; i++){
            SS[k+i]->infer_full(Ns, Xs,Ds,&R[Ns*(Ns+1)*i]);
        }
	return 0;
}
int draw(int k, int s, int Nd, double* X, int* D, double* R, int m){
    if (SS[k]==0){
	printf("trying to use deleted GP\n");
	return -1;
    };
    for (int i=0; i<s; i++){
        SS[k+i]->draw(Nd, X, D, &R[Nd*m*i], m);
    }
    return 0;
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