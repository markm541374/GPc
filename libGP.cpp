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
#include "GPbayesopt.h"
#include "timing.h"
    
std::vector<GP*> SS;

int newGP(int D, int N, int kindex){
	GP *p = new GP(D,N,kindex);
	SS.push_back(p);

	return SS.size()-1;
}

int newGP_timing(int D, int N, int kindex){
	GP *p = new GP_timing(D,N,kindex);
	SS.push_back(p);

	return SS.size()-1;
}

int newEI_direct(int D, int N, int kindex){
	GP *p = new EI_direct(D,N,kindex);
	SS.push_back(p);

	return SS.size()-1;
}
int newEI_random(int D, int N, int kindex){
	GP *p = new EI_random(D,N,kindex);
	SS.push_back(p);

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

int set_Y(int k, double* Y){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};

	SS[k]->set_Y(Y);
	return 0;
}
int set_S(int k, double* Sin){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};

	SS[k]->set_S(Sin);
	return 0;
}
int set_D(int k, int* D){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};

	SS[k]->set_D(D);
	return 0;
}
int set_X(int k, double* X){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};

	SS[k]->set_X(X);
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
int fac(int k){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
	return SS[k]->fac();
}
int build_K(int k){
	if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
	return SS[k]->build_K();
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

int getnext(int k,double* lb, double* ub, double* argmin, double* min, int npts){
    if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
	return SS[k]->getnext(lb,ub,argmin,min, npts);
}

int timing(int k,int c, double* T){
    if (SS[k]==0){
		printf("trying to use deleted GP\n");
		return -1;
	};
	return SS[k]->timing(c, T);
}
#ifdef __cplusplus
}
#endif