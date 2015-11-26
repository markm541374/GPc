/*
 * hypsearch.cpp
 *
 *  Created on: 24 Oct 2015
 *      Author: mark
 */

#include <vector>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include "kernel.h"
#include "direct.h"
#include <cblas.h>
#include <lapacke.h>

const double PI = 3.141592653589793238463;
const double L2PI = log(PI*2);

//GP variables
int D;
int N;
int ki;
std::vector<double> ih;

std::vector<double> X;
std::vector<double> Y;
std::vector<int> Dx;
std::vector<double> S;

std::vector<double>  Kxx;
std::vector<double> Yd;

//default direct hyperparameters

int maxint = 5000;
double fglob = -1e8;

//set direct hyperparameters
extern "C" void SetHypSearchPara(int mx, double fg){
	maxint = mx;
	fglob = fg;
	return;
}
//llk function to call direct from
double llk(int directsearchdim, double* hyp){

	ih[0] = pow(10.,2.*hyp[0]);
	for (int i=1; i<D+1; i++){
		ih[i] = 1./pow(10.,2.*hyp[i]);
	}
	//buildK
	for (int i=0; i<N; i++){
		Kxx[i*N+i] = kern[ki](&X[i*D], &X[i*D],Dx[i],Dx[i],D,&ih[0]);
		for (int j=0; j<i; j++){
			Kxx[i*N+j] = Kxx[i+N*j] = kern[ki](&X[i*D], &X[j*D],Dx[i],Dx[j],D,&ih[0]);
		}
		Kxx[i*N+i]+=S[i];

	}

	//cho factor
	LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',N,&Kxx[0],N);
	for (int i=0; i<N; i++){
		Yd[i]= Y[i];
	}

	//solve against Y
	LAPACKE_dpotrs(LAPACK_ROW_MAJOR,'L',N,1,&Kxx[0],N,&Yd[0],1);
	//calc the llk
	double R;
	R = - 0.5*N*L2PI;
	for (int i=0; i<N; i++){
		R-=log(Kxx[i*(N+1)]);
	}
	R -= 0.5*cblas_ddot(N,&Y[0],1,&Yd[0],1);

	return -R;
}

extern "C" int HypSearchMLE(int d, int n, double* Xin, double* Yin, double* Sin, int* Din, double* lb, double* ub, int kernelindex, double* Rhyp, double* lk){

	ki = kernelindex;
	N = n;
	D = d;
	ih.resize(D+1);
	X.resize(N*D);
	Y.resize(N);
	S.resize(N);
	Dx.resize(N);
	Kxx.resize(N*N);
	Yd.resize(N);

	for (int i=0; i<N*D; i++){
		X[i] = Xin[i];
	}
	for (int i=0; i<N; i++){
		Y[i] = Yd[i]= Yin[i];
		S[i] = Sin[i];
		Dx[i] = Din[i];
	}
	std::vector<double> xbest = std::vector<double>(D+1,0.);

	direct(D+1,&lb[0],&ub[0],maxint,fglob,&xbest[0],lk,llk);

	for (int i=0; i<D+1; i++){
		Rhyp[i] = pow(10.,xbest[i]);
	}
	return 0;
}