/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <vector>
#include <cmath>
#include <cblas.h>
#include <lapacke.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernel.h"

static const double PI = 3.141592653589793238463;
static const double L2PI = log(PI*2);


class GP_LKonly{
public:
	int D;
	int N;
	int K;
	//double lk;
        GP_LKonly(int d, int n, double* Xin, double* Yin, double* Sin, int* Din, int kindex, double* hyp, double* R);
	
	
};
GP_LKonly::GP_LKonly(int d, int n, double* Xin, double* Yin, double* Sin, int* Din, int kindex, double* hyp, double* R){
		D=d;
		N=n;
		K=kindex;
		//lk = 0.;
		//hyp process
		std::vector<double> ih = std::vector<double>(D+1);
		ih[0] = pow(hyp[0],2);
		for (int i=1; i<D+1; i++){
			ih[i] = 1./pow(hyp[i],2);
		}
		//buildK
		std::vector<double>Kxx = std::vector<double>(N*N);
		for (int i=0; i<N; i++){
			Kxx[i*N+i] = kern[K](&Xin[i*D], &Xin[i*D],Din[i],Din[i],D,&ih[0]);
			for (int j=0; j<i; j++){
				Kxx[i*N+j] = Kxx[i+N*j] = kern[K](&Xin[i*D], &Xin[j*D],Din[i],Din[j],D,&ih[0]);
				//if (i<10){printf("%f %f %d %d %d %f %f %f %f _ ",Xin[i*D], Xin[j*D],Din[i],Din[j],D,ih[0],ih[1],ih[2],k(&Xin[i*D], &Xin[j*D],Din[i],Din[j],D,&ih[0]));}
			}
			Kxx[i*N+i]+=Sin[i];

		}
		//cho factor
		LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',N,&Kxx[0],N);
		std::vector<double>Yd = std::vector<double>(N);
		for (int i=0; i<N; i++){
				Yd[i]= Yin[i];
		}
		//solve agains Y
		LAPACKE_dpotrs(LAPACK_ROW_MAJOR,'L',N,1,&Kxx[0],N,&Yd[0],1);
		//calc the llk

		R[0] = - 0.5*N*L2PI;
		//printf("\n1 %f\n",R[0]);
		for (int i=0; i<N; i++){
			R[0]-=log(Kxx[i*(N+1)]);
		}
		//printf("2 %f\n",R[0]);
		R[0] -= 0.5*cblas_ddot(N,&Yin[0],1,&Yd[0],1);
		//printf("3 %f\n",R[0]);

	}

class GP{
public:
    int D;
    int N;
    int K;
    int maxinfer;
    std::vector<double> ih;
    std::vector<double>  Kxx;
    std::vector<double> X;
    std::vector<double> Y;
    std::vector<double> Yd;
    std::vector<int> Dx;
    std::vector<double> S;
    std::vector<double> Ksx;
    std::vector<double> Ksx_T;
    GP(int d, int n, int kindex);
    int build_K();
    
    int set_Y(double* Yin);
    int set_S(double* Sin);
    int set_X(double* Xin);
    int set_D(int* Din);
    int set_hyp(double* hyp);
    void ping();
    int fac();
    int presolv();
    int infer_diag(int Ns, double* Xs, int* Ds, double* R);
    int infer_m(int Ns, double* Xs, int* Ds, double* R);
    int infer_full(int Ns, double* Xs, int* Ds, double* R);
    int llk(double* R);
    virtual int getnext(double* lb, double* ub, double* argmin, double* min, int npts){}
    virtual double acq(double* x){}
    virtual int timing(int x, double* T){};
};

GP::GP(int d, int n, int kindex){
		D = d;
		if (D>=10){
			printf("WARN: D>9 not supported in derivative notation!!!!!");
		}
		N = n;
		K = kindex;
                maxinfer = 400;
                printf("%d",maxinfer);
		Y = std::vector<double>(N);
		Yd = std::vector<double>(N);
		X = std::vector<double>(D*N);
		S = std::vector<double>(N);
		Dx = std::vector<int>(N);
		Kxx = std::vector<double>(N*N);
		Ksx = std::vector<double>(N*maxinfer);
		Ksx_T = std::vector<double>(N*maxinfer);
                if (K==0){
                    ih = std::vector<double>(D+1);
                }
                else if (K==1){
                    ih = std::vector<double>(3);
                }
                else{
                    printf("XXXX Bad kernel index %d XXX",K);
                }
		
                
	}

int GP::llk(double* R){
	R[0] = - 0.5*N*L2PI;
	for (int i=0; i<N; i++){
		R[0]-=log(Kxx[i*(N+1)]);
	}
	R[0] -= 0.5*cblas_ddot(N,&Y[0],1,&Yd[0],1);
	return 0;
}
int GP::infer_m(int Ns, double* Xs, int* Ds, double* R){
	//populate Kxs
    if (Ns>=this->maxinfer){
        
        this->maxinfer = 2*Ns;
        //printf("resize %d ",maxinfer);
    	Ksx.resize(N*maxinfer);
        Ksx_T.resize(N*maxinfer);
    }
	for (int i=0; i<Ns; i++){
		for (int j=0; j<N; j++){
			Ksx[i*N+j] = kern[K](&X[j*D],&Xs[i*D],Dx[j],Ds[i],D,&ih[0]);
		}
	}
	for (int i=0; i<Ns; i++){
		R[i] = cblas_ddot(N,&Y[0],1,&Ksx[i*N],1);
	}

	return 0;
}

int GP::infer_full(int Ns, double* Xs, int* Ds, double* R){
	//populate Kxs
    if (Ns>=maxinfer){
        maxinfer = 2*Ns;
        //printf("resize %d ",maxinfer);
    	Ksx.resize(N*maxinfer);
        Ksx_T.resize(N*maxinfer);
    }
	for (int i=0; i<Ns; i++){
		for (int j=0; j<N; j++){
			Ksx_T[i+j*Ns] = Ksx[i*N+j] = kern[K](&X[j*D],&Xs[i*D],Dx[j],Ds[i],D,&ih[0]);
		}
		R[Ns+Ns*i+i] = kern[K](&Xs[i*D],&Xs[i*D],Ds[i],Ds[i],D,&ih[0]);
		for (int h=i+1; h<Ns; h++){
			R[Ns+Ns*h+i]=R[Ns+Ns*i+h] = kern[K](&Xs[h*D],&Xs[i*D],Ds[h],Ds[i],D,&ih[0]);
		}
	}
    //I'm unconvinced this is the best way of doing it
	int c = LAPACKE_dpotrs(LAPACK_ROW_MAJOR,'L',N,Ns,&Kxx[0],N,&Ksx_T[0],Ns);

    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,Ns,Ns,N,-1.,&Ksx[0],N,&Ksx_T[0],Ns,1.,&R[Ns],Ns);
	for (int i=0; i<Ns; i++){
		R[i] = cblas_ddot(N,&Y[0],1,&Ksx[i*N],1);
	}

	return c;
}
int GP::infer_diag(int Ns, double* Xs, int* Ds, double* R){
	//populate Kxs
    //printf("x%d",maxinfer);
    if (Ns>=maxinfer){
        
        maxinfer = 2*Ns;
        //printf("resize %d ",maxinfer);
    	Ksx.resize(N*maxinfer);
        Ksx_T.resize(N*maxinfer);
    }
	for (int i=0; i<Ns; i++){
		for (int j=0; j<N; j++){
			Ksx_T[i+j*Ns] = Ksx[i*N+j] = kern[K](&X[j*D],&Xs[i*D],Dx[j],Ds[i],D,&ih[0]);
		}
	}

	int c = LAPACKE_dpotrs(LAPACK_ROW_MAJOR,'L',N,Ns,&Kxx[0],N,&Ksx_T[0],Ns);

	for (int i=0; i<Ns; i++){
		R[i] = cblas_ddot(N,&Y[0],1,&Ksx[i*N],1);
		R[Ns+i] = kern[K](&Xs[i],&Xs[i],Ds[i],Ds[i],D,&ih[0]) - cblas_ddot(N,&Ksx_T[i],Ns,&Ksx[i*N],1);
	}

	return c;
}
int GP::build_K(){
	for (int i=0; i<N; i++){
		Kxx[i*N+i] = kern[K](&X[i*D], &X[i*D],Dx[i],Dx[i],D,&ih[0]);
		for (int j=0; j<i; j++){
			Kxx[i*N+j] = Kxx[i+N*j] = kern[K](&X[i*D], &X[j*D],Dx[i],Dx[j],D,&ih[0]);
		}
		Kxx[i*N+i]+=S[i];

	}
	return 0;
}
int GP::set_hyp(double* hyp){
    if (K==0){
        ih[0] = pow(hyp[0],2);
	for (int i=1; i<D+1; i++){
		ih[i] = 1./pow(hyp[i],2);
	}
    }
    else if (K==1){
        ih[0] = pow(hyp[0],2);
        ih[1] = hyp[1];
        ih[2] = pow(hyp[2],2);
    }
    else{
        printf("XXXX Bad kernel index %d XXX",K);
    }
	
	return 0;
}

int GP::presolv(){
	return LAPACKE_dpotrs(LAPACK_ROW_MAJOR,'L',N,1,&Kxx[0],N,&Y[0],1);
}

int GP::fac(){

	return LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',N,&Kxx[0],N);
}

int GP::set_X(double* Xin){
	for (int i=0; i<N*D; i++){
			X[i] = Xin[i];
	}
	return 0;
}
int GP::set_D(int* Din){
	for (int i=0; i<N; i++){
			Dx[i] = Din[i];
	}
	return 0;
}
int GP::set_Y(double* Yin){
	for (int i=0; i<N; i++){
			Y[i] = Yd[i]= Yin[i];
	}
	return 0;
}
int GP::set_S(double* Sin){
	for (int i=0; i<N; i++){
			S[i] = Sin[i];
	}
	return 0;
}
void GP::ping(){
	printf("----------------------------------------\nGP in memory:\nX:\n");
	for (int i=0; i<N; i++){
		printf("[");
		for (int j=0; j<D; j++){
			printf("%f  ",X[i*D+j]);
		}
		printf("]\n");
	}
	printf("Y:\n[");

		for (int i=0; i<N; i++){
			printf("%f  ",Y[i]);
		}
	printf("]\nD:\n[");
		for (int i=0; i<N; i++){
			printf("%d  ",Dx[i]);
		}
	printf("]\nS:\n[");
		for (int i=0; i<N; i++){
			printf("%e  ",S[i]);
		}
	printf("]\nih:\n[");

	for (int i=0; i<D+1; i++){
		printf("%f  ",ih[i]);
	}

	printf("]\nKxx:\n[");
	for (int i=0; i<N; i++){
		printf("[");
		for (int j=0; j<N; j++){
			printf("%e  ",Kxx[i*N+j]);
		}
		printf("]\n");
	}
	printf("----------------------------------------\n");
	return;
}
