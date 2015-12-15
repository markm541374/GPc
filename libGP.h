/*
 * libGPd.h
 *
 *  Created on: 26 Oct 2015
 *      Author: mark
 */

#ifndef LIBGPD_H_
#define LIBGPD_H_


extern "C" int HypSearchMLE(int d, int n, double* Xin, double* Yin, double* Sin, int* Din, double* lb, double* ub, double* Rhyp, double lk);
extern "C" void SetHypSearchPara(int mx, double fg);
int infer_full(int k,int Ns,double* Xs, int* Ds, double* R);
int llk(int k, double* R);
int infer_m(int k,int Ns,double* Xs, int* Ds, double* R);
int infer_diag(int k,int Ns,double* Xs, int* Ds, double* R);
int draw(int k, int Nd, double* X, int* D, double* R, int m);
int presolv(int k);
int build_K(int k);
int fac(int k);
int set_hyp(int k, double* h);
int set_X(int k, double* X);
int set_D(int k, int* D);
int set_S(int k, double* Sin);
int set_Y(int k, double* Y);
int ping(int k);
void killGP(int k);
int newGP_LKonly(int D, int N, double* Xin, double* Yin, double* Sin, int* Din, double* hyp, double* R);
int newGP(int D, int N);
int infer_LCB(int k, int n, double* X, int* D, double p, double* R);
int infer_EI(int k, int n, double* X, int* D, double* R);
#endif /* LIBGPD_H_ */