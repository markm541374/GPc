/*
 * kernel.h
 *
 *  Created on: 24 Oct 2015
 *      Author: mark
 */

#ifndef KERNEL_H_
#define KERNEL_H_

extern "C" double k(double *x1, double *x2, int d1, int d2, int D, double* ih, double kindex);
//extern "C" double (*kern)(double *x1, double *x2, int d1, int d2, int D, double* ih);

typedef double (*FP)(double *x1, double *x2, int d1, int d2, int D, double* ih);

extern "C" const FP kern[1];


#endif /* KERNEL_H_ */