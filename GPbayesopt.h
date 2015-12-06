/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   GPbayesopt.h
 * Author: mark
 *
 * Created on 26 November 2015, 23:12
 */

#ifndef GPBAYESOPT_H
#define GPBAYESOPT_H

class EI_direct : public GP{
    public:
        using GP::GP;
        int getnext(double* lb, double* ub, double* argmin, double* min, int npts);
        double acq(double* x);
};

class EI_random : public GP{
    public:
        using GP::GP;
        int getnext(double* lb, double* ub, double* argmin, double* min, int npts);
        double acq(double* x);
};


#endif /* GPBAYESOPT_H */

