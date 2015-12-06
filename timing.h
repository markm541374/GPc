/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   timing.h
 * Author: mark
 *
 * Created on 02 December 2015, 19:07
 */

#ifndef TIMING_H
#define TIMING_H


class GP_timing : public GP{
    public:
        using GP::GP;
        
        int timing(int, double*);
};
#endif /* TIMING_H */

