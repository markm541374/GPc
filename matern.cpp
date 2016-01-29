/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
const double FIVETHIRDS = 5./3.;
const double SQRT5 = sqrt(5.);
int mat52conv(double *h, int D, double* ih){
    ih[0] = pow(h[0],2);
    for (int i=1; i<D+1; i++){
	ih[i] = 1./pow(h[i],2);
    }
    return 0;
}

//all are searched in log space
int mat52searchconv(double *h, int D, double* ih){
    
    for (int i=0; i<D+1; i++){
        //printf("_%f",h[i]);
	ih[i] = pow(10.,h[i]);
    }
    return 0;
}

double mat52(double *x1, double *x2, int d1, int d2, int D, double* ih, double* smodel){
	double r2 = 0.;
	for (int i=0; i<D; i++){
		r2+=pow((x1[i]-x2[i]),2)*ih[i+1];
	}
        double sq5r2 = SQRT5*sqrt(r2);
        double core = exp(-sq5r2);
	
	if (d1==0 and d2==0){
		//no derivatives
		return ih[0]*(1.+sq5r2+FIVETHIRDS*r2)*core;
	}
        
        //else{printf("%d %d",d1,d2);}
	std::vector<int> V = std::vector<int>(D,0);
	div_t v1;
	div_t v2;
	int S = 0;

	int sign = 1;
	for (int i=0; i<D; i++){
		v1 = div(d1,pow(8,D-i-1));
		V[D-i-1] += v1.quot;
		S+=v1.quot;
		d1 = v1.rem;
		v2 = div(d2,pow(8,D-i-1));
		V[D-i-1] += v2.quot;
		sign += v2.quot;
		S+=v2.quot;
		d2 = v2.rem;
	}
	int P = 1;
	for (int i=0; i<D; i++){
		P*=V[i]+1;
	}
	sign = 2*(sign%2) -1;
	if (S==1){
		//first derivative
                int i =0;
		while (V[i]==0){i+=1;} //i no indexes he required dimension
                
		//printf("invalid derivatives %d %d",d1,d2);
                return ih[i+1]*(x1[i]-x2[i])*(-FIVETHIRDS)*(1+sq5r2)*core*double(sign);
	}
	else if (S==2){
		if (P==3){
			//second derivative
			printf("invalid derivatives %d %d",d1,d2);
                        return 0.;
		}
		else if (P==4){
			//first derivative on two axes
			printf("invalid derivatives %d %d",d1,d2);
                        return 0.;
		}
		else{
			printf("invalid derivatives %d %d",d1,d2);
			return 0.;
		}
	}
	else if (S==3){
		if (P==4){
			//third derivative
			printf("invalid derivatives %d %d",d1,d2);
                        return 0.;
		}
		else if (P==6){
			//second and first derivative
			printf("invalid derivatives %d %d",d1,d2);
                        return 0.;
		}
		else if (P==8){
			//three first derivatives
			printf("invalid derivatives %d %d",d1,d2);
                        return 0.;
		}
		else{
			printf("invalid derivatives %d %d",d1,d2);
			return 0.;
		}
	}
	else if(S==4){
		if (P==16){
			//four first derivatives
			printf("invalid derivatives %d %d",d1,d2);
                        return 0.;
		}
		else if (P==12){
			//one second and two first
			printf("invalid derivatives %d %d",d1,d2);
                        return 0.;
		}
		else if (P==9){
			//two second derivatives
			printf("invalid derivatives %d %d",d1,d2);
                        return 0.;
		}
		else if (P==8){
			//third and first derivative
			printf("invalid derivatives %d %d",d1,d2);
                        return 0.;
		}
		else if (P==5){
                    //fourth derivative
                    printf("invalid derivatives %d %d",d1,d2);
                    return 0.;
		}
		else{
			printf("invalid derivatives %d %d",d1,d2);
			return 0.;
		}
	}
	else{
		printf("invalid derivativesx %d %d %d",d1,d2,S);
		return 0.;
	}

}