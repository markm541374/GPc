#include <vector>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

double squexp(double *x1, double *x2, int d1, int d2, int D, double* ih){
	double expon = 0.;
	for (int i=0; i<D; i++){
		expon-=pow((x1[i]-x2[i]),2)*ih[i+1];
	}
	double core = ih[0]*exp(0.5*expon);
	if (d1==0 and d2==0){
		//no derivatives
		return core;
	}
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
		while (V[i]==0){i+=1;}
		return -ih[i+1]*(x1[i]-x2[i])*double(sign)*core;
	}
	else if (S==2){
		if (P==3){
			//second derivative
			int i =0;
			while (V[i]==0){i+=1;}
			return ih[i+1]*(ih[i+1]*pow((x1[i]-x2[i]),2)-1.) * double(sign)*core;
		}
		else if (P==4){
			//first derivative on two axes
			int i = 0;
			while (V[i]==0){i+=1;}
			int j = i+1;
			while (V[j]==0){j+=1;}
			printf("%d,%d\n",i,j);
			return ih[j+1]*(x1[j]-x2[j])*ih[i+1]*(x1[i]-x2[i])*double(sign)*core;
		}
		else{
			printf("invalid derivatives");
			return 0.;
		}
	}
	else if (S==3){
		if (P==4){
			//third derivative
			//(li**2)*(3*xi-li*xi**3)
			int i = 0;
			while (V[i]==0){i+=1;}
			double x = (x1[i]-x2[i]);
			double l = ih[i+1];
			return pow(l,2)*(3.*x-l*pow(x,3))*double(sign)*core;
		}
		else if (P==6){
			//second and first derivative
			//-xi*li*lj*(lj*xj**2-1)
			//j is the repeated axis
			int i = 0;
			while (V[i]!=1){i+=1;}
			int j = 0;
			while (V[j]!=2){j+=1;}
			printf("[%d,%d]\n",i,j);
			double xi = (x1[i]-x2[i]);
			double li = ih[i+1];
			double xj = (x1[j]-x2[j]);
			double lj = ih[j+1];
			return -xi*li*lj*(lj*pow(xj,2)-1)*double(sign)*core;
		}
		else if (P==8){
			//three first derivatives
			int i = 0;
			while (V[i]==0){i+=1;}
			int j = i+1;
			while (V[j]==0){j+=1;}
			int k = j+1;
			while (V[k]==0){k+=1;}
			printf("%d,%d,%d\n",i,j,k);
			//- xi*li*xj*lj*xk*lk
			return -ih[k+1]*(x1[k]-x2[k])*ih[j+1]*(x1[j]-x2[j])*ih[i+1]*(x1[i]-x2[i])*double(sign)*core;

			return 0.;
		}
		else{
			printf("invalid derivatives");
			return 0.;
		}
	}
	else if(S==4){
		if (P==16){
			//four first derivatives
			//xi*li*xj*lj*xk*lk*xl*ll
			int i = 0;
			while (V[i]==0){i+=1;}
			int j = i+1;
			while (V[j]==0){j+=1;}
			int k = j+1;
			while (V[k]==0){k+=1;}
			int l = k+1;
			while (V[k]==0){l+=1;}
			return ih[l+1]*(x1[l]-x2[l])*ih[k+1]*(x1[k]-x2[k])*ih[j+1]*(x1[j]-x2[j])*ih[i+1]*(x1[i]-x2[i])*double(sign)*core;
		}
		else if (P==12){
			//one second and two first
			//lk*(lk*xk**2-1)*xi*li*xj*lj
			//k is the repeated axis

			int i = 0;
			while (V[i]!=1){i+=1;}
			int j = i+1;
			while (V[j]==0){j+=1;}
			int k = 0;
			while (V[k]!=2){k+=1;}
			double xi = (x1[i]-x2[i]);
			double li = ih[i+1];
			double xj = (x1[j]-x2[j]);
			double lj = ih[j+1];
			double xk = (x1[k]-x2[k]);
			double lk = ih[k+1];

			return lk*(lk*pow(xk,2)-1)*xi*li*xj*lj*double(sign)*core;
		}
		else if (P==9){
			//two second derivatives
			//li*(li*xi**2-1)*lj*(lj*xj**2-1)
			int i = 0;
			while (V[i]==0){i+=1;}
			int j = i+1;
			while (V[j]==0){j+=1;}
			double xi = (x1[i]-x2[i]);
			double li = ih[i+1];
			double xj = (x1[j]-x2[j]);
			double lj = ih[j+1];

			return li*(li*pow(xi,2)-1)*lj*(lj*pow(xj,2)-1)*double(sign)*core;
		}
		else if (P==8){
			//third and first derivative
			//-li*xi*(lj**2)*(3*xj-lj*xj**3)
			//j is the repeated axis
			int i = 0;
			while (V[i]!=1){i+=1;}
			int j = 0;
			while (V[j]!=3){j+=1;}
			double xi = (x1[i]-x2[i]);
			double li = ih[i+1];
			double xj = (x1[j]-x2[j]);
			double lj = ih[j+1];
			return -li*xi*pow(lj,2)*(3*xj-lj*pow(xj,3))*double(sign)*core;
		}
		else if (P==5){
			//fourth derivative
			//(li*xi)**4 - 6*(li**3)*(xi**2) + 3*li**2
			int i = 0;
			while (V[i]==0){i+=1;}
			double xi = (x1[i]-x2[i]);
			double li = ih[i+1];
			return (pow(li*xi,4) - 6*pow(li,3)*pow(xi,2) + 3*pow(li,2))*double(sign)*core;
		}
		else{
			printf("invalid derivatives");
			return 0.;
		}
	}
	else{
		printf("invalid derivatives");
		return 0.;
	}

}
typedef double (*FP)(double*, double*, int, int, int, double*);

extern "C" const FP kern[1] = {&squexp};

extern "C" double k(double *x1, double *x2, int d1, int d2, int D, double* ih, int kindex){
	return kern[kindex](&x1[0], &x2[0], d1, d2, D, &ih[0]);
}
//double (*kern)(double *x1, double *x2, int d1, int d2, int D, double* ih) = &k;