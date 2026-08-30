//----------------------------------------------------------------------
//   compile: nvcc cell2D_adh.cu -lcuda -lcufft -lcublas -O3 -o celladh
//----------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <stdlib.h>
#include <cmath>
#include <cuda.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>

#define N 512
#define N2  (N*N)
#define M_PI 3.141592654

using namespace std;

typedef struct {
    double re;
    double im;
} COMPLEX;


//===========================================================================
// CUDA Kernels
//===========================================================================
//------------------------------------------------------------------------------
// Scaling helper - returens value of applied adh depending on D (interpoltes) 
//------------------------------------------------------------------------------
__host__ double scale_adh(double D) {


    static const double D_ref[] = { 0.2,  0.3,   0.4,  0.6,  0.8,  1.0,  1.2,  1.5};
    static const double adh[] = { 686.875,692.978, 659.61, 537.646, 405.432, 298.999, 223.128, 151.289};


    const int n = sizeof(D_ref)/sizeof(D_ref[0]);
    if (D <= D_ref[0])   return adh[0];
    if (D >= D_ref[n-1]) return adh[n-1];
    for (int i = 0; i < n-1; ++i)
        if (D <= D_ref[i+1]) {
            double t = (D - D_ref[i]) / (D_ref[i+1] - D_ref[i]);   //linear interp
            return adh[i] + t*(adh[i+1] - adh[i]);
        }
    return adh[n-1];
}

//---------------------------------------------------------------------------
// update rho1 and rho2
//---------------------------------------------------------------------------
__global__ void nonlin_update(COMPLEX *rhoA,  COMPLEX  *NrhoA, double dt, int k, double delta, double delta2) {

    double rrx1, rry1, rrx2, rry2, r1, r2;

    double adh1 = 0.0;
    double adh2 = 0.0;

    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    int i,j,im,ip,jm,jp;
    i= idx/N;
    j= idx%N;

    ip=i+1;
    im=i-1;
    jp=j+1;
    jm=j-1;

    if(i==N-1){ip=0;}
    if(j==N-1){jp=0;}
    if(i==0){im=N-1;}
    if(j==0){jm=N-1;}

    if(idx<N2)
    {
        
        double dx = 0.3125;
        double L=N*dx;

        r1 = rhoA[idx].re;
 	r2 = rhoA[idx].im;
	
	// gradients
        rrx1 = (rhoA[j+N*ip].re - rhoA[j+N*im].re)/2.0/dx;
        rry1 = (rhoA[jp+N*i].re - rhoA[jm+N*i].re)/2.0/dx;
	    rrx2 = (rhoA[j+N*ip].im - rhoA[j+N*im].im)/2.0/dx;
        rry2 = (rhoA[jp+N*i].im - rhoA[jm+N*i].im)/2.0/dx;

	// adhesion
	double a_t = 5.0;//*(223.128/686.875);
	//if(k>50000){a_t=0.0;}



    double g1=rrx1*rrx1+rry1*rry1; 
    double g2=rrx2*rrx2+rry2*rry2;   
    double denom1=sqrt(1.0+g1); 
    double denom2=sqrt(1.0+g2);
    double nx1,ny1,nx2,ny2;
    nx1=rrx1/ denom1;
    ny1=rry1/ denom1;
    nx2=rrx2/ denom2;
    ny2=rry2/ denom2;

    adh1 += -a_t * (rrx1 * nx2 + rry1 * ny2);
    adh2 += -a_t * (rrx2 * nx1 + rry2 * ny1);




	// time update: phase field, adhesion, volume exclusion
    NrhoA[idx].re =  r1 + dt*( -(1.0 - r1)*(delta - r1)*r1 + adh1 - 5.0*r2*r2*r1 );
 	NrhoA[idx].im =  r2 + dt*( -(1.0 - r2)*(delta2 - r2)*r2 + adh2 - 5.0*r1*r1*r2 );

    // const double eps = 2e-5;
    // if (NrhoA[idx].re < eps){ NrhoA[idx].re = 0.0;}
    // if (NrhoA[idx].im < eps){ NrhoA[idx].im = 0.0;}


    }
};

__global__ void measure_adh(COMPLEX *rhoA, double *gadh,  double dx) {
    double rrx1, rry1, rrx2, rry2;

    double adh1 = 0.0;
    double adh2 = 0.0;

    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    int i,j,im,ip,jm,jp;
    i= idx/N;
    j= idx%N;

    ip=i+1;
    im=i-1;
    jp=j+1;
    jm=j-1;

    if(i==N-1){ip=0;}
    if(j==N-1){jp=0;}
    if(i==0){im=N-1;}
    if(j==0){jm=N-1;}

    if(idx<N2)
    {
        
        double dx = 0.3125;
        double L=N*dx;


	
	// gradients
        rrx1 = (rhoA[j+N*ip].re - rhoA[j+N*im].re)/2.0/dx;
        rry1 = (rhoA[jp+N*i].re - rhoA[jm+N*i].re)/2.0/dx;
	    rrx2 = (rhoA[j+N*ip].im - rhoA[j+N*im].im)/2.0/dx;
        rry2 = (rhoA[jp+N*i].im - rhoA[jm+N*i].im)/2.0/dx;

	// adhesion
	double a_t = 5.0;//*(223.128/686.875);
	// if(k<50000){a_t=0.0;}



    
    double g1=rrx1*rrx1+rry1*rry1; 
    double g2=rrx2*rrx2+rry2*rry2;   
    double denom1=sqrt(1.0+g1); 
    double denom2=sqrt(1.0+g2);
    double nx1,ny1,nx2,ny2;
    nx1=rrx1/ denom1;
    ny1=rry1/ denom1;
    nx2=rrx2/ denom2;
    ny2=rry2/ denom2;

     adh1 += -a_t * (rrx1 * nx2 + rry1 * ny2);
     adh2 += -a_t * (rrx2 * nx1 + rry2 * ny1);

    gadh[idx]=fabs(adh1)+fabs(adh2);
    
}};
//---------------------------------------------------------------------------
__global__  void diffusion_evolution(COMPLEX *rhoAf, COMPLEX *rhoAg, double *cor1)
{
    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(idx<N2)
    {

    rhoAg[idx].re = cor1[idx]*rhoAf[idx].re;
    rhoAg[idx].im = cor1[idx]*rhoAf[idx].im;

    }
};



//---------------------------------------------------------------------------
__global__ void complexToReal1(COMPLEX *rhoA, double *rho)
{
    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(idx<N2)
    {
        rho[idx] = rhoA[idx].re;
    }
};


//---------------------------------------------------------------------------
__global__ void complexToReal2(COMPLEX *rhoA, double *rho)
{
    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(idx<N2)
    {
        rho[idx] = rhoA[idx].im;
    }
};




//===========================================================================

int main(int argc, char** argv)
{
    // for folder name
    int number=202;

    dim3 dGRID,dBLOCK;
    int GPUID=0;

    COMPLEX  *rhoA, *GrhoA, *GNrhoA;
    double *rho,*Grho;
    double *cor1,*Gcor1;

    cublasHandle_t handle;
    cufftHandle Gfftplan;

    int memN2c,memN2r;

    double totmass, totmass2, delta, delta2,  dt=0.001;
    double r0,D_r;
    double r02, mass0, mass02;
    double qx[N],qy[N],qsq;
        


    D_r= atof(argv[1]);


        double dx = 0.3125;
        double L=N*dx;	
    double dx2=dx*dx;
    double dk, scale = 1.0*N*N, mu=0.1;

    int steps, blind;

    double time=0.,timestart=0.0;
  

    // initialize fields

    rho= new double[N2];
    cor1 = new double[N2];
    rhoA = new COMPLEX[N2];

    // ready GPU-computing
    cudaSetDevice(GPUID);
    dBLOCK=dim3(512,1,1);
    dGRID=dim3(512,1,1);

    // Create CUDA FFT plan
    cufftPlan2d(&Gfftplan, N, N, CUFFT_Z2Z);
    cublasCreate(&handle);

    memN2c=N2*sizeof(COMPLEX);
    memN2r=N2*sizeof(double);

    //complex arrays on GPU
    cudaMalloc((void**)&GrhoA, memN2c);
    cudaMalloc((void**)&GNrhoA, memN2c);


    //real arrays on GPU
    cudaMalloc((void**)&Grho,memN2r);
    cudaMalloc((void**)&Gcor1,memN2r);

    //read input params
     //cin >> steps;
     //cin >> blind;
     //cin >> r0;
     //cin >> D_r;

	steps=100000;
	blind=1000;
	r0=10.0;
	


    r02=r0*r0;
    dk=2.*M_PI/L;



    double *gadh;
    cudaMalloc((void**)&gadh,memN2r); 


    if (true)
    {
        // initial conditions
        totmass=0.;

        for(int i=0; i<N; i++)
            for (int j=0; j<N; j++)
            {


                if ( dx*dx*(((i-N/2+25)*(i-N/2+25)+(j-N/2)*(j-N/2))) <= r02)
                {
                    rhoA[j+N*i].re =1.0;

                }
                else
                {
                    rhoA[j+N*i].re = 0.0;
                }

                if ( dx*dx*(((i-N/2-25)*(i-N/2-25)+(j-N/2)*(j-N/2))) <= r02)
                {
                    rhoA[j+N*i].im =1.0;

                }
                else
                {
                    rhoA[j+N*i].im = 0.0;
                }

            }
}


    //calculate q's
    for(int i=0; i<=N/2; i++)
    {
        qx[i]=dk*i;
        qy[i]=dk*i;
    }
    for(int i=1; i<N/2; i++)
    {
        qx[N-i]=-dk*i;
        qy[N-i]=-dk*i;
    }

    // cor matrix
    for(int i=0; i<N; i++)
        for (int j=0; j<N; j++)
        {
        qsq = qx[i]*qx[i]+qy[j]*qy[j];
        cor1[j+N*i] = exp(-dt*D_r*qsq)/scale;
        }

    cudaMemcpy(GrhoA,rhoA,  memN2c, cudaMemcpyHostToDevice);
    cudaMemcpy(Gcor1,cor1,  memN2r, cudaMemcpyHostToDevice);

complexToReal1<<<dGRID, dBLOCK>>>(GrhoA, Grho);
cublasDasum(handle, N2, Grho, 1, &mass0);

complexToReal2<<<dGRID, dBLOCK>>>(GrhoA, Grho);
cublasDasum(handle, N2, Grho, 1, &mass02);

delta=0.5;
delta2=0.5;

ofstream csv("adh_vs_D.csv", ios::app);
if(csv.tellp()==0) csv<<"adhval,D,time,ADH\n";

//--------------------------------------------------------------------------------------
// timesteps
//--------------------------------------------------------------------------------------



    for(int k=0; k<steps+1; k++)
    {

        time+=dt;

        nonlin_update<<<dGRID, dBLOCK>>>(GrhoA, GNrhoA, dt, k, delta, delta2);

        cufftExecZ2Z(Gfftplan,(cufftDoubleComplex*)GNrhoA,(cufftDoubleComplex*)GrhoA,CUFFT_FORWARD);
        diffusion_evolution<<<dGRID, dBLOCK>>>( GrhoA, GNrhoA, Gcor1);
        cufftExecZ2Z(Gfftplan,(cufftDoubleComplex*)GNrhoA,(cufftDoubleComplex*)GrhoA,CUFFT_INVERSE);


        complexToReal1<<<dGRID, dBLOCK>>>(GrhoA, Grho);
        //calculate volume fix point1
        cublasDasum(handle, N2, Grho, 1, &totmass);
        delta = 0.5 + mu*(totmass-mass0);

        if(k%(blind)==0){
            // cudaMemcpy(rho, Grho, memN2r, cudaMemcpyDeviceToHost);

            // //ausgabe
            // ofstream outf1;
            // ostringstream name1;
            // name1<<"out_"<<number<<"/rho-t"<<time-dt<<".txt";
            // outf1.open(name1.str().c_str());
            // for(int i=0;i<N;i++){
            //     for(int j=0;j<N;j++){
            //         if(fabs(rho[j+N*i]) > 0.001){outf1<<rho[j+N*i]<<" ";}
            //         else{outf1<<0.0<<" ";}
            //     }
            //     outf1<<endl;
            // }
            // outf1.close();
            }

        complexToReal2<<<dGRID, dBLOCK>>>(GrhoA, Grho);
        //calculate volume fix point2
        cublasDasum(handle, N2, Grho, 1, &totmass2);
        delta2 = 0.5 + mu*(totmass2-mass02);


         
          

            if(k%(blind)==0){
                cudaMemcpy(rho, Grho, memN2r, cudaMemcpyDeviceToHost);

                //ausgabe
                // ofstream outf1;
                // ostringstream name1;
                // name1<<"out_"<<number<<"/rho2-t"<<time-dt<<".txt";
                // outf1.open(name1.str().c_str());
                // for(int i=0;i<N;i++){
                //     for(int j=0;j<N;j++){
                //         if(fabs(rho[j+N*i]) > 0.001){outf1<<rho[j+N*i]<<" ";}
                //         else{outf1<<0.0<<" ";}
                //     }
                //     outf1<<endl;
                // }
                // outf1.close();




                measure_adh<<<dGRID,dBLOCK>>>(GrhoA,gadh,dx);
                double sum_adh=0.0;
                cublasDasum(handle, N2, gadh, 1, &sum_adh);
            
                csv<<"5"<<","<<D_r<<","<<(time-dt)<<","<<sum_adh<<"\n";
                csv.flush();


            }


        

    }

    csv.close();
    return 0;
}

    


