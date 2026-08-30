 //----------------------------------------
// compile with:
// nvcc --std=c++17 unity.cu -lcuda -lcufft -lcublas -O3 -o unity
//----------------------------------------


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
#include <string>
#include <filesystem>

#include <random>
#include <curand_kernel.h>

//dimensions of box and canvas
#define N 128
#define N2  (N*N)
#define MOUT 512
#define BLOCK_SIZE 512

// #define N 256
// #define N2  (N*N)

// -------------------------
// constants (phys. world size, intr. cutoff etc...)
// -------------------------

__constant__ double d_Lcell;
__constant__ double d_dx;
__constant__ double d_R0;
__constant__ double d_cutoff;


#define M_PI 3.141592654

using namespace std;

typedef struct {
    double re;
    double im;
} COMPLEX;



//===========================================================================
// FUNCS
//===========================================================================


////////////////////////////////////////////////////////////////////
//////////////////////    UPDATE + its helpers   ////////////////////////////////
////////////////////////////////////////////////////////////////////

//------------------------------------------------------------------------------
// Sampling of "pixels" between local boxes relative to big world canvas, 
// for sampling either a) "pixel" value of neighbour or b) gradient value of neighbour (both for non lin update)
//  --> !NON-PERIODIC! 
//------------------------------------------------------------------------------
__device__ __forceinline__ double bilinear_sample_shifted( const COMPLEX* __restrict__ field,int cell,
    int i, int j,
    int sx, int sy,
    double fx, double fy
){
    //which pair and .re or .im
    int pair = cell / 2;
    int part = cell % 2;
    int off = pair * N2;

    //samplede point lives between point0 and point1(kind of a box)
    int x1 = i - sx;
    int y1 = j - sy;
    int x0 = x1 - 1;
    int y0 = y1 - 1;

    //non per.
    if ((unsigned)x1 >= (unsigned)N || (unsigned)y1 >= (unsigned)N) return 0.0;
    if ((unsigned)x0 >= (unsigned)N || (unsigned)y0 >= (unsigned)N) return 0.0;

    // value of surrounding pixels
    double v11, v01, v10, v00;
     //check wheter cell to sample is .re or .im living
    if (part == 0) {
        v11 = field[off + x1*N + y1].re;
        v01 = field[off + x0*N + y1].re;
        v10 = field[off + x1*N + y0].re;
        v00 = field[off + x0*N + y0].re;
    } else {
        v11 = field[off + x1*N + y1].im;
        v01 = field[off + x0*N + y1].im;
        v10 = field[off + x1*N + y0].im;
        v00 = field[off + x0*N + y0].im;
    }

    // fraction
    //weights
    double wx1 = 1.0 - fx;
    double wy1 = 1.0 - fy;

    //value of allround field*corres[ponding weight]
    return (wx1 * wy1) * v11
         + (fx  * wy1) * v01
         + (wx1 * fy ) * v10
         + (fx  * fy ) * v00;
}


//------------------------------------------------------------------------------
// Periodic pixel helper
//------------------------------------------------------------------------------
__device__ inline int pbc(int x) {
    if (x >= N) x -= N;
    if (x < 0)  x += N;
    return x;
}


//------------------------------------------------------------------------------
// Gradients per re-im field, central difference, periodic
//------------------------------------------------------------------------------
__global__ void compute_gradients(const COMPLEX* __restrict__ rhoA,COMPLEX* __restrict__ gradx, COMPLEX* __restrict__ grady,int nPairs){
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int pair = blockIdx.y;


    if (idx >= N2 || pair >= nPairs) return;

    int i = idx / N;
    int j = idx % N;

    int ip = pbc(i + 1);
    int im = pbc(i - 1);
    int jp = pbc(j + 1);
    int jm = pbc(j - 1);

    int off = pair * N2;
    double dx = d_dx;

    gradx[off + idx].re =(rhoA[off + ip*N + j].re - rhoA[off + im*N + j].re) / (2.0 * dx);
    grady[off + idx].re = (rhoA[off + i*N + jp].re - rhoA[off + i*N + jm].re) / (2.0 * dx);

    gradx[off + idx].im = (rhoA[off + ip*N + j].im - rhoA[off + im*N + j].im) / (2.0 * dx);
    grady[off + idx].im =(rhoA[off + i*N + jp].im - rhoA[off + i*N + jm].im) / (2.0 * dx);
}

//------------------------------------------------------------------------------
// Per-"pixel" update, updates the .re and .im part of given "pixel" of field
//------------------------------------------------------------------------------
__global__ void nonlin_update(COMPLEX *rhoA, COMPLEX *NrhoA, double dt, int k, double sdk, double fmi,double repulsion, 
    double *delta, double *mass0, int nCells, int bundles, int running_r,int timing, int running_k, COMPLEX *Gfd,
    COMPLEX *gradx,COMPLEX *grady,const double *cx,const double *cy) {



    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int pair = blockIdx.y;
    //global idx of pair cells
    int cell_re = 2 * pair;
    int cell_im = 2 * pair + 1;

    if (idx >= N2 || cell_im >= nCells|| cell_re >= nCells) return;


    //indices 
    int offset_i = pair * N2;
    int i = idx / N;
    int j = idx % N;
    double dx = d_dx;

     //field and delta   
    double r_re = rhoA[offset_i + idx].re;
    double r_im = rhoA[offset_i + idx].im; // 

    double delta_i_re = delta[cell_re];
    double delta_i_im  = delta[cell_im];

    //cemters
    double cx_re = cx[cell_re];
    double cy_re = cy[cell_re];

    double cx_im = cx[cell_im];
    double cy_im = cy[cell_im];

    //gradients + update components
    double grad_rex = gradx[offset_i + idx].re;
    double grad_rey = grady[offset_i + idx].re;

    double grad_imx = gradx[offset_i + idx].im;
    double grad_imy = grady[offset_i + idx].im;

    double adh_re = 0.0;
    double rep_re = 0.0;

    double adh_im = 0.0;
    double rep_im = 0.0;

    double rhs_re = -(1.0 - r_re) * (delta_i_re - r_re) * r_re;
    double rhs_im = -(1.0 - r_im) * (delta_i_im - r_im) * r_im;

    //how far interactions go
    double cutoff  = d_cutoff;
    double cutoff2 = cutoff * cutoff;

    //adhesion matrix
    double A[6][6] = {
        // 1    2    3    4    5    6
        {0.0, sdk, 0.0, 0.0, fmi, 0.0}, // 1
        {sdk, fmi, sdk, 0.0, fmi, fmi}, // 2
        {0.0, sdk, 0.0, sdk, 0.0, 0.0}, // 3
        {0.0, 0.0, sdk, 0.0, sdk, 0.0}, // 4
        {fmi, fmi, 0.0, sdk, fmi, sdk}, // 5
        {0.0, fmi, 0.0, 0.0, sdk, 0.0} }; // 6
     double repul =repulsion;
    // double a1c;
    // double a2c;
    // i also just can pass it over in func
    //int pairs=  nCells/2;
    //int tot_bundles =pairs/3;

    
   
        if(mass0[cell_re]>0 && mass0[cell_im]>0){
            // if MY pair is here (mass>0)
            //get pair (complex) out of array (ME)
            //int off = pair*N2;
            
                //MY cell type
                int t1, t2;

                if (pair%3 == 0)
                {
                
                    t1 = 1;   // 2
                    t2 = 4;   //5
                }
                else if (pair%3 == 1)
                {
                    
                    t1 = 2;   // 3
                    t2 = 3;   //4
                }
                else
                {
                    t1 = 0;   //1
                    t2 = 5;   //6
                }
                // MY row
                int my_row = (pair/3) /bundles;
                int my_bundle = pair/3;


     // sum over ALL cells (but me)
           for (int c = 0; c < nCells; c++) {

                    if (mass0[c] <= 0.0) continue;

                    int pair_other = c / 2;
                    int part = c % 2;

                    int other_row = (pair_other / 3) / bundles;
                    int other_bundle = pair_other / 3;

                    int to;
                    
              
                    if (pair_other % 3 == 0) {
                        to = (part == 0) ? 1 : 4;   // 2,5
                    } else if (pair_other % 3 == 1) {
                        to = (part == 0) ? 2 : 3;   // 3,4
                    } else {
                        to = (part == 0) ? 0 : 5;   // 1,6
                    }

                     bool other_is_1256 = (to == 0 || to == 1 || to == 4 || to == 5);

                    /////////// RE to other //////////////////////////////
                    if (c != cell_re) {
                        //how far apart?
                        double dx_c_re = cx[c] - cx_re;
                        double dy_c_re = cy[c] - cy_re;
                        double dist2_re = dx_c_re * dx_c_re + dy_c_re * dy_c_re;

                        //if boxes too far apart do not contribute!
                        if (dist2_re < cutoff2){
                            //true pixel offsrt + full/fraction parts
                            double sx_f_re = dx_c_re / dx;
                            double sy_f_re = dy_c_re / dx;
                            int sx_re = (int)floor(sx_f_re);
                            int sy_re = (int)floor(sy_f_re);
                            double subx_re = sx_f_re - (double)sx_re;
                            double suby_re = sy_f_re - (double)sy_re;

                            //shift field/grad
                            double r_other_re = bilinear_sample_shifted(rhoA, c, i, j, sx_re, sy_re, subx_re, suby_re);
                            double grad_jx_re = bilinear_sample_shifted(gradx, c, i, j, sx_re, sy_re, subx_re, suby_re);
                            double grad_jy_re = bilinear_sample_shifted(grady, c, i, j, sx_re, sy_re, subx_re, suby_re);

                            //normalize
                            double g2_re = grad_jx_re * grad_jx_re + grad_jy_re * grad_jy_re;
                            double denom_re = sqrt(1.0 + g2_re);
                            double nx_re = grad_jx_re / denom_re;
                            double ny_re = grad_jy_re / denom_re;

                            bool me_is_1256_1 = (t1 == 0 || t1 == 1 || t1 == 4 || t1 == 5);
                            bool connec_1256 = me_is_1256_1 && other_is_1256;
                            double a1c = 0.0;

                            if (abs(t1 - to) == 1 && my_bundle == other_bundle) { //sdk
                                a1c = A[t1][to];
                            }
                            else if(connec_1256 &&( my_bundle == other_bundle)){ //not allow fmi inside

                                a1c = 0.0;
                            }
                            else if (abs(t1 - to) == 1 && my_bundle != other_bundle){ //to prevent cross bundle sdk 

                                a1c = 0.0;
                            }
                            else{
                                a1c = A[t1][to];
                            }

                            //differential adhesion
                            // a1c = A[t1][to];

                            //SUM
                            adh_re += -a1c * (grad_rex * nx_re + grad_rey * ny_re);
                            rep_re += -repul * r_other_re * r_other_re * r_re;
                        }
                    }

                    ///////////// IM to other //////////////////////////////
                    if (c != cell_im) {
                        //centers (how far apart)
                        double dx_c_im = cx[c] - cx_im;
                        double dy_c_im = cy[c] - cy_im;
                        double dist2_im = dx_c_im * dx_c_im + dy_c_im * dy_c_im;
                   
                        if (dist2_im < cutoff2){
                            //true pixel offsrt + full/fraction parts
                            double sx_f_im = dx_c_im / dx;
                            double sy_f_im = dy_c_im / dx;
                            int sx_im = (int)floor(sx_f_im);
                            int sy_im = (int)floor(sy_f_im);
                            double subx_im = sx_f_im - (double)sx_im;
                            double suby_im = sy_f_im - (double)sy_im;


                            //shift
                            double r_other_im = bilinear_sample_shifted(rhoA, c, i, j, sx_im, sy_im, subx_im, suby_im);
                            double grad_jx_im = bilinear_sample_shifted(gradx, c, i, j, sx_im, sy_im, subx_im, suby_im);
                            double grad_jy_im = bilinear_sample_shifted( grady, c, i, j, sx_im, sy_im, subx_im, suby_im);

                            //norm

                            double g2_im = grad_jx_im * grad_jx_im + grad_jy_im * grad_jy_im;
                            double denom_im = sqrt(1.0 + g2_im);
                            double nx_im = grad_jx_im / denom_im;
                            double ny_im = grad_jy_im / denom_im;

                            bool me_is_1256_2 = (t2 == 0 || t2 == 1 || t2 == 4 || t2 == 5);
                            bool connec_1256 = me_is_1256_2 && other_is_1256;
                            double a2c = 0.0;

                            if (abs(t2 - to) == 1 && my_bundle == other_bundle) { //sdk
                                a2c = A[t2][to];
                            }
                            else if(connec_1256 &&( my_bundle == other_bundle)){ //not allow fmi inside

                                a2c = 0.0;
                            }
                            else if (abs(t2 - to) == 1 && my_bundle != other_bundle){ //to prevent cross bundle sdk 

                                a2c = 0.0;
                            }
                            else{
                                a2c = A[t2][to]; //
                            }

                            adh_im += -a2c * (grad_imx * nx_im + grad_imy * ny_im);
                            rep_im += -repul * r_other_im * r_other_im * r_im;
                        }
                    }
                }     //new fields of ME
                        
                        NrhoA[offset_i + idx].re =r_re + dt*(rhs_re + adh_re + rep_re );
                        NrhoA[offset_i + idx].im =r_im + dt*( rhs_im+ adh_im + rep_im );

                        
                        // Gfd[offset_i + idx].re = dt*( rhs_re + adh_re + rep_re );
                        // Gfd[offset_i + idx].im = dt*( rhs_im + adh_im + rep_im );

                        //clamp if needed
                        const double eps = 2e-5;
                        if (NrhoA[offset_i+idx].re < eps){ NrhoA[offset_i + idx].re = 0.0;}
                        if (NrhoA[offset_i + idx].im < eps){ NrhoA[offset_i + idx].im = 0.0;}
    }
    
    
    
};

////////////////////////////////////////////////////////////////////
//////////      DIFFUSION     ///////////////
////////////////////////////////////////////////////////////////////


//------------------------------------------------------------------------------
// Simple, one cell-pair-at-a-time diffusion (in FFT space)
// --> used for initial diff.
//------------------------------------------------------------------------------

__global__  void diffusion_evolution(COMPLEX *rhoAf, COMPLEX *rhoAg, double *cor1)
{
    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(idx<N2)
    {

    rhoAg[idx].re = cor1[idx]*rhoAf[idx].re;
    rhoAg[idx].im = cor1[idx]*rhoAf[idx].im;

    }
};

//------------------------------------------------------------------------------
// Batched FFT-based diffusion
// --> used for time-step diff.
//------------------------------------------------------------------------------
__global__ void diffusion_evolution_many(COMPLEX *rhoAf, COMPLEX *rhoAg, double *cor1, int pairs){ //,int N2_loc
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    int pair  = blockIdx.y;

    if (local >= N2 || pair >= pairs) return;

    int idx = pair * N2 + local;

    rhoAg[idx].re = cor1[local] * rhoAf[idx].re;
    rhoAg[idx].im = cor1[local] * rhoAf[idx].im;
}



//////////////////////////////////////////////////////////////////
//////////      Transalte Complex to real fields --> for mass     
/////////////////////////////////////////////////////////////////
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


///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
/////////////////       CENTERS / SHIFTS  / WORLD CANVAS    ///////////////////////////
//////////////////////////////////////////////////////////////////////////


//------------------------------------------------------------------------------
// Compute current center-of-mass (COM) x/y 
// instead of "sum" --> COM on a circle throgh circle angle mapping back to x/y (because periodical local box)
//------------------------------------------------------------------------------
__global__ void compute_circular_com( const COMPLEX* __restrict__ rhoA, double* __restrict__ comx, double* __restrict__ comy,int nCells)
{
     int cell = blockIdx.x;
    int tid  = threadIdx.x;

    if (cell >= nCells) return;

    __shared__ double sw[BLOCK_SIZE];
    __shared__ double scx[BLOCK_SIZE];
    __shared__ double ssx[BLOCK_SIZE];
    __shared__ double scy[BLOCK_SIZE];
    __shared__ double ssy[BLOCK_SIZE];

    double sum_w  = 0.0;
    double sum_cx = 0.0;
    double sum_sx = 0.0;
    double sum_cy = 0.0;
    double sum_sy = 0.0;

    const double L = d_Lcell;
    const double twopi_over_N = 2.0 * M_PI / (double)N;

    int pair = cell / 2;
    int part = cell % 2;
    int off  = pair * N2;

    for (int idx = tid; idx < N2; idx += blockDim.x) {
        int i = idx / N;
        int j = idx % N;

        double w;
        if (part == 0){
            w=  rhoA[off + idx].re ;
        }
        else{

            w=rhoA[off + idx].im;
        } 

        sum_w += w;

        // i=0 -> -pi, i=N/2 -> 0
        double thx = twopi_over_N * (double)i - M_PI;
        double thy = twopi_over_N * (double)j - M_PI;

        double sx, cxv, sy, cyv;
        sincos(thx, &sx, &cxv);
        sincos(thy, &sy, &cyv);

        sum_cx += w * cxv;
        sum_sx += w * sx;
        sum_cy += w * cyv;
        sum_sy += w * sy;
    }

    sw[tid]  = sum_w;
    scx[tid] = sum_cx;
    ssx[tid] = sum_sx;
    scy[tid] = sum_cy;
    ssy[tid] = sum_sy;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sw[tid]  += sw[tid + s];
            scx[tid] += scx[tid + s];
            ssx[tid] += ssx[tid + s];
            scy[tid] += scy[tid + s];
            ssy[tid] += ssy[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double W = sw[0];
        if (W <= 1e-14) {
            comx[cell] = 0.0;
            comy[cell] = 0.0;
            return;
        }
        
        // angle and map back 
        double angx = atan2(ssx[0], scx[0]);
        double angy = atan2(ssy[0], scy[0]);

        comx[cell] = (angx / (2.0 * M_PI)) * L;
        comy[cell] = (angy / (2.0 * M_PI)) * L;
    }
}


//------------------------------------------------------------------------------
// Apllies COM shift to field (recenters box so box middle = field COM)
//------------------------------------------------------------------------------
__global__ void field_recenter_shift(const COMPLEX* __restrict__ rhoIN,COMPLEX* __restrict__ rhoOUT,
    const double* __restrict__ comx,const double* __restrict__ comy,
    int nCells)
{
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;

    //indices, offsets...
    int pair = blockIdx.y;
    int nPairs= nCells/2;
    int off = pair * N2;
    int cell_re = 2 * pair;
    int cell_im = 2 * pair + 1;
    int i = idx / N;
    int j = idx % N;

    if (idx >= N2 || pair >= nPairs) return;

    const double dx  = d_dx;
    const double thr = 2.0 * dx;

    // RE cell
    if (cell_re < nCells) {
        double cx_loc = comx[cell_re];
        double cy_loc = comy[cell_re];

        if (fabs(cx_loc) <= thr && fabs(cy_loc) <= thr) {
            rhoOUT[off + idx].re = rhoIN[off + idx].re;
        } 
        else{
            //shift w. the weighted sampling (full + fraction split)
            double sx_f = cx_loc / dx;
            double sy_f = cy_loc / dx;

            int sx = (int)floor(sx_f);
            int sy = (int)floor(sy_f);

            double fx = sx_f - (double)sx;
            double fy = sy_f - (double)sy;

            int x0 = pbc(i + sx);
            int y0 = pbc(j + sy);
            int x1 = pbc(x0 + 1);
            int y1 = pbc(y0 + 1);

            double v00 = rhoIN[off + x0 * N + y0].re;
            double v10 = rhoIN[off + x1 * N + y0].re;
            double v01 = rhoIN[off + x0 * N + y1].re;
            double v11 = rhoIN[off + x1 * N + y1].re;

            double wx0 = 1.0 - fx;
            double wy0 = 1.0 - fy;

            rhoOUT[off + idx].re =
                (wx0 * wy0) * v00 +
                (fx  * wy0) * v10 +
                (wx0 * fy ) * v01 +
                (fx  * fy ) * v11;
        }
    }

     // IM cell
    if (cell_im < nCells) {

        double cx_loc = comx[cell_im];
        double cy_loc = comy[cell_im];

        if (fabs(cx_loc) <= thr && fabs(cy_loc) <= thr){
            rhoOUT[off + idx].im = rhoIN[off + idx].im;
        } 
        else{
        
        //shift

            double sx_f = cx_loc / dx;
            double sy_f = cy_loc / dx;

            int sx = (int)floor(sx_f);
            int sy = (int)floor(sy_f);

            double fx = sx_f - (double)sx;
            double fy = sy_f - (double)sy;

            int x0 = pbc(i + sx);
            int y0 = pbc(j + sy);
            int x1 = pbc(x0 + 1);
            int y1 = pbc(y0 + 1);

            double v00 = rhoIN[off + x0 * N + y0].im;
            double v10 = rhoIN[off + x1 * N + y0].im;
            double v01 = rhoIN[off + x0 * N + y1].im;
            double v11 = rhoIN[off + x1 * N + y1].im;

            double wx0 = 1.0 - fx;
            double wy0 = 1.0 - fy;

            rhoOUT[off + idx].im =
                (wx0 * wy0) * v00 +
                (fx  * wy0) * v10 +
                (wx0 * fy ) * v01 +
                (fx  * fy ) * v11;
        }
    }

    
}

//------------------------------------------------------------------------------
// Apllies COM shift to cx/cy (global box centers trackers) 
//------------------------------------------------------------------------------
__global__ void center_shift(double* __restrict__ cx,double* __restrict__ cy,const double* __restrict__ comx,const double* __restrict__ comy, int nCells)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;

    if (cell >= nCells) return;

    const double dx  = d_dx;
    const double thr = 2.0 * dx;

    double cx_loc = comx[cell];
    double cy_loc = comy[cell];

    if (fabs(cx_loc) <= thr && fabs(cy_loc) <= thr) return;

    cx[cell] += cx_loc;
    cy[cell] += cy_loc;
}
//------------------------------------------------------------------------------
// FILE WRITE HELPER
//Make NxN boxes written into big world dim
//------------------------------------------------------------------------------

__global__ void render_cell_world(const COMPLEX* __restrict__ rhoA,double* __restrict__ out,
    const double* __restrict__ cx,const double* __restrict__ cy,
    int cell,double dx)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int M2 = MOUT * MOUT;

    if (idx >= M2) return;

    int I = idx / MOUT;
    int J = idx % MOUT;
    double X = I * dx;
    double Y = J * dx;
    int pair = cell / 2;
    int part = cell % 2;
    int off = pair * N2;
    double Lcell = N * dx;
    double half = 0.5 * Lcell;

    double x_rel = X - cx[cell];
    double y_rel = Y - cy[cell];

    //if the world pixel is not the box
    if (x_rel < -half || x_rel >= half ||
        y_rel < -half || y_rel >= half) {
        out[idx] = 0.0;
        return;
    }

    double u = x_rel / dx + 0.5 * N;
    double v = y_rel / dx + 0.5 * N;

    //same as in the other sampling with the full + fraction part
    
    int i0 = (int)floor(u);
    int j0 = (int)floor(v);

    //if not in the local box
    if (i0 < 0 || j0 < 0 || i0 >= N-1 || j0 >= N-1) {
        out[idx] = 0.0;
        return;
    }


    double fu = u - (double)i0;
    double fv = v - (double)j0;

    double v00, v10, v01, v11;

    if (part == 0) {
        v00 = rhoA[off + i0*N     + j0    ].re;
        v10 = rhoA[off + (i0+1)*N + j0    ].re;
        v01 = rhoA[off + i0*N     + (j0+1)].re;
        v11 = rhoA[off + (i0+1)*N + (j0+1)].re;
    } else {
        v00 = rhoA[off + i0*N     + j0    ].im;
        v10 = rhoA[off + (i0+1)*N + j0    ].im;
        v01 = rhoA[off + i0*N     + (j0+1)].im;
        v11 = rhoA[off + (i0+1)*N + (j0+1)].im;
    }

    out[idx] =
        (1.0 - fu) * (1.0 - fv) * v00
      + fu         * (1.0 - fv) * v10
      + (1.0 - fu) * fv         * v01
      + fu         * fv         * v11;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////
//// SPAWNUING STUFF //////////////////
////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////


//------------------------------------------------------------------------------
// Given center and radius^2 --> make cell area
//------------------------------------------------------------------------------

__host__ __device__ double make_cell(int i, int j,double ci, double cj, double dx, double rad)
{   

    double x = i - ci; 
    double y = j - cj;

    if(dx*dx*(x*x + y*y) <= rad)
        return 1.0;
    else
        return 0.0;
};



//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//===========================================================================
// MAIN SIMULATION SETUP + LOOP
//===========================================================================


//unity *folder* *rows* *bundles (per row)* *repulson* *FMI* *SDK* *diffusion*
//unity cool_run 3 6 5 2 7 0.7

//===========================================================================

int main(int argc, char* argv[] )
{   
    if (argc < 4) { //8 //11pos
        cerr << "use: cell_test *folder* *rows* *bundles* *repulson* *FMI* *SDK* *diffusion";
        return 1;
    }
    //folder name
    std::string number;
    for (int i = 1; i < argc; i++)
    {
        number += argv[i];
        if (i < argc - 1) number += "_";
    }

 // bundle-row size 
    int rows=std::stoi(argv[2]); 
    int bundles=std::stoi(argv[3]);
    int nCells=rows*bundles*6;
    int nPairs=nCells/2;
    // Model Parameters

    double repulsion = std::stod(argv[4]);
    double FMI=std::stod(argv[5]);
    double SDK=std::stod(argv[6]);
    double D_r= std::stod(argv[7]);


    // Sim. sizes - space, time, event timer
    double  L=40.0, dt=0.001; //
    double dx=L/(double)N;	
    const double Lworld = MOUT*dx ;  // big world /  global canvas
    double dx2=dx*dx;
    double dk, scale = 1.0*N*N, mu=0.1;//dont touch these(?)
    dk=2.*M_PI/L;

    //cell size
    double r0;
    double r02;
    r0= 12; //radius  
    r02=r0*r0;
    double cutoff =2.0*r0; 

    int steps, blind;
    double time=0.,timestart=0.0;
    blind = 1000;
    int timing =15000;

    int running_r=0;
    int running_k=0;

    steps = (rows)*timing; //+3/+2 cause of decay offset (buffer at back)
    int* events = new int[rows+2];

    for (int i = 0; i < (rows+2); i++)
    {
        events[i] = i *timing+timing;
    }


    const double growth_rate = 25;
    double gs =((N / 256.0)*(100/L))*((N / 256.0)*(100/L));
    double growth_time;

    if(timing>15000){

        growth_time = 15000;
    }
    else {
        growth_time=timing;

    }

    // int ox3 =std::stoi(argv[4]);
    // int oy3=std::stoi(argv[5]);
    // int ox4=std::stoi(argv[6]);
    // int oy4=std::stoi(argv[7]);
    // int ox1 =std::stoi(argv[8]);
    // int oy1=std::stoi(argv[9]);
    // int ox6=std::stoi(argv[10]);
    // int oy6=std::stoi(argv[11]);

    std::mt19937 gen(1);
    std::uniform_int_distribution<int> dist(0, 3);
    
    // int ox3 =22; 
    // int oy3=5;
    // int ox4=12; 
    // int oy4=15;
    // int ox1 =15;
    // int oy1=15; 
    // int ox6=20;
    // int oy6=7; 
    

    // fields and update diag. storage
    COMPLEX *rhoA,*GrhoA,*GNrhoA; //all fields -stored as pairs (complex)
    COMPLEX *fd_ini,*Gfd;
    double *fd,*Gfd_single;

    double *rho,*Grho; //one field->for output
    double *cor1,*Gcor1; //the diff-fft matrix
    double *delta,*Gdelta; //each field/cell own delta
    double *mass0, *Gmass0,*totmass,*s_time; //current mass, target mass, time

    
    
    //For local-box: gradients and centers
    COMPLEX *Ggradx, *Ggrady;
    double *Gcomx, *Gcomy;
    double *cx, *cy;
    double *Gcx, *Gcy;

    ////////////////////////////////
    ///    CUDA MEMORY-INITIALIZATION 
    /////////////////////////
    cublasHandle_t handle;
    cufftHandle Gfftplan;
    cufftHandle Gfftplan_batch;

    int memN2c, memN2r;

    // pass to global
    cudaMemcpyToSymbol(d_dx, &dx, sizeof(double));
    cudaMemcpyToSymbol(d_cutoff, &cutoff, sizeof(double));
    cudaMemcpyToSymbol(d_Lcell, &L, sizeof(double));

    /////////////////////////  MEMOPRY ALLOC (host/cpu) ////////////////
    // initialize fields, masses, coms etc..
    rho     = new double[N2];
    cor1    = new double[N2];
    delta   = new double[nCells];
    mass0   = new double[nCells];
    totmass = new double[nCells];
    s_time  = new double[nCells];
    cx = new double[nCells];
    cy = new double[nCells];

    // one COMPLEX field per pair
    rhoA = new COMPLEX[nPairs * N2];
    //for update
    fd = new double[N2];
    fd_ini = new COMPLEX[nPairs*N2];

    //GPU-computing memory dimesions
    dim3 dGRID, dBLOCK;
    int GPUID = 0;
    cudaSetDevice(GPUID);
    dBLOCK = dim3(N,1,1);
    dGRID  = dim3(N,1,1);

    dim3 blockPair(BLOCK_SIZE);
    dim3 gridPair((N2 + BLOCK_SIZE - 1) / BLOCK_SIZE, nPairs);
    dim3 gridCells((nCells + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    ////for output writing/////
    int M2 = MOUT * MOUT;

    double *rho_world;
    double *Grho_world;

    rho_world = new double[M2];
    cudaMalloc((void**)&Grho_world, M2 * sizeof(double));

    dim3 blockWorld(BLOCK_SIZE);
    dim3 gridWorld((M2 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    ///////////////////////////////////// MEMOPRY ALLOC (gpu)////////////////////////////
    memN2c=N2*sizeof(COMPLEX);
    memN2r=N2*sizeof(double);

    //complex arrays on GPU
    cudaMalloc((void**)&GrhoA, nPairs*memN2c);
    cudaMalloc((void**)&GNrhoA,nPairs*memN2c);
    cudaMalloc((void**)&Gfd, nPairs*memN2c);

    //real arrays on GPU
    cudaMalloc((void**)&Grho,memN2r);
    cudaMalloc((void**)&Gfd_single,memN2r);
    cudaMalloc((void**)&Gcor1,memN2r);
    cudaMalloc((void**)&Gdelta,nCells*sizeof(double));
    cudaMalloc((void**)&Gmass0,nCells*sizeof(double));

    //local box gradients and centers, coms
    cudaMalloc((void**)&Gcx, nCells * sizeof(double));
    cudaMalloc((void**)&Gcy, nCells * sizeof(double));
    cudaMalloc((void**)&Ggradx, nPairs * memN2c);
    cudaMalloc((void**)&Ggrady, nPairs * memN2c);
    cudaMalloc((void**)&Gcomx, nCells * sizeof(double));
    cudaMalloc((void**)&Gcomy, nCells * sizeof(double));
        

    //////////////////////////// FFT ////////////////////////////////////
    // Create CUDA (and FFT) plans
    cufftPlan2d(&Gfftplan, N, N, CUFFT_Z2Z); //for ini/singular controlled diffusion FFT
    cublasCreate(&handle); // for sums

    //bathced FFT plan for all pairs
    ////////
    int n[2] = {N, N};
    int inembed[2]  = {N, N};
    int onembed[2] = {N, N};

    cufftPlanMany(&Gfftplan_batch,2,n,
        inembed,  1, N2,
        onembed, 1, N2,
        CUFFT_Z2Z,
        nPairs
    );


     // calculate q's 
    double qx[N],qy[N],qsq;
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
    

    //initial diff smooth matrix - stringer by blind step size   
    double *cor1_init, *Gcor1_init;

    cor1_init = new double[N2];
    cudaMalloc((void**)&Gcor1_init, memN2r);


    for(int i=0; i<N; i++)
        for (int j=0; j<N; j++)
        {
            qsq = qx[i]*qx[i]+qy[j]*qy[j];
            cor1_init[j+N*i] = exp(-(blind*dt)*D_r*qsq)/scale;
        }

    
    /////////////////////////////////////
    // SETUP OVER HERE 
    /////////////////////////////////////


    /////////////////////////////////////
    // INITIALIZATION OF VARIABLES AND FIELDS
    /////////////////////////////////////   

    /////////////////////////////////////////// 0 ini ////////////////////////////////
     // initialize everything to zero
    for (int p= 0;p < nPairs*N2; p++)
    {
        rhoA[p].re=0.0; //fields
        rhoA[p].im=0.0;
        fd_ini[p].re=0.0; //updates 
        fd_ini[p].im=0.0;

    }

    for (int c = 0;c < nCells;c++) 
    {
        delta[c]=0.5;
        mass0[c]=0.0;
        totmass[c]=0.0;
        s_time[c]=0.0; 

        cx[c] = 0.0;
        cy[c] = 0.0;
    }

    ///////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////// INI SPAWN /////////////////////////////////
    if (true){
        const std::string fname = "healthy_half_turned_test_0.txt";

        const double world_center_i = 256.0;   
        const double world_center_j = 256.0;
        const double global_scale   = 18.0;

        //read file - store entries - spawn
    
        double *xs    = new double[nCells];
        double *ys    = new double[nCells];
        int    *pairs = new int[nCells];
        bool   *parts   = new bool[nCells];
        int n = 0;
        
            std::ifstream file(fname);
            std::string id, type, line;
            double x, y, z; char c;

            while (std::getline(file, line)) {
                if (line.empty()) continue;

                //read from file
                std::stringstream ss(line);
                ss >> id >> type;
                ss >> c >> y >> c >> x >> c >> z >> c;   
                int row    = std::stoi(id.substr(0, 1));
                int bundle = std::stoi(id.substr(3));
                int t      = std::stoi(type.substr(1));

                if (!(t >= 1 && t <= 6 &&row >= 0 && row < rows && bundle >= 0 && bundle < bundles)) continue; // do not look at PRs which does not match type (there are R8s) or are over the set bundle/row

                int base = (row * bundles + bundle) * 3;   //first pair indx of this bundle

                int pair; bool part;
                if      (t == 2) { pair = base + 0; part = true;  }  // R2
                else if (t == 5) { pair = base + 0; part = false; }  // R5
                else if (t == 3) { pair = base + 1; part = true;  }  // R3
                else if (t == 4) { pair = base + 1; part = false; }  // R4
                else if (t == 1) { pair = base + 2; part = true;  }  // R1
                else if (t == 6) { pair = base + 2; part = false; }  // R6
                else continue;

                xs[n]    = x;
                ys[n]    = y;
                pairs[n] = pair;
                parts[n]   = part;
                n++;
            }
        

        if (n > 0) { //if i did collec PRs

            //file center
            double fcx = 0.0, fcy = 0.0;
            for (int k = 0; k < n; k++) { fcx += xs[k]; fcy += ys[k]; }
            fcx /= n;
            fcy /= n;

            //spawn so file center= world center
            for (int k = 0; k < n; k++) {
                int pair = pairs[k];
                bool part  = parts[k];
                int cell = 2 * pair + (part ? 0 : 1);
                if (pair < 0 || pair >= nPairs || cell < 0 || cell >= nCells) continue;

                // x/y swapped + flipped to match the sim frame
                cx[cell] = (world_center_i - global_scale * (xs[k] - fcx)) * dx;
                cy[cell] = (world_center_j + global_scale * (ys[k] - fcy)) * dx;

                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++) {
                        int idx = pair * N2 + (j + N * i);
                        
                        if (part){
                            rhoA[idx].re = make_cell(i, j, N/2, N/2, dx, r02/4);
                        }
                        else{ 
                            rhoA[idx].im = make_cell(i, j, N/2, N/2, dx, r02/4);
                        }
                    }
            }
        }

        delete[] xs;
        delete[] ys;
        delete[] pairs;
        delete[] parts;
    }
    
    /////////////////////////////////////////////////////////////////////////////////////////////  
 ///// INI UPDATE/TRANSFER ////////////////   
    // pass initialization to GPU, from here everything lives there (except occasional outputs)    
    cudaMemcpy(GrhoA,rhoA, nPairs * memN2c, cudaMemcpyHostToDevice);
    cudaMemcpy(Gcor1,cor1,  memN2r, cudaMemcpyHostToDevice);
    cudaMemcpy(Gcor1_init, cor1_init, memN2r, cudaMemcpyHostToDevice);
    cudaMemcpy(Gcx, cx, nCells * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Gcy, cy, nCells * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Gdelta,delta, nCells * sizeof(double),cudaMemcpyHostToDevice);

    cudaMemcpy(Gfd,fd_ini, nPairs * memN2c, cudaMemcpyHostToDevice); //pass update memory to gpu

    //initial diffusion of ALL
    for(int p =0;p< nPairs;p++){
    

                // GrhoA-> GNrhoA->GrhoA
                cufftExecZ2Z(Gfftplan, (cufftDoubleComplex*)(GrhoA + p*N2),(cufftDoubleComplex*)(GNrhoA + p*N2),CUFFT_FORWARD);
                diffusion_evolution<<<dGRID, dBLOCK>>>( GNrhoA + p*N2, GNrhoA + p*N2, Gcor1_init);
                cufftExecZ2Z(Gfftplan,(cufftDoubleComplex*)(GNrhoA + p*N2),(cufftDoubleComplex*)(GrhoA + p*N2),CUFFT_INVERSE);
        }       

    // initial masses for all cells
    for (int p = 0;p <nPairs; p++){

        complexToReal1<<<dGRID, dBLOCK>>>(GrhoA + p*N2, Grho);
        cublasDasum(handle, N2, Grho, 1, &mass0[2*p]);

        complexToReal2<<<dGRID, dBLOCK>>>(GrhoA + p*N2, Grho);
        cublasDasum(handle, N2, Grho, 1, &mass0[2*p+1]);
    }

    // pass mass to GPU
    cudaMemcpy(Gmass0,mass0, nCells * sizeof(double),cudaMemcpyHostToDevice);
    

  
    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////
    //--------------------------------------------------------------------------------------
    // timesteps
    //--------------------------------------------------------------------------------------
    //////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////// 
    //////////////////////////////////////////////////////////
        std::ostringstream debug_folder;
        debug_folder << "out_debug_" << number;

    //     std::ostringstream mass_folder;
    //     mass_folder << "out_mass_" << number;

    //     std::ostringstream upd_folder;
    //     upd_folder << "out_upd_" << number;

        std::filesystem::remove_all(debug_folder.str());
    //     std::filesystem::remove_all(mass_folder.str());
    //     std::filesystem::remove_all(upd_folder.str());

        std::filesystem::create_directories(debug_folder.str());
    //     std::filesystem::create_directories(mass_folder.str());
    //     std::filesystem::create_directories(upd_folder.str());
    // /////////////////////////////////////////////

    for(int k=0; k<steps+1; k++)
    {   
        
    
        time += dt;

 //deliver updated delta+mass=centers vec
        cudaMemcpy(cx, Gcx, nCells * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(cy, Gcy, nCells * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Gdelta, delta, nCells*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Gmass0, mass0, nCells*sizeof(double), cudaMemcpyHostToDevice);

        //gradients
        // current field, x/y Gradients to update, n. of cell pairs (num. of Complex)
        compute_gradients<<<gridPair, blockPair>>>(GrhoA,Ggradx,Ggrady,nPairs);

        /////////////
        /// UPDATE --> in comes ft
        ////////////
        /// 1. f* = ft + dt*update

        //func handles pairs as i need all cells in one func for interactions (cause adh/rep depends on everyone!)
        // current field, field* (gets further updated in fft), time step size, current step, parameters adh-rep-doublewell, 
        // current mass, number of Cells (overall) and bundles per row, current row, inter-event time, current event timer,
        // update memorizer
        // Gardients and centers
        nonlin_update<<<gridPair, blockPair>>>(GrhoA, GNrhoA, dt, k, SDK, FMI, repulsion, Gdelta, Gmass0, nCells, bundles, running_r, timing, running_k, Gfd, Ggradx, Ggrady, Gcx, Gcy); //

        /// 1. ft+1 = FFT-diffusion update(f*)
        //diffusion in batch on all cells
        cufftExecZ2Z(Gfftplan_batch, (cufftDoubleComplex*)(GNrhoA),(cufftDoubleComplex*)(GrhoA),CUFFT_FORWARD);
        diffusion_evolution_many<<<gridPair, blockPair>>>(GrhoA,GNrhoA,Gcor1,nPairs);
        cufftExecZ2Z(Gfftplan_batch,(cufftDoubleComplex*)(GNrhoA ),(cufftDoubleComplex*)(GrhoA),CUFFT_INVERSE);


       /////////////
        /// UPDATE END --> ft+1 from here
        ////////////

        //RECENTERING
        // 1. compute offsets
        // 2. recenter fields
        // 3. recenter field centers relative to canvas
        compute_circular_com<<<nCells, BLOCK_SIZE>>>(GrhoA, Gcomx, Gcomy, nCells);
        field_recenter_shift<<<gridPair, blockPair>>>(GrhoA, GNrhoA, Gcomx, Gcomy, nCells);
        center_shift<<<gridCells, blockPair>>>(Gcx, Gcy, Gcomx, Gcomy, nCells);

        // memory update
        cudaMemcpy(GrhoA, GNrhoA, nPairs * memN2c, cudaMemcpyDeviceToDevice);
        cudaMemcpy(cx, Gcx, nCells * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(cy, Gcy, nCells * sizeof(double), cudaMemcpyDeviceToHost);

        //  mass + output --> loop over each COMPLEX pair
        for (int p = 0; p < nPairs; p++) {

            if (mass0[2*p] == 0.0 || mass0[2*p+1] == 0.0) {
                continue;
            }

            ////////////// re //////////////////////////
            complexToReal1<<<dGRID, dBLOCK>>>(GrhoA + p*N2, Grho);
            cublasDasum(handle, N2, Grho, 1, &totmass[2*p]);

            if (k % blind == 0) {
                int cell = 2*p;

                render_cell_world<<<gridWorld, blockWorld>>>(GrhoA, Grho_world, Gcx, Gcy, cell, dx);
                cudaMemcpy(rho_world, Grho_world, M2 * sizeof(double), cudaMemcpyDeviceToHost);

                ofstream outf1;
                ostringstream name1;
                name1 << debug_folder.str() << "/rho" << cell << "-t" << time-dt << ".txt";
                outf1.open(name1.str().c_str());

                for (int i = 0; i < MOUT; i++) {
                    for (int j = 0; j < MOUT; j++) {
                        double v = rho_world[j + MOUT*i];
                        if (fabs(v) > 0.001) { outf1 << v << " "; }
                        else { outf1 << 0.0 << " "; }
                    }
                    outf1 << endl;
                }
                outf1.close();

                // double diff_m = fabs(mass0[cell] - totmass[cell]);

                // ofstream outf2;
                // ostringstream name2;
                // name2 << mass_folder.str() << "/rho" << cell << ".txt";
                // outf2.open(name2.str().c_str(), std::ios::app);
                // outf2 << time-dt << " " << diff_m << " ";
                // outf2 << endl;

                // complexToReal1<<<dGRID, dBLOCK>>>(Gfd + p*N2, Gfd_single);
                // cudaMemcpy(fd, Gfd_single, memN2r, cudaMemcpyDeviceToHost);

                // ofstream outf3;
                // ostringstream name3;
                // name3 << upd_folder.str() << "/updatediff" << cell << "-t" << time-dt << ".txt";
                // outf3.open(name3.str().c_str());
                // for (int i = 0; i < N; i++) {
                //     for (int j = 0; j < N; j++) {
                //         outf3 << fd[j + N*i] << " ";
                //     }
                //     outf3 << endl;
                // }
                // outf3.close();
            }

            ////////////// im //////////////////////////
            complexToReal2<<<dGRID, dBLOCK>>>(GrhoA + p*N2, Grho);
            cublasDasum(handle, N2, Grho, 1, &totmass[2*p+1]);

            // if (k % blind == 0) {
            //     int cell = 2*p + 1;

            //     double diff_m = fabs(mass0[cell] - totmass[cell]);

            //     ofstream outf1;
            //     ostringstream name1;
            //     name1 << mass_folder.str() << "/rho" << cell << ".txt";
            //     outf1.open(name1.str().c_str(), std::ios::app);
            //     outf1 << time-dt << " " << diff_m << " ";
            //     outf1 << endl;
            // }

            // ///////////////// growth / delta //////////////////////////////
            // if (k >= s_time[2*p] && k < s_time[2*p] + growth_time) {
            //     mass0[2*p]   += growth_rate * dt * gs;
            //     mass0[2*p+1] += growth_rate * dt * gs;
            // }

            delta[2*p]   = 0.5 + mu * (totmass[2*p]   - mass0[2*p]);
            delta[2*p+1] = 0.5 + mu * (totmass[2*p+1] - mass0[2*p+1]);

            // output im
            if (k % blind == 0) {
                int cell = 2*p + 1;

                render_cell_world<<<gridWorld, blockWorld>>>(GrhoA, Grho_world, Gcx, Gcy, cell, dx);
                cudaMemcpy(rho_world, Grho_world, M2 * sizeof(double), cudaMemcpyDeviceToHost);

                ofstream outf1;
                ostringstream name1;
                name1 << debug_folder.str() << "/rho" << cell << "-t" << time-dt << ".txt";
                outf1.open(name1.str().c_str());

                for (int i = 0; i < MOUT; i++) {
                    for (int j = 0; j < MOUT; j++) {
                        double v = rho_world[j + MOUT*i];
                        if (fabs(v) > 0.001) { outf1 << v << " "; }
                        else { outf1 << 0.0 << " "; }
                    }
                    outf1 << endl;
                }
                outf1.close();

                // complexToReal2<<<dGRID, dBLOCK>>>(Gfd + p*N2, Gfd_single);
                // cudaMemcpy(fd, Gfd_single, memN2r, cudaMemcpyDeviceToHost);

                // ofstream outf2;
                // ostringstream name2;
                // name2 << upd_folder.str() << "/updatediff" << cell << "-t" << time-dt << ".txt";
                // outf2.open(name2.str().c_str());

                // for (int i = 0; i < N; i++) {
                //     for (int j = 0; j < N; j++) {
                //         outf2 << fd[j + N*i] << " ";
                //     }
                //     outf2 << endl;
                // }
                //  outf2.close();
    }
}

    }

    delete[] rhoA;
    delete[] rho;
    delete[] cor1;
    delete[] delta;
    delete[] mass0;
    delete[] totmass;
    delete[] events;

    cudaFree(GrhoA);
    cudaFree(GNrhoA);
    cudaFree(Grho);
    cudaFree(Gcor1);
    cudaFree(Gdelta);



    cublasDestroy(handle);
    cufftDestroy(Gfftplan);

    return 0;
}