#include "mex.h"
#include "cuda.h"
#include "cusparse.h"
#include "curand.h"
#include "stdlib.h"
#include <algorithm>
#include "curand_kernel.h"
#include "time.h"
#define HOG_THREAD_CNT 1
#define NITER 200

using namespace std;

struct Params {
float *Aval;
int *ArowCSR;
int *Acol;
float *b;
float *x;
int N;
int D;
float *devData;
curandState *state;
};
//KERNEL CODE FOR RANDOM NUMBER GENERATION
__global__ void  generate_normal_kernel(struct Params params)
{
int i,j,k;
int e;
float grad;
curandState localState;


for (i = threadIdx.x; i < HOG_THREAD_CNT; i += HOG_THREAD_CNT)
 {
        localState = params.state[threadIdx.x];
        for (j = 0; j < NITER ; j ++) 
        { 
            e = floorf(curand_uniform(&localState)*params.N);
            //params.devData[i*NITER + j]=e; 
            grad = 0.0;
            for(k = params.ArowCSR[e]; k<params.ArowCSR[e+1]; k++)
            {
               grad += ((params.Aval[k])*(params.x[params.Acol[k]]));
            }
            grad = grad - params.b[e];                                // params.Aval[params.ArowCSR[e-1]-1] corresponds to the first element of e-th row
            //params.devData[i*NITER + j] = grad;
            
            for(k = params.ArowCSR[e]; k<params.ArowCSR[e+1]; k++)
            {
                params.x[params.Acol[k]] = params.x[params.Acol[k]] - 0.1*(grad*params.Aval[k] +   0.1*params.x[params.Acol[k]]);
            }    
           
            //params.devData[i*NITER + j] = params.x[params.Acol[k]] - 1.0*(grad*params.Aval[k] +   1.0*params.x[params.Acol[k]]);
        }     
 }
} 

// KERNEL CODE FOR RANDOM NUMBER SEED INITIALIZATION
__global__ void setup_kernel(curandState *state, unsigned int seed)
{
    int id = threadIdx.x;
    /* Each thread gets different seed, a different sequence
       number, no offset */
    curand_init(seed, id, 0, &state[id]);
}

void mexFunction( int nlhs, mxArray *plhs[],
 int nrhs, const mxArray *prhs[])
{
 struct Params funcparams;
 int dims0[1];//[2];    // For storing matrix dimensions
 int nnz,D,N;         // nnz: Number of non-zero elements in A; D: dimensions of A
 // Allocate memory for storing the matrix x,A and b
 float *x = 0;          // CPU
 float* b = 0;
 float* Aval = 0;
 int* Acol = 0;
 int* Arow = 0;
 int* ArowCSR = 0;
 
 float* gpub = 0;     // GPU
 float* gpux = 0;
 float* gpuAval = 0;       
 int* gpuAcol = 0;
 int* gpuArow = 0;
 int* gpuArowCSR = 0;
 // Allocate memory of random number generation
 curandState *state;
 float *hostData,*devData;
 // Initialize Cusparse
 cusparseStatus_t status;
 cusparseHandle_t handle=0;
 cusparseMatDescr_t descr=0;
 
 // Validate Inputs
 if (nrhs != 4) {
 mexErrMsgTxt("engine requires 4 input arguments");
 } else if (nlhs != 1) {
 mexErrMsgTxt("engine requires 1 output argument");
 }
 if ( !mxIsSingle(prhs[2])|| !mxIsSingle(prhs[3]) ) {
 mexErrMsgTxt("A and b must be single precision");
 }
 if ( !mxIsUint32(prhs[0])|| !mxIsUint32(prhs[0]) ) {
 mexErrMsgTxt("Rows and colums of the sparse matrix should be unsigned integers");
 }
 // Get the various dimensions to this problem.
 nnz = mxGetM(prhs[0]); /* Number of nnz in sparse matrix */
 // Fetch Inputs
 Arow =  (int*) mxGetData(prhs[0]);
 Acol =  (int*) mxGetData(prhs[1]);
 Aval =  (float*) mxGetData(prhs[2]);
 D = *max_element(Acol, Acol+nnz) + 1;
 N = *max_element(Arow, Arow+nnz) + 1;
 mexPrintf("N = %d, D = %d \n",N,D);
 b = (float*) mxGetData(prhs[3]);
 
 //This section is for testing
 for (int i=0; i<nnz; i++)
    mexPrintf("%d \t %d \t %f \n ", Arow[i], Acol[i], Aval[i]);
 
 dims0[0]=D;
 //dims0[1]=2*NX;
 plhs[0] = mxCreateNumericArray(1,dims0,mxSINGLE_CLASS,mxREAL);
 x = (float*) mxGetData(plhs[0]);
 // Set random seed for initializing x
 srand (time(NULL));
 for(int i = 0; i<D ; i++)
    x[i] = (float)(i+1);//x[i] = (rand()%100)/100.0; //testing phase
// Testing section ends here
 
 // Allocating space on the cpu for random numbers
 hostData = (float*) malloc ((HOG_THREAD_CNT)*NITER*sizeof(hostData[0]));
  
 // Allocating space on GPU for random Numbers
 cudaMalloc((void **) &devData,HOG_THREAD_CNT*NITER*sizeof(devData[0]));
 cudaMalloc((void **)&state, HOG_THREAD_CNT*sizeof(state[0]));
  
 // Allocating Aval, Acol, Arow, ArowCSR on GPU
 cudaMalloc ((void **)&gpuAval, nnz*sizeof(gpuAval[0]));
 cudaMalloc ((void **)&gpuAcol, nnz*sizeof(gpuAcol[0]));
 cudaMalloc ((void **)&gpuArow, nnz*sizeof(gpuArow[0]));
 cudaMalloc ((void **)&gpuArowCSR, (N+1)*sizeof(gpuArowCSR[0]));
 cudaMalloc ((void **)&gpub, N*sizeof(gpub[0]));
 cudaMalloc ((void **)&gpux, D*sizeof(gpux[0]));
 // Copying Aval, Acol, Arow from host to device
 cudaMemcpy (gpuAval, Aval, nnz*sizeof(Aval[0]), cudaMemcpyHostToDevice);
 cudaMemcpy (gpuAcol, Acol, nnz*sizeof(Acol[0]), cudaMemcpyHostToDevice);
 cudaMemcpy (gpuArow, Arow, nnz*sizeof(Arow[0]), cudaMemcpyHostToDevice);
 cudaMemcpy (gpub, b, N*sizeof(b[0]), cudaMemcpyHostToDevice);
 cudaMemcpy (gpux, x, D*sizeof(x[0]), cudaMemcpyHostToDevice);
 
 /* initialize cusparse library */ 
 status= cusparseCreate(&handle); 
 if (status != CUSPARSE_STATUS_SUCCESS)
 mexPrintf("CUSPARSE Library initialization failed"); 
 /* create and setup matrix descriptor */ 
 status= cusparseCreateMatDescr(&descr);
 if (status != CUSPARSE_STATUS_SUCCESS)
 mexPrintf("Matrix descriptor initialization failed"); 
 cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
 cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO); 
 // Convert from COO to CSR 
 status = cusparseXcoo2csr(handle,gpuArow,nnz,N,gpuArowCSR,CUSPARSE_INDEX_BASE_ZERO);
 if (status != CUSPARSE_STATUS_SUCCESS) 
    mexPrintf("Conversion from COO to CSR format failed");
 
 //Copy the results of conversion
 mexPrintf("No errors so far");
 ArowCSR = (int*) malloc ((N+1)*sizeof(ArowCSR[0]));
 cudaMemcpy (ArowCSR, gpuArowCSR, (size_t)(N+1)*sizeof(ArowCSR[0]), cudaMemcpyDeviceToHost);
// if (status != cudaSuccess) 
 //   mexPrintf("Copying COO to CSR to HOST  failed");

 mexPrintf("Printing CSR Values \n");
 for (int i =0; i<(N+1); i++)
     mexPrintf("Index=%d, Value = %d \t",i,ArowCSR[i]);
//    mexPrintf("%d \t",ArowCSR[i]);
 
 // Set up Kernel for random number seed initializaton
 setup_kernel<<<1,HOG_THREAD_CNT>>>(state,time(NULL));
 mexPrintf("RANDOM NUMBER SEED INITIALIZED \n");
 
 funcparams.Aval = gpuAval;
 funcparams.ArowCSR = gpuArowCSR;
 funcparams.Acol = gpuAcol;
 funcparams.b = gpub;
 funcparams.x = gpux;
 funcparams.N = N;
 funcparams.D = D;
 funcparams.devData = devData;
 funcparams.state = state;

 
 // Generate Random Numbers
 generate_normal_kernel<<<1, HOG_THREAD_CNT>>>(funcparams);
 cudaMemcpy (hostData, devData, HOG_THREAD_CNT*NITER*sizeof(hostData[0]), cudaMemcpyDeviceToHost);
 
 // displaying results
 
 for(int i =0; i < HOG_THREAD_CNT*NITER ; i++) {
        if(i%NITER == 0)
            mexPrintf("\n");
        mexPrintf("%f\t",hostData[i]);
    }


//cudaMalloc ((void **)&cosRes, 2*NN*sizeof(cosRes[0]));
/* cudaMalloc ((void **)&Aarg, N*D*sizeof(Aarg[0]));
 cudaMalloc ((void **)&barg, N*sizeof(barg[0]));
 cudaMalloc((void **)&xRes , D*sizeof(xRes[0]));  
// Copy A, b to the GPU.
 //cudaMemcpy (cosArg, phase, NN*sizeof(phase[0]), cudaMemcpyHostToDevice);
 //cudaMemcpy (aaa, amp, NN*sizeof(aaa[0]), cudaMemcpyHostToDevice);
 cudaMemcpy (Aarg, A, N*D*sizeof(Aarg[0]), cudaMemcpyHostToDevice);
 cudaMemcpy (barg, b, N*sizeof(barg[0]), cudaMemcpyHostToDevice);
 cudaMemcpy (xRes, x, D*sizeof(xRes[0]), cudaMemcpyHostToDevice);
 funcParams.res = xRes;
 funcParams.A = Aarg;
 funcParams.b = barg;
 funcParams.N = N;
 funcParams.D = D;
 
 hog_main<<<1,HOG_THREAD_CNT>>>(funcParams);
// "A" should now be in the array pointer "cosRes" on the device.
// We'll need to copy it to A
// "aaa", "cosArg" are NY by NX, while "cosRes" is NY by 2*NX
// (although everything here is stored in linear memory)
// Copy the result, which is A, from the device to the host
 cudaMemcpy (x, xRes, D*sizeof(x[0]), cudaMemcpyDeviceToHost);
// Done! */
// Free up the allocations on the CPU
 free(hostData);
 free(ArowCSR);
 
// Free up the allocations on the GPU
 cudaFree(devData);
 cudaFree(state);
 cudaFree(gpuAval);
 cudaFree(gpuArow);
 cudaFree(gpuAcol);
 cudaFree(gpuArowCSR);
 cudaFree(gpux);
 cudaFree(gpub);
}
