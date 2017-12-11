#include <time.h>
#include <sys/time.h>
#include <chrono>
#include "predict.h"

using namespace std::chrono;

void create_g(double *G,double theta, double phi) {
    double g = -9.8;
    double mass = 38;
    double volume = 0.0117;
    double p = 1000;
    double B = p*volume*g;
    double W = mass *g;
    double F_net = W-B;
    //net force of gravity in x
    G[0] = F_net*sin(theta);
    //net force of gravity in y
    G[1] = F_net*cos(theta)*sin(phi);
    //net force of gravity in z
    G[2] = F_net*cos(theta)*cos(phi);
    //net force in roll
    G[3] = 0;
    //net force in pitch
    G[4] = 0;
    //net force in yaw
    G[5] = 0;
}

void create_m(double *m) {
    double mass = 38;
    double w = 10;
    double d = 10;
    double h = 10;
    //first 3 are mass
    m[0] = mass;
    m[1] = mass;
    m[2] = mass;
    //moment of inertia of roll
    m[3] = mass/12*(pow(w,2)+pow(h,2));
    //moment of inertia of pitch
    m[4] = mass/12*(pow(d,2)+pow(h,2));
    //moment of inertia of yaw
    m[5] = mass/12*(pow(w,2)+pow(d,2));
}

void create_L(double *L) {
    //quad motors to center of pitch
    double i1 = 0.377952;
    //drive motors to center of yaw
    double i2 = 0.28702;
    //drive motors to center of pitch
    // double i3 = 0.035306;
    //quad motors to center of roll
    double i4 = 0.162052;
    //yaw motors to center of yaw
    double i5 = 0.45466;

    for(int i = 0; i<6*8;i++){
        L[i] = 0;
    }
    //row 1 - surge
    L[2] = 1;
    L[3] = 1;
    //row 2 - sway
    L[8] = 1;
    L[9] = 1;
    //row 3 - heave
    L[20] = 1;
    L[21] = 1;
    L[22] = 1;
    L[23] = 1;
    //row 4 - roll
    L[28] = -1*i4;
    L[29] = i4;
    L[30] = -1*i4;
    L[31] = i4;
    //row 5 - pitch
    // L[34] = i3;
    // L[35] = i3;
    L[34] = 0;
    L[35] = 0;
    L[36] = i1;
    L[37] = i1;
    L[38] = -1*i1;
    L[39] = -1*i1;
    //row 6 - yaw
    L[40] = i5;
    L[41] = -1*i5;
    L[42] = -1*i2;
    L[43] = i2;

    // for (size_t i = 0; i < 6; i++) {
    //     for (size_t j = 0; j < 8; j++) {
    //         printf("%f, ", L[(8*i)+j]);
    //     }
    //     printf("\n");
    // }

}

__device__ int sgn(double f) {
    if(f == 0) return 1;
    return (0 < f) - (f < 0);
}

__device__ int drag(double v) {
    return 20+v+.14*v*v;
}

/***********************************************************************
* Max Forward Trust : 34.6961 N
* Max Reverse Trust : 29.3583 N
************************************************************************/

__global__
void shooter(double *M, double *G, double *v, double *L, double *d_target,  double * errors,double *range, int seconds) {
    int i = threadIdx.x;

    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;

    int x_size = gridDim.x;
    int y_size = gridDim.y;
    int z_size = gridDim.z;

    __shared__ double t[6];
    __shared__ double u[8];
    __shared__ double local_v[6];
    __shared__ double local_d[6];
    u[0] =  sgn(d_target[1]) * (range[2]+((range[3]-range[2])/(y_size-1))*y);
    u[1] =  sgn(d_target[1]) * (range[2]+((range[3]-range[2])/(y_size-1))*y);
    u[2] =  sgn(d_target[0]) * (range[0]+((range[1]-range[0])/(x_size-1))*x);
    u[3] =  sgn(d_target[0]) * (range[0]+((range[1]-range[0])/(x_size-1))*x);
    u[4] =  sgn(d_target[2]) * (range[4]+((range[5]-range[4])/(z_size-1))*z);
    u[5] =  sgn(d_target[2]) * (range[4]+((range[5]-range[4])/(z_size-1))*z);
    u[6] =  sgn(d_target[2]) * (range[4]+((range[5]-range[4])/(z_size-1))*z);
    u[7] =  sgn(d_target[2]) * (range[4]+((range[5]-range[4])/(z_size-1))*z);
    t[i] = 0;
    for(int k = 0; k<8; k++){
        t[i]+=L[(i*8)+k]*u[k];
    }
    local_d[i] = 0;
    local_v[i] = v[i];

    for (size_t k = 0; k < seconds; k++) {
        /* code */
        local_v[i] += (t[i] - G[i] - drag(local_v[i]))/M[i];
        local_d[i] += local_v[i];
    }
    __syncthreads();
    __shared__ double e[3];
    e[i] = (d_target[i]- local_d[i])*(d_target[i]- local_d[i]);
    if(i == 0) {
        errors[z*x_size*y_size+y*x_size+x] = e[0]+e[1]+e[2];

        // printf("%f,%f,%f,%f,%f,%f,%.5e\n",t[0],t[1],t[2],local_d[0],local_d[1],local_d[2],errors[z*x_size*y_size+y*x_size+x]);
    }
}
//http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
__global__
void reduce_error(double *errors, int *in_index,int *out_index) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(blockDim.x*blockDim.x == gridDim.x){
        in_index[i] = i;
    }
    extern __shared__ int indexs[];
    indexs[tid] = in_index[i];
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            if(errors[indexs[tid]]>errors[indexs[tid+s]]) {

                indexs[tid] = indexs[tid+s];
            }

        }
    __syncthreads();
    }
    if(tid==0)out_index[blockIdx.x]=indexs[0];
}

__global__
void reduce_error_final(double *errors, int * in_index,int *out_index) {
    unsigned int tid = threadIdx.x;
    // unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    extern __shared__ int indexs[];
    indexs[tid] = in_index[tid];
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            if(errors[indexs[tid]]>errors[indexs[tid+s]]) {

                indexs[tid] = indexs[tid+s];
            }

        }
    __syncthreads();
    }
    if(tid==0){
        // printf("Found error %.5e\n", errors[indexs[0]]);
        out_index[0]=indexs[0];
    }
}

void iToXYZ(int i, int*xyz,int n) {
    // printf("I = %d\n", i);
    xyz[2] = i/(n*n);
    i -= xyz[2]*(n*n);
    xyz[1] = i/n;
    i -= xyz[1]*(n);
    xyz[0] = i;
    // printf("Got (%d,%d,%d)\n", xyz[0],xyz[1],xyz[2]);
}

void __cudaCheckError() {
    cudaError err = cudaGetLastError();
    if( err != cudaSuccess){
        printf("cudaCheckError() failed at %s:%i : %s\n", cudaGetErrorString( err ) );
    }
}

void predict(int n, double error_bound, int seconds) {
    int d = 3;
    double *M, *G,  *v, *d_target, *range, *errors;
    int *in_index, *out_index, *out_index2,*final_index;

    double *L;
    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&M, d*sizeof(double));
    cudaMallocManaged(&G, d*sizeof(double));
    cudaMallocManaged(&v, d*sizeof(double));
    cudaMallocManaged(&range, d*2*sizeof(double));
    cudaMallocManaged(&errors, n*n*n*sizeof(double));
    cudaMallocManaged(&in_index, n*n*n*sizeof(int));
    cudaMallocManaged(&out_index, n*n*sizeof(int));
    cudaMallocManaged(&out_index2, n*sizeof(int));
    cudaMallocManaged(&final_index, sizeof(int));

    cudaMallocManaged(&d_target, d*sizeof(double));
    cudaMallocManaged(&L, d*8*sizeof(double));

    create_g(G,0,0);
    create_m(M);
    create_L(L);
    // int seconds = 1;
    d_target[0] = 1;
    d_target[1] = 1;
    d_target[2] = -2;

    v[0] = 0;
    v[1] = 0;
    v[2] = 0;

    for (int i = 0; i < d; i++) {
        /* code */
        v[i] = 0;
    }

    // Run kernel on 1M elements on the GPU

    // int numBlocks = 2;
    double error = 1e9;
    dim3 numBlocks(n,n,n);
    // dim3 threadsPerBlock(3, 1);
    int threadsPerBlock(3);
    int pos[3] = {0,0,0};
    double x_motor;
    double y_motor;
    double z_motor;
    range[0] = 0;
    range[1] = 35;
    range[2] = 0;
    range[3] = 35;
    range[4] = 0;
    range[5] = 100;
    int c = 0;
    // printf("X = %f:%f\nY = %f:%f\nZ = %f:%f\n", range[0], range[1], range[2], range[3], range[4], range[5]);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    
    while(1){
        c++;
        shooter<<<numBlocks, threadsPerBlock>>>(M, G, v, L, d_target,errors,range,seconds);
        cudaDeviceSynchronize();
        __cudaCheckError();
        reduce_error<<<n*n,n,n*sizeof(int)>>>(errors,in_index,out_index);
        cudaDeviceSynchronize();
        __cudaCheckError();
        reduce_error<<<n,n,n*sizeof(int)>>>(errors,out_index,out_index2);
        cudaDeviceSynchronize();
        __cudaCheckError();

        reduce_error_final<<<1,n,n*sizeof(int)>>>(errors,out_index2,final_index);
        cudaDeviceSynchronize();
        __cudaCheckError();
        error = errors[*final_index];
        iToXYZ(*final_index,pos,n);
        x_motor = (range[0]+((range[1]-range[0])/(n-1))*pos[0]);
        y_motor = (range[2]+((range[3]-range[2])/(n-1))*pos[1]);
        z_motor = (range[4]+((range[5]-range[4])/(n-1))*pos[2]);
        if(abs(error)<error_bound){
            break;
        }
        double range_new[6];
        range_new[0] = (range[0]+((range[1]-range[0])/(n-1))*float(pos[0]-0.5));
        range_new[1] = (range[0]+((range[1]-range[0])/(n-1))*float(pos[0]+0.5));
        range_new[2] = (range[2]+((range[3]-range[2])/(n-1))*float(pos[1]-0.5));
        range_new[3] = (range[2]+((range[3]-range[2])/(n-1))*float(pos[1]+0.5));
        range_new[4] = (range[4]+((range[5]-range[4])/(n-1))*float(pos[2]-0.5));
        range_new[5] = (range[4]+((range[5]-range[4])/(n-1))*float(pos[2]+0.5));
        range[0] = range_new[0];
        range[1] = range_new[1];
        range[2] = range_new[2];
        range[3] = range_new[3];
        range[4] = range_new[4];
        range[5] = range_new[5];
        // printf("X = %.5e\nY = %.5e\nZ = %.5e\n", range[1]-range[0], range[3]-range[2], range[5]-range[4]);
        // if(c==40) break;
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    printf("predict returned after Wall Time: %f seconds\n",duration/1e6);
    printf("With n=%d and error bound=%.5e\nTook %d iterations to each result with final error %.5e\nMotor 1=%f\nMotor 2=%f\nMotor 3=%f\nMotor 4=%f\nMotor 5=%f\nMotor 6=%f\nMotor 7=%f\nMotor 8=%f\n", n, error_bound, c, error,x_motor,x_motor,y_motor,y_motor,z_motor,z_motor,z_motor,z_motor);


    // Free memory
    cudaFree(M);
    cudaFree(G);
    cudaFree(v);
    cudaFree(errors);
    cudaFree(in_index);
    cudaFree(out_index);
    cudaFree(out_index2);
    cudaFree(final_index);
    cudaFree(range);
    cudaFree(d_target);
    cudaFree(L);
}
