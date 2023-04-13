#include <iostream>
#include <cuda.h>
//you can change the grid_size
#define GRID_SIZE 32
//you can change the block_size
#define BLOCK_SIZE 32
#define V3

// Final Version
// Version 3 - performed unrolling of the k loop 
__global__ void cnn(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
               float *d_input, float * d_weight, float * d_output){
  unsigned int n = threadIdx.z + blockIdx.z*blockDim.z;
  unsigned int k = threadIdx.y + blockIdx.y*blockDim.y*8;
  unsigned int pq = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int p = pq/Q;
  unsigned int q = pq%Q;
  float sum = 0.0;
  float sum2 = 0.0;
  float sum3 = 0.0;
  float sum4 = 0.0;
  float sum5 = 0.0;
  float sum6 = 0.0;
  float sum7 = 0.0;
  float sum8 = 0.0;
  float input = 0.0;
  if(n < N && p < P && q < Q && k < K-7) {
    unsigned int ij = p * u; // input height
    unsigned int ii = q * v; // input width
    for (unsigned int c = 0; c<C; c ++) { 
      for (unsigned int r = 0; r<R; r ++) { 
        for (unsigned int s = 0; s < S; s ++) {
            input = d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s];
            sum +=  input * d_weight[k*C*R*S+c*R*S+r*S+s];
            sum2 += input * d_weight[(k+1)*C*R*S+c*R*S+r*S+s];
            sum3 += input * d_weight[(k+2)*C*R*S+c*R*S+r*S+s];
            sum4 += input * d_weight[(k+3)*C*R*S+c*R*S+r*S+s];
            sum5 += input * d_weight[(k+4)*C*R*S+c*R*S+r*S+s];
            sum6 += input * d_weight[(k+5)*C*R*S+c*R*S+r*S+s];
            sum7 += input * d_weight[(k+6)*C*R*S+c*R*S+r*S+s];
            sum8 += input * d_weight[(k+7)*C*R*S+c*R*S+r*S+s];
        }
      }
    }
    d_output[n*K*P*Q + k*P*Q + p*Q +q] = sum;
    d_output[n*K*P*Q + (k+1)*P*Q + p*Q +q] = sum2;
    d_output[n*K*P*Q + (k+2)*P*Q + p*Q +q] = sum3;
    d_output[n*K*P*Q + (k+3)*P*Q + p*Q +q] = sum4;
    d_output[n*K*P*Q + (k+4)*P*Q + p*Q +q] = sum5;
    d_output[n*K*P*Q + (k+5)*P*Q + p*Q +q] = sum6;
    d_output[n*K*P*Q + (k+6)*P*Q + p*Q +q] = sum7;
    d_output[n*K*P*Q + (k+7)*P*Q + p*Q +q] = sum8;
  }
}

#ifdef V1
// Version 1 - no optimizations
__global__ void cnn(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
               float *d_input, float * d_weight, float * d_output){
  unsigned int q = threadIdx.z + blockIdx.z*blockDim.z;
  unsigned int p = threadIdx.y + blockIdx.y*blockDim.y;
  unsigned int n = threadIdx.x + blockIdx.x*blockDim.x;
  if(n < N && p < P && q < Q) {
    unsigned int ij = p * u; // input height
    unsigned int ii = q * v; // input width
    for(unsigned int k=0; k<K; k ++) { // output height
      for (unsigned int c = 0; c<C; c ++) { // output width
        for (unsigned int r = 0; r<R; r ++) { // filter height
          for (unsigned int s = 0; s < S; s ++) {// filter width
            //output_seq[n][k][p][q] += input [n][c][ij+r][ii+s] * weight[k][c][r][s];
            d_output[n*K*P*Q + k*P*Q + p*Q + q] += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+c*R*S+r*S+s];
          }   
        }   
      }   
    }   
  }
}
#endif

#ifdef V2
// Version 2 - optimized to coalesce access to d_output in global memory and register use
__global__ void cnn(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
               float *d_input, float * d_weight, float * d_output){
  unsigned int n = threadIdx.z + blockIdx.z*blockDim.z;
  unsigned int k = threadIdx.y + blockIdx.y*blockDim.y;
  unsigned int pq = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int p = pq/Q;
  unsigned int q = pq%Q;
  float sum = 0.0;
  if(n < N && p < P && q < Q) {
    unsigned int ij = p * u; // input height
    unsigned int ii = q * v; // input width
    for (unsigned int c = 0; c<C; c ++) { 
      for (unsigned int r = 0; r<R; r ++) { 
        for (unsigned int s = 0; s < S; s ++) {
            sum += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+c*R*S+r*S+s];
        }   
      }   
    }   
    d_output[n*K*P*Q + k*P*Q + p*Q +q] = sum;
  }   
}
#endif

#ifdef V4
// Version 4 - tiling into shared memory
#define TS 32
__global__ void cnn(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
               float *d_input, float * d_weight, float * d_output){
  unsigned int n = threadIdx.z + blockIdx.z*blockDim.z;
  unsigned int k = threadIdx.y + blockIdx.y*blockDim.y;
  unsigned int tx = threadIdx.x;
  unsigned int pq = tx + blockIdx.x*blockDim.x;
  unsigned int p = pq/Q;
  unsigned int q = pq%Q;
  __shared__ float s_weight[TS];
  float sum = 0.0;
  float input = 0.0;
  int c_incr = TS / (R*S);
  s_weight[tx] = 0.0;
  if(n < N  && k < K) {
    unsigned int ij = p * u; // input height
    unsigned int ii = q * v; // input width
    for (unsigned int cc = 0; cc<C; cc += c_incr) { 
      // Populate portion of weight
      int sc = tx / (R*S);
      int sr = tx % (R*S) / S;
      int ss = tx % (R*S) % S;
      if(cc + sc < C)
        s_weight[tx] = d_weight[k*C*R*S + (cc+sc)*R*S + sr*S + ss];
      __syncthreads();
      for (unsigned int i = 0; i<c_incr*R*S; i++) { 
        int c = i / (R*S);
        int r = i % (R*S) / S;
        int s = i % (R*S) % S;
        input = d_input[n*C*H*W + (cc+c)*H*W + (ij+r)*W + ii+s];
        sum +=  input * s_weight[i];
      }
      __syncthreads();
    }
    if(p < P && q < Q)
      d_output[n*K*P*Q + k*P*Q + p*Q +q] = sum;
  }
}
#endif

#ifdef V5
//Version 5 - combining tiling into shared memory and unrolling
#define TS 32
__global__ void cnn(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
               float *d_input, float * d_weight, float * d_output){
  unsigned int n = threadIdx.z + blockIdx.z*blockDim.z;
  unsigned int k = threadIdx.y + blockIdx.y*blockDim.y*4;
  unsigned int tx = threadIdx.x;
  unsigned int pq = tx + blockIdx.x*blockDim.x;
  unsigned int p = pq/Q;
  unsigned int q = pq%Q;
  __shared__ float s_weight1[TS];
  __shared__ float s_weight2[TS];
  __shared__ float s_weight3[TS];
  __shared__ float s_weight4[TS];
  float sum1 = 0.0;
  float sum2 = 0.0;
  float sum3 = 0.0;
  float sum4 = 0.0;
  float input = 0.0;
  int c_incr = TS / (R*S);
  s_weight1[tx] = 0.0;
  s_weight2[tx] = 0.0;
  s_weight3[tx] = 0.0;
  s_weight4[tx] = 0.0;
  if(n < N  && k < K-1) {
    unsigned int ij = p * u; // input height
    unsigned int ii = q * v; // input width
    unsigned int c, r, s;
    for (unsigned int cc = 0; cc<C; cc += c_incr) { // output width
      // Populate portion of weight
      c = tx / (R*S);
      r = tx % (R*S) / S;
      s = tx % (R*S) % S;
      s_weight1[tx] = d_weight[k*C*R*S + (cc+c)*R*S + s*S + s];
      s_weight2[tx] = d_weight[(k+1)*C*R*S + (cc+c)*R*S + r*S + s];
      s_weight3[tx] = d_weight[(k+2)*C*R*S + (cc+c)*R*S + r*S + s];
      s_weight4[tx] = d_weight[(k+3)*C*R*S + (cc+c)*R*S + r*S + s];
      __syncthreads();
      for (unsigned int i = 0; i<c_incr*R*S; i++) { 
        c = i / (R*S);
        r = i % (R*S) / S;
        s = i % (R*S) % S;

        // Copy input into register
        input = d_input[n*C*H*W + (cc+c)*H*W + (ij+r)*W + ii+s];

        // Compute each partial sum
        sum1 +=  input * s_weight1[i];
        sum2 +=  input * s_weight2[i];
        sum3 +=  input * s_weight3[i];
        sum4 +=  input * s_weight4[i];
      }
      __syncthreads();
    }
    if(p < P && q < Q) {
      // Set output in global memory
      d_output[n*K*P*Q + k*P*Q + p*Q +q] = sum1;
      d_output[n*K*P*Q + (k+1)*P*Q + p*Q +q] = sum2;
      d_output[n*K*P*Q + (k+2)*P*Q + p*Q +q] = sum3;
      d_output[n*K*P*Q + (k+3)*P*Q + p*Q +q] = sum4;
    }
  }
}
#endif
