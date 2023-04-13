/*
 * CS 6235 Project - Implementation of CNN computation in Kokkos
 * Luke Majors
 */
#include <Kokkos_Core.hpp>
#include <cstdio>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  
  auto ex = Kokkos::DefaultExecutionSpace().name();
  printf("Using Execution Space: %s\n", ex);
  
  // Define problem sizes
  int N{128};// = atoi(argv[1]);
  int C{3};// = atoi(argv[2]);
  int K{64};// = atoi(argv[3]);
  int H{112};// = atoi(argv[4]);
  int W{112};// = atoi(argv[5]);
  int R{3};// = atoi(argv[6]);
  int S{3};// = atoi(argv[7]);
  int u{2};// = atoi(argv[8]);
  int v{2};// = atoi(argv[9]);
  int P = (H-R)/u + 1;
  int Q = (W-S)/v + 1;


  // Create Views for host and device
  Kokkos::View<float*
  
  float *output_seq = new float[N*K*P*Q];
  memset(output_seq,0, N * K * P * Q*sizeof(float));
  float *output_par = new float[N*K*P*Q];
  memset(output_par,0, N * K * P * Q*sizeof(float));
  float *input = new float[N*C*H*W];
  float *weight = new float[K*C*R*S];
  
  // Assign initial values
  for(unsigned int n=0; n<N; ++n){
    for(unsigned int c=0; c<C; ++c){
      for(unsigned int h=0; h<H; ++h){
        for(unsigned int w=0; w<W; ++w){
          input[n*C*H*W + c*H*W + h*W + w] =  ((float)(n+c+h+w));
        }
      }
    }
  }
  for (unsigned int k=0; k<K; k++) {
    for (unsigned int c=0; c<C; c++) {
      for (unsigned int r =0; r<R; r++) {
        for (unsigned int s =0; s<S; s++) {
          //weight[k][c][r][s] = ((float) (k+c+r+s));
          weight[k*C*R*S + c*R*S + r*S + s] = ((float) (k+c+r+s));
        }
      }
    }
  }

  // Sequential version
  for(unsigned int n=0; n<N; n++) {                   // minibatch size
    for(unsigned int k=0; k<K; k ++) {                // output feature map
      for(unsigned int c=0; c<C; c ++) {              // input feature map
        for(unsigned int p=0; p<P; p ++) {            // output height
          unsigned int ij = p * u;                    // input height
          for (unsigned int q = 0; q<Q; q ++) {       // output width
            unsigned int ii = q * v;                  // input width
            for (unsigned int r = 0; r<R; r ++) {     // filter height
              for (unsigned int s = 0; s < S; s ++) { // filter width
                //output_seq[n][k][p][q] += input [n][c][ij+r][ii+s] * weight[k][c][r][s];
                output_seq[n*K*P*Q + k*P*Q + p*Q + q] += input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] 
                                                         * weight[k*C*R*S+c*R*S+r*S+s];
              }
            }
          }
        }
      }
    }
  }

  // GPU Version
  // Define data: kokkos uses views
  const int n = 10;

  //Kokkos::View<int> sum("sum", 0);
  int sum = 0;
  Kokkos::parallel_for("CNN", n, KOKKOS_LAMBDA (const int& i, int& sum){sum += i*i;}, sum);
  printf(
      "Sum of squares of integers from 0 to %i, "
      "computed in parallel, is %i\n",
      n - 1, sum);

  // Sequential Implementation
  int seqSum = 0;
  for (int i = 0; i < n; ++i) {
    seqSum += i * i;
  }
  printf(
      "Sum of squares of integers from 0 to %i, "
      "computed sequentially, is %i\n",
      n - 1, seqSum);
  Kokkos::finalize();
  return (sum == seqSum) ? 0 : -1;
}
