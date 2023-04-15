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

  // Device Views
  Kokkos::View<float****> d_input("d_input", N, C, H, W);
  Kokkos::View<float****> d_weight("d_weight", K, C, R, S);
  Kokkos::View<float****> d_output("d_output", N, K, P, Q);

  // Host Views
  Kokkos::View<float****>::HostMirror h_input = Kokkos::create_mirror_view(d_input);
  Kokkos::View<float****>::HostMirror h_weight= Kokkos::create_mirror_view(d_weight);
  Kokkos::View<float****>::HostMirror h_output= Kokkos::create_mirror_view(d_output);

  float *output_seq = new float[N*K*P*Q];
  memset(output_seq,0, N * K * P * Q*sizeof(float));
  memset(h_output.data(),0, N * K * P * Q*sizeof(float));
  
  // Assign initial values
  for(unsigned int n=0; n<N; ++n){
    for(unsigned int c=0; c<C; ++c){
      for(unsigned int h=0; h<H; ++h){
        for(unsigned int w=0; w<W; ++w){
          h_input(n,c,h,w) =  ((float)(n+c+h+w));
        }
      }
    }
  }
  for (unsigned int k=0; k<K; k++) {
    for (unsigned int c=0; c<C; c++) {
      for (unsigned int r =0; r<R; r++) {
        for (unsigned int s =0; s<S; s++) {
          //weight[k][c][r][s] = ((float) (k+c+r+s));
          h_weight(k, c, r, s) = ((float) (k+c+r+s));
        }
      }
    }
  }

  // Copy Host Views to Device Views
  Kokkos::deep_copy(d_input, h_input);
  Kokkos::deep_copy(d_weight, h_weight);

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
                output_seq[n*K*P*Q + k*P*Q + p*Q + q] += h_input(n, c, ij+r, ii+s) 
                                                         * h_weight(k, c, r, s);
              }
            }
          }
        }
      }
    }
  }

  // GPU Version
  // Define Range of computation
  Kokkos::parallel_for("CNN", N, 
    KOKKOS_LAMBDA (const int& i){
      d_output(i,0,0,0) = d_input(0,0,0,0) * d_weight(0,0,0,0);
    });

  Kokkos::deep_copy(h_output, d_output);

  // Check output
  for (unsigned int n=0; n<N; n++) {
    for (unsigned int k=0; k<K; k++) {
      for (unsigned int p =0; p<P; p++) {
        for (unsigned int q =0; q<Q; q++) {
          if(h_output(n, k, p, q) != output_seq[n*K*P*Q + k*P*Q + p*Q + q]) {
            printf("Incorrect Output\n");
            break;
          }
        }
      }
    }
  }

  Kokkos::finalize();
  return 0;
}
