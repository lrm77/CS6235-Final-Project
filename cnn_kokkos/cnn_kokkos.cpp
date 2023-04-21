/*
 * CS 6235 Project - Implementation of CNN computation in Kokkos
 * Luke Majors
 */
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <chrono>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
  
  auto ex = Kokkos::DefaultExecutionSpace().name();
  printf("Using Execution Space: %s\n", ex);
  
  // Define problem sizes
  //int N{128};// = atoi(argv[1]);
  //int C{3};// = atoi(argv[2]);
  //int K{64};// = atoi(argv[3]);
  //int H{112};// = atoi(argv[4]);
  //int W{112};// = atoi(argv[5]);
  //int R{3};// = atoi(argv[6]);
  //int S{3};// = atoi(argv[7]);
  //int u{2};// = atoi(argv[8]);
  //int v{2};// = atoi(argv[9]);
  int N = atoi(argv[1]);
  int C = atoi(argv[2]);
  int K = atoi(argv[3]);
  int H = atoi(argv[4]);
  int W = atoi(argv[5]);
  int R = atoi(argv[6]);
  int S = atoi(argv[7]);
  int u = atoi(argv[8]);
  int v = atoi(argv[9]);
  int P = (H-R)/u + 1;
  int Q = (W-S)/v + 1;


  // Create Views for host and device

  // Device Views
  Kokkos::View<float****, Kokkos::LayoutRight> d_input("d_input", N, C, H, W);
  Kokkos::View<float****, Kokkos::LayoutRight> d_weight("d_weight", K, C, R, S);
  Kokkos::View<float****, Kokkos::LayoutRight> d_output("d_output", N, K, P, Q);

  // Host Views
  auto h_input = Kokkos::create_mirror_view(d_input);
  auto h_weight= Kokkos::create_mirror_view(d_weight);
  auto h_output= Kokkos::create_mirror_view(d_output);

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

  // Initialize timer
  Kokkos::Timer timer;

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

  // Record sequential implementation in milliseconds
  double seq_time = timer.seconds()*1000;

  // GPU Version
  // Define Range of computation
  // Specifies Tile Size - equivalent to block size when using CUDA backend
  //auto range = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{P*Q, K/8, N}, {64, 1, 1});
  // Let Kokkos automatically set block size
  auto range = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{P*Q, K/8, N});
  timer.reset();
  Kokkos::parallel_for("CNN", range, 
    KOKKOS_LAMBDA (const int pq, const int kk, const int n){
      unsigned p = pq/Q;
      unsigned q = pq%Q;
      unsigned k = kk*8;

      float sum1 = 0.0;
      float sum2 = 0.0;
      float sum3 = 0.0;
      float sum4 = 0.0;
      float sum5 = 0.0;
      float sum6 = 0.0;
      float sum7 = 0.0;
      float sum8 = 0.0;

      float input = 0.0;

      unsigned ij = p * u; // input height
      unsigned ii = q * v; // input width

      for (unsigned c = 0; c<C; c ++) { 
        for (unsigned r = 0; r<R; r ++) { 
          for (unsigned s = 0; s < S; s ++) {
            input = d_input(n, c, ij+r, ii+s);
            sum1 += input * d_weight(k  , c, r, s);
            sum2 += input * d_weight(k+1, c, r, s);
            sum3 += input * d_weight(k+2, c, r, s);
            sum4 += input * d_weight(k+3, c, r, s);
            sum5 += input * d_weight(k+4, c, r, s);
            sum6 += input * d_weight(k+5, c, r, s);
            sum7 += input * d_weight(k+6, c, r, s);
            sum8 += input * d_weight(k+7, c, r, s);
          }
        }
      }
      d_output(n, k  , p, q) = sum1;
      d_output(n, k+1, p, q) = sum2;
      d_output(n, k+2, p, q) = sum3;
      d_output(n, k+3, p, q) = sum4;
      d_output(n, k+4, p, q) = sum5;
      d_output(n, k+5, p, q) = sum6;
      d_output(n, k+6, p, q) = sum7;
      d_output(n, k+7, p, q) = sum8;
    });

  // Block until parallel execution is complete
  Kokkos::fence();
  double par_time = timer.seconds()*1000;

  Kokkos::deep_copy(h_output, d_output);

  // Check output
  bool fail = false;
  for (unsigned int n=0; n<N && !fail; n++) {
    for (unsigned int k=0; k<K && !fail; k++) {
      for (unsigned int p =0; p<P && !fail; p++) {
        for (unsigned int q =0; q<Q && !fail; q++) {
          if(h_output(n, k, p, q) != output_seq[n*K*P*Q + k*P*Q + p*Q + q]) {
            printf("Incorrect Output: h_output(%d, %d, %d, %d) != %f\n", n, k, p, q, output_seq[n*K*P*Q + k*P*Q + p*Q + q]);
            fail = true;
            break;
          }
        }
      }
    }
  }

  if(!fail)
    printf("Sequential Time: %f, Parallel Time: %f, Speedup: %f\n", seq_time, par_time, seq_time/par_time);
  delete output_seq;
  }
  Kokkos::finalize();
  return 0;
}
