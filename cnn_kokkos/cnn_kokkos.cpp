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

  Kokkos::Timer timer;

  // Sequential version
  auto seq_start = std::chrono::high_resolution_clock::now();

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

  auto seq_elapsed = std::chrono::high_resolution_clock::now() - seq_start;
  long long seq_elapsed_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(seq_elapsed).count();
  printf("COMPUTE_TIME_IN_MICROSECONDS: %lld\n", seq_elapsed_microseconds);
  double seq_time = timer.seconds()*1000;

  // GPU Version
  // Define Range of computation
  auto range = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{N, K, P*Q}, {0, 8, 0});
  timer.reset();
  Kokkos::parallel_for("CNN", range, 
    KOKKOS_LAMBDA (const int n, const int k, const int pq){
      unsigned p = pq/Q;
      unsigned q = pq%Q;
      float sum = 0.0;
      float input = 0.0;
      unsigned ij = p * u; // input height
      unsigned ii = q * v; // input width
      for (unsigned c = 0; c<C; c ++) { 
        for (unsigned r = 0; r<R; r ++) { 
          for (unsigned s = 0; s < S; s ++) {
            input = d_input(n, c, ij+r, ii+s);
            sum +=  input * d_weight(k, c, r, s);
          }
        }
      }
      d_output(n, k, p, q) = sum;
    });
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
