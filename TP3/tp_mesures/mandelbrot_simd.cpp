
#pragma clang fp contract(fast)

#include "utils.h"
#include "stopwatch.h"
#include <iostream>

using namespace Eigen;

/* This version exploits two low level parallelisms:
    1 - SIMD vectorization by processing P consecutive pixels at once (typically P=4 for AVX enabled CPUs)
    2 - instruction pipelining by processing NC columns at once
*/
template<typename Real, int NC=4>
long int mandelbrot_simd(ArrayXX<Real> &img, Real x0, Real y0, Real x1, Real y1, int max_iter)
{
  using namespace Eigen::internal;
  using Packet = typename Eigen::internal::packet_traits<Real>::type;
  enum { packetSize = Eigen::internal::packet_traits<Real>::size };

  const int iters_before_test = 8;
  max_iter = (max_iter / iters_before_test) * iters_before_test;

  int m = img.rows();
  int n = img.cols();
  
  Real sx = (x1-x0)/Real(n-1);
  Packet sy = pset1<Packet>((y1-y0)/Real(m-1));

  long int count = 0;

  #pragma omp parallel for
  for(int x = 0; x < n; x+=NC)
  {
    Packet pzr_start[NC];
    Packet pcr_start[NC];
    for(int k=0; k<NC; ++k) {
      pzr_start[k] = pset1<Packet>(x0 + (x+k) * sx);
      pcr_start[k] = pzr_start[k];
    }

    for(int y = 0; y < m; y += packetSize)
    {
      Packet pcr[NC]; // <-> c.real
      Packet pci[NC]; // <-> c.imag
      Packet pzr[NC]; // <-> z.real
      Packet pzi[NC]; // <-> z.imag
      Packet pzr_buf[NC];   // temporary
      Packet pix_iter[NC];  // number of iterations
      for(int k=0; k<NC; ++k) {
        pcr[k] = pcr_start[k];
        pzr[k] = pzr_start[k];
        pci[k] = pmadd(plset<Packet>(y),sy,pset1<Packet>(y0));
        pzi[k] = pci[k];
        pix_iter[k] = pset1<Packet>(1);
      }

      int j = 0;
      
      for(; j<max_iter/iters_before_test; ++j)
      {
        for(int i=0 ; i<iters_before_test; ++i) {
          // Compute z = square(z) + c
          for(int k=0; k<NC; ++k)
            pzr_buf[k] = pmul(pset1<Packet>(2), pzr[k]);
          for(int k=0; k<NC; ++k)
            pzr[k] = pmadd(pzr[k], pzr[k], pcr[k]);
          for(int k=0; k<NC; ++k)
            pzr[k] = psub(pzr[k], pmul(pzi[k], pzi[k]));
          for(int k=0; k<NC; ++k)
            pzi[k] = pmadd(pzr_buf[k], pzi[k], pci[k]);
        }
        // check if numext::abs2(z) > 4
        Packet norm[NC], mask[NC];
        bool any = false;
        for(int k=0; k<NC; ++k) {
          norm[k] = pmadd(pzr[k], pzr[k], pmul(pzi[k], pzi[k]));
          mask[k] = pcmp_le(norm[k], pset1<Packet>(4));
          any = any || predux_any(mask[k]);
        }
        if (!any)
          break;

        for(int k=0; k<NC; ++k) {
          pix_iter[k] = padd(pand(pset1<Packet>(iters_before_test), mask[k]), pix_iter[k]);
        }
      }
      #ifndef _OPENMP
      count += j;
      #endif
      for(int k=0; k<NC; ++k) {
        pstore(&img(y,x+k), pix_iter[k]);
      }
    }
  }
  return count*packetSize*iters_before_test*NC;
}


//--------------------------------------------------------------------------------
int main(int argc, char **argv)
{
  int n = 2048; // image size
  int rep = 1;  // number of repetitions for measurements
  if (argc>1)
    n = std::atoi(argv[1]);
  if (argc>2)
    rep = std::atoi(argv[2]);

  int ps = std::max<int>(Eigen::internal::packet_traits<double>::size, 4);
  if (n%ps>0) {
    std::cout << "Error: the image size must be a multiple of " << ps << "\n";
    return -1;
  }
  
  ArrayXXd img(n,n);
  
  long int count;
  StopWatch t(true);
  for(int i=0; i<rep; ++i) {
    count = mandelbrot_simd<double,1>(img, 0.273771332381423218946, 0.595859541361479164066, 0.273771332946091993361, 0.595859541784980744876, 10000);
  }
  t.stop();

  std::cout << "Running time: " << double(t.elapsed())/1000/rep << "s";
  if(rep>1)
    std::cout << " (average time for computing a single image)";
  std::cout << "\n";
  std::cout << "Save file to disk...\n";
  if (count>0)
    std::cout << "Nombre moyen d'itérations par pixel : " << (count/n/n) << "\n";
  save_image("mandelbrot.jpg", img);
  return 0;
}
