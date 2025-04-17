
#pragma clang fp contract(fast)

#include "utils.h"
#include "stopwatch.h"
#include <iostream>

using namespace Eigen;

//--------------------------------------------------------------------------------
template<typename Real>
long int mandelbrot(ArrayXX<Real> &img, Real x0, Real y0, Real x1, Real y1, int max_iter)
{
  using cplx = std::complex<Real>;
  int m = img.rows();
  int n = img.cols();

  auto X = Eigen::ArrayX<Real>::LinSpaced(n,x0,x1);
  auto Y = Eigen::ArrayX<Real>::LinSpaced(m,y0,y1);

  // évalue la suite pour le point c
  // et retourne le nombre d'itérations à la détection de la divergence
  auto kernel = [max_iter] (std::complex<Real> c) {
    auto z = c;
    for(int i=0; i<max_iter; ++i) {
        z = square(z) + c;
        if(numext::abs2(z) > 4)
          return i;
    }
    return max_iter;
  };

  // for each column of the image
  #pragma omp parallel for
  for(int i = 0; i < n; ++i)
  {
    Real cr = X(i);
    img.col(i) = Y.unaryExpr([cr,kernel](Real ci) { return Real(kernel(cplx(cr, ci))); });
  }

  return img.sum();
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

  ArrayXXd img(n,n);

  long int count;
  StopWatch t(true);
  for(int i=0; i<rep; ++i) {
    count = mandelbrot<double>(img, 0.273771332381423218946, 0.595859541361479164066, 0.273771332946091993361, 0.595859541784980744876, 10000);
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
