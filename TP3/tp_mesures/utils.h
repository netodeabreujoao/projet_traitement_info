#pragma once

#pragma clang fp contract(fast)

#define cimg_display 0
#include "CImg.h"
#include <Eigen/Core>

template<typename Real>
std::complex<Real> square(const std::complex<Real> &x) {
  Real xr = real(x);
  Real xi = imag(x);
  return std::complex<Real>(xr*xr - xi*xi, 2*xr*xi);
}

template<typename MatrixType>
void save_image(const char* filename, const MatrixType &data)
{
  using namespace cimg_library;
  auto cpy = (data.array() - data.minCoeff()).eval();
  cpy = cpy/cpy.maxCoeff();
  CImg<unsigned char> img(data.cols(), data.rows(), 1, 1);

  for(int j=0; j<data.cols(); ++j)
    for(int i=0; i<data.rows(); ++i)
      img(j,i) = cpy(i,j)*255;

  img.save(filename);
}
