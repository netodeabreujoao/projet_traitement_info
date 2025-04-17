#include "utils.h"
#include "stopwatch.h"

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel(int *result, 
const double *xcoord, const double *ycoord, int limit, int n)
{
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t < n)
    {
        double x, y, x0, y0;
        x0 = x = xcoord[t];
        y0 = y = ycoord[t];
        for (int i = 0; i < limit; i++)
        {
            if (x * x + y * y >= 4)
            {
                result[t] = i;
                return;
            }
 
            double zx = x * x - y * y + x0;
            y = 2 * x * y + y0;
            x = zx;
        } 
        result[t] = 0;
    }
}

void MandelbrotKernel(int *result, const double *xcoord, const double *ycoord, const int limit, const int n)
{
    size_t dsize = n * sizeof(double);
    size_t isize = n * sizeof(int);
    size_t csize = limit * sizeof(int);
 
    double* d_xcoord;
    cudaMalloc(&d_xcoord, dsize);
    double* d_ycoord;
    cudaMalloc(&d_ycoord, dsize);
    int* d_result;
    cudaMalloc(&d_result, isize);
    int* d_colors;
    cudaMalloc(&d_colors, csize);
 
    cudaMemcpy(d_xcoord, xcoord, dsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ycoord, ycoord, dsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, isize, cudaMemcpyHostToDevice);
 
    int threadsPerBlock = 512;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_result, d_xcoord, d_ycoord, limit, n);
 
    cudaMemcpy(result, d_result, isize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // Free device memory
    cudaFree(d_xcoord);
    cudaFree(d_ycoord);
    cudaFree(d_result);
    cudaFree(d_colors);
}
using namespace cimg_library;
using namespace Eigen;

#include <iostream>
__host__
void mandelbrot(ArrayXXd &img, double x0, double y0, double x1, double y1, int max_iter)
{
  int m = img.rows();
  int n = img.cols();
  
  auto x = Eigen::ArrayX<double>::LinSpaced(n,x0,x1);
  auto y = Eigen::ArrayX<double>::LinSpaced(m,y0,y1);

  ArrayXXd X = x.replicate(1,m).transpose();
  ArrayXXd Y = y.replicate(1,m);
  ArrayXXi tmp(m,n);
  MandelbrotKernel(tmp.data(), X.data(), Y.data(), max_iter, m*n);
  img = tmp.cast<double>();
}

int main(int argc, char **argv)
{
  int n = 2048; // image size
  int rep = 1;  // number of repetitions for measurements
  if (argc>1)
    n = std::atoi(argv[1]);
  if (argc>2)
    rep = std::atoi(argv[2]);
  
  ArrayXXd img(n,n);
  StopWatch t(true);
  for(int i=0; i<rep; ++i) {
    mandelbrot(img, 0.273771332381423218946, 0.595859541361479164066, 0.273771332946091993361, 0.595859541784980744876, 10000);
  }
  t.stop();

  std::cout << "Running time: " << double(t.elapsed())/1000/rep << "s";
  if(rep>1)
    std::cout << " (average time for computing a single image)";
  std::cout << "\n";
  std::cout << "Save file to disk...\n";
  save_image("mandelbrot.jpg", img);
  return 0;
}


