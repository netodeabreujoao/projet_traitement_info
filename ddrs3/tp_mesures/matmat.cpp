
#include <Eigen/Core>
#include <iostream>
using namespace Eigen;

EIGEN_DONT_INLINE
void gemm(const MatrixXf &A, const MatrixXf &B, MatrixXf &C) {
    C.noalias() = A*B;
}

 int main(int argc, char **argv) {

    int n = 1000;
    int k = 10;
    if (argc>1)
        n = std::atoi(argv[1]);
    if (argc>2)
        k = std::atoi(argv[2]);
    MatrixXf A = MatrixXf::Random(n,n);
    MatrixXf B = MatrixXf::Random(n,n);
    MatrixXf C(n,n);

    for (int i=0; i<k; ++i) {
        gemm(A, B, C);
        B = C;
    }
    return int(C.sum());
 }
