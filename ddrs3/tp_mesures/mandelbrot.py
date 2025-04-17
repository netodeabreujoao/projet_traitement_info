import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from numba import jit, prange

@jit(nopython=True)

def mandelbrot(m, n, x0, y0, x1, y1, max_iter):

    # fonction calculant la suite pour le point c du plan complex
    # et retournant le nombre d'itérations à la détection de la divergence
    def kernel(c, max_it):
        z = c
        for i in range(max_it):
            z = z**2 + c
            if abs(z) > 2:
                return i
        return max_iter

    X = np.linspace(x0, x1, n)
    Y = np.linspace(y0, y1, m)

    img = np.zeros((m,n))
    # for each row of the image
    for i in range(0,m):
        img[i] = np.array([kernel(x + 1j*Y[i],max_iter) for x in X])
    return img

def main():
    n =2048; # image size
    t0 = time.time()
    img = mandelbrot(n, n, 0.273771332381423218946, 0.595859541361479164066, 0.273771332946091993361, 0.595859541784980744876, 10000)
    t1 = time.time()
    print("Running time: ", t1-t0, "s")
    print("Saving file...")
    plt.imsave('mandelbrot.jpg', img)
    return 0

if __name__ == '__main__':
    sys.exit(main())
