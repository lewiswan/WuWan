# math_fun.pxd
from cython cimport view

# announcement cdef function
# announcement inline function
cdef double BJ0(double value) noexcept nogil
cdef double BJ1(double value) noexcept nogil

cdef double* bessel_zeros(int num_zeros, double r, double a) noexcept nogil
cdef void gauss_legendre_points(long n, double a, double b, double[64] points, double[64] weights) noexcept nogil
