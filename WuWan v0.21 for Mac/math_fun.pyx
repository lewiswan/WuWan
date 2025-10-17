# math_fun.pyx
import cython
from cython cimport view
from libc.math cimport sqrt, M_PI, cos, sin
from libc.stdlib cimport malloc, free

# cimport gsl_wrappers
from gsl_wrappers cimport (
    gsl_sf_bessel_zero_J0, gsl_sf_bessel_zero_J1,
    gsl_integration_glfixed_table, gsl_integration_glfixed_table_alloc,
    gsl_integration_glfixed_table_free, gsl_integration_glfixed_point
)

@cython.cdivision(True)
@cython.boundscheck(False)  
@cython.wraparound(False)   
cdef inline double BJ0(double value) noexcept nogil:
    cdef double cache_1 = value * value
    cdef double cache_r  = 1.0
    cdef double cache_p  = - 0.25 * cache_1
    cdef double cache_2
    cdef double cache_3
    if value > 6:
        cache_2 = value - 0.25 * M_PI
        cache_3 = 1 / (4 * cache_1)
        cache_r = sqrt(2 / M_PI / value) * (cos(cache_2) * (1 - 0.28125 * cache_3 + 1.79443359375 * cache_3 * cache_3) - sin(cache_2) * (- 0.125 + 0.5859375 * cache_3 * 2 - 7.2674560546875 * cache_3 * cache_3 * 2) / value)
    else:
        cache_r = 1 + cache_p
        cache_p *= - cache_1 / 16.0
        cache_r += cache_p
        cache_p *= - cache_1 / 36.0
        cache_r += cache_p
        cache_p *= - cache_1 / 64.0
        cache_r += cache_p
        cache_p *= - cache_1 / 100.0
        cache_r += cache_p
        cache_p *= - cache_1 / 144.0
        cache_r += cache_p
        cache_p *= - cache_1 / 196.0
        cache_r += cache_p
        cache_p *= - cache_1 / 256.0
        cache_r += cache_p
        cache_p *= - cache_1 / 324.0
        cache_r += cache_p
        cache_p *= - cache_1 / 400.0
        cache_r += cache_p
        cache_p *= - cache_1 / 484.0
        cache_r += cache_p
        cache_p *= - cache_1 / 576.0
        cache_r += cache_p
    return cache_r

@cython.cdivision(True)
@cython.boundscheck(False)  
@cython.wraparound(False)   
cdef inline double BJ1(double value) noexcept nogil:
    cdef double cache_1 = value * value
    cdef double cache_r  = 0
    cdef double cache_p  = 0.5 * value
    cdef double cache_2
    cdef double cache_3
    if value > 6:
        cache_2 = value - 0.75 * M_PI
        cache_3 = 1 / (4 * cache_1)
        cache_r = sqrt(2 / M_PI / value) * (cos(cache_2) * (1 + 0.46875 * cache_3 - 2.30712890625 * cache_3 * cache_3) - sin(cache_2) * (0.375 - 0.8203125 * cache_3 / 2 + 8.8824462890625 * cache_3 * cache_3 / 2) / value)
    else:
        cache_r = cache_p
        cache_p *= - cache_1 / 8.0
        cache_r += cache_p
        cache_p *= - cache_1 / 24.0
        cache_r += cache_p
        cache_p *= - cache_1 / 48.0
        cache_r += cache_p
        cache_p *= - cache_1 / 80.0
        cache_r += cache_p
        cache_p *= - cache_1 / 120.0
        cache_r += cache_p
        cache_p *= - cache_1 / 168.0
        cache_r += cache_p
        cache_p *= - cache_1 / 224.0
        cache_r += cache_p
        cache_p *= - cache_1 / 288.0
        cache_r += cache_p
        cache_p *= - cache_1 / 360.0
        cache_r += cache_p
        cache_p *= - cache_1 / 440.0
        cache_r += cache_p
        cache_p *= - cache_1 / 528.0
        cache_r += cache_p
    return cache_r

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double* bessel_zeros(int num_zeros, double r, double a) noexcept nogil:
        """
        Quickly calculate the zero point of J0(mr)*J1(ma)
        parameters:
            num_zeros: Required number of zeros 
            r: Already calculated r in the founction is r/H
            a: Already calculated a in the founction is a/H
        """
        cdef int n_extra_j0 = int(num_zeros) if r > 1e-10 else 0
        cdef int n_extra_j1 = int(num_zeros)
        cdef int i
        cdef int j = 0, k = 0
        cdef int total_size = num_zeros + 1
        cdef double *zeros = <double*>malloc(total_size * sizeof(double))
        zeros[0] = 0

        if r > 1e-10:
            for i in range(num_zeros):
                if gsl_sf_bessel_zero_J0(k + 1) / r > gsl_sf_bessel_zero_J1(j + 1) / a:
                    zeros[i + 1] = gsl_sf_bessel_zero_J1(j + 1) / a
                    j += 1
                else:
                    zeros[i + 1] = gsl_sf_bessel_zero_J0(k + 1) / r
                    k += 1
        else:
            for i in range(num_zeros):
                zeros[i + 1] = gsl_sf_bessel_zero_J1(i + 1) / a
               
        return zeros

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void gauss_legendre_points(long n, double a, double b, double[64] points, double[64] weights) noexcept nogil:
    """
    compute the gauss legendre points and weight in the range between [a, b]
    return: (points, weights)
    """
    cdef gsl_integration_glfixed_table *table
    cdef double xi, wi
    cdef int i

    # allowcate the memory
    table = gsl_integration_glfixed_table_alloc(n)

    try:
        for i in range(n):
            gsl_integration_glfixed_point(a, b, i, &xi, &wi, table)
            points[i] = xi 
            weights[i] = wi 
    finally:
        # release GSL memory
        gsl_integration_glfixed_table_free(table)
    
