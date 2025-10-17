import cython
import numpy as np
cimport numpy as cnp
from libc.stdio cimport printf
from libc.time cimport time_t, time, clock_t, clock, CLOCKS_PER_SEC
from time import perf_counter
from libc.stdlib cimport malloc, free, calloc
from cython cimport view
from math_fun cimport bessel_zeros
from processing_function cimport Input_num, index_search, Variable_Assignment
from integrand_solver cimport gaussian_quadrature_integrate

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def Calculation(double[:, :] Data_array):
    """Deifine parameters: """
    ### --- Basic parameters ---
    cdef double[144] Coe_Matrix, Coe_Matrix_copy
    cdef int num_zeros = 120
    cdef int num_cols = 10
    cdef int i
    ### --- model parameters ---
    cdef double[2] evaluation
    cdef double[6] z, nu, E
    cdef double[18] b
    cdef double[20] Coe 
    cdef int[28] ipiv
    cdef double q, a, H
    ### --- calculator parameters ---
    cdef double[6] F1, F2, F3
    cdef int alpha, index
    cdef double[64] weights, points
    ### --- Dynamically allocated memory ---
    cdef double* zeros
    ### --- result array ---
    cdef double[10] result_displacement
    ### --- time counter ---
    cdef double start, end, elapsed
    """Assignment parameters: """
    ### --- alpha = 1 -> choose fully bounded ---
    alpha = 1
    ### --- q: load stress. a: load radius ---
    q = Data_array[10, 1]
    a = Data_array[10, 3]
    ### --- Input_num is a function assignment the E, nu, z ---
    Input_num(Data_array, z, nu, E, evaluation, 0)
    ### --- H is the total depth of each layer (expect the last layer) ---
    H = z[4]
    ### --- get the index of layer, 1st / 2nd / ... ---
    index = index_search(z, evaluation)
    ### --- parmeter of Coe_Matrix
    F2[0] = (E[0] / E[1]) * ((1.0 + nu[1]) / (1.0 + nu[0]))
    F2[1] = (E[1] / E[2]) * ((1.0 + nu[2]) / (1.0 + nu[1]))
    F2[2] = (E[2] / E[3]) * ((1.0 + nu[3]) / (1.0 + nu[2]))
    F2[3] = (E[3] / E[4]) * ((1.0 + nu[4]) / (1.0 + nu[3]))
    F2[4] = (E[4] / E[5]) * ((1.0 + nu[5]) / (1.0 + nu[4]))
    """Compute deflections: """
    start = perf_counter()
    Variable_Assignment(z, nu, E, evaluation, H, q, a, F2, Coe_Matrix)
    for i in range(num_cols):
        evaluation[0] = Data_array[i + 1, 6]
        zeros = bessel_zeros(num_zeros, evaluation[0] / H, a / H)
        result_displacement[i] = gaussian_quadrature_integrate(z, nu, E, evaluation, H, q, a, alpha, F1, F2, F3, zeros, 0, 121, index, points, weights, Coe_Matrix, Coe_Matrix_copy, b, Coe, ipiv)
    end = perf_counter()
    elapsed = end - start
    print(f"Run time: {elapsed:.9f} s")
    free(zeros)
    return result_displacement[0], result_displacement[1], result_displacement[2], result_displacement[3], result_displacement[4], result_displacement[5], result_displacement[6], result_displacement[7], result_displacement[8], result_displacement[9]