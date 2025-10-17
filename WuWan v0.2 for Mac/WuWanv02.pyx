import cython
import numpy as np
cimport numpy as cnp
from libc.stdio cimport printf
from libc.string cimport memcpy
from libc.time cimport time_t, time, clock_t, clock, CLOCKS_PER_SEC
from time import perf_counter
from libc.stdlib cimport malloc, free, calloc
from libc.math cimport floor, ceil, round, exp
from libc.stdint cimport int64_t
from libc.math cimport sqrt, fabs, M_PI, cos, sin, pow, cbrt
from cython cimport view
from libc.string cimport memset

cdef double[144] Coe_Matrix ###Global array
cdef double[144] Coe_Matrix_copy

# announcement the extern C
cdef extern from "gsl/gsl_sf_bessel.h" nogil:
    # announce the function
    double gsl_sf_bessel_J0(double x)
    double gsl_sf_bessel_J1(double x)
    double gsl_sf_bessel_zero_J0(int n) 
    double gsl_sf_bessel_zero_J1(int n)

# announcement the extern C
cdef extern from "gsl/gsl_integration.h" nogil:
    # C Structure
    ctypedef struct gsl_integration_glfixed_table:
        size_t n                    # number of nodes （size_t is Unsigned integer）
        double *x                   # nodes
        double *w                   # weights
        
    gsl_integration_glfixed_table * gsl_integration_glfixed_table_alloc(size_t n) # Allocate memory and return a table pointer
    void gsl_integration_glfixed_table_free(gsl_integration_glfixed_table * t) # Release table memory
    int gsl_integration_glfixed_point(double a, double b, size_t i,
                                    double *xi, double *wi,
                                    const gsl_integration_glfixed_table *t)

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
cdef void Input_num(double[:, :] Data_array, double* z, double* nu, double* E,
                   double* evaluation, int index_point) noexcept nogil:
    z[0] = 0.0
    z[1] = Data_array[2, 3] 
    z[2] = Data_array[3, 3] + z[1]
    z[3] = Data_array[4, 3] + z[2]
    z[4] = Data_array[5, 3] + z[3]
    z[5] = 1e10
    
    # poisson's ratio
    nu[0] = 0.0
    nu[1] = Data_array[2, 2]
    nu[2] = Data_array[3, 2]
    nu[3] = Data_array[4, 2]
    nu[4] = Data_array[5, 2]
    nu[5] = Data_array[6, 2]

    # Young's modulus
    E[0] = 0.0
    E[1] = Data_array[2, 1]
    E[2] = Data_array[3, 1]
    E[3] = Data_array[4, 1]
    E[4] = Data_array[5, 1]
    E[5] = Data_array[6, 1]

    evaluation[0] = Data_array[index_point, 6]
    evaluation[1] = 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int index_search(double *z, double *evaluation) noexcept nogil:
    cdef int low = 0
    cdef int high = 5
    cdef int mid
    cdef double target = evaluation[1]

    while low <= high:
        mid = (low + high) // 2
        if z[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    if low <= 0:
        return 1

    return low 

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void Variable_Assignment(double* z, double* nu, double* E, double* evaluation, double H, double q, double a, double* F2) noexcept nogil:
    Coe_Matrix[0] = 0.0   ### Placeholder 0
    Coe_Matrix[1] = 0.0   ### Placeholder 0
    Coe_Matrix[2] = 0.0   ### Diagonal elements [0,0]: 0
    Coe_Matrix[3] = - 2.0
    Coe_Matrix[4] = 0.0
    Coe_Matrix[5] = 4 * nu[1] - 1
    Coe_Matrix[6] = 0.0
    Coe_Matrix[7] = 0.0
    ### second row
    Coe_Matrix[8] = 0.0   ### Placeholder 0
    Coe_Matrix[9] = 0.0
    Coe_Matrix[10] = - 1.0   ### Diagonal elements [1,1]: -1
    Coe_Matrix[11] = 2 * nu[1]
    Coe_Matrix[12] = 2 * nu[1]
    Coe_Matrix[13] = 0.0
    Coe_Matrix[14] = 0.0
    Coe_Matrix[15] = 0.0
    ### third row
    Coe_Matrix[16] = 2.0
    Coe_Matrix[17] = 0.0
    Coe_Matrix[18] = Coe_Matrix[5]   ### Diagonal elements [2,2]: 2F3[1] + 4nu[1] - 1
    Coe_Matrix[19] = 0.0
    Coe_Matrix[20] = 0.0
    Coe_Matrix[21] = 0.0
    Coe_Matrix[22] = 1.0 - 4 * nu[2]
    Coe_Matrix[23] = - 1.0
    ### 4th row
    Coe_Matrix[24] = 0.0
    Coe_Matrix[25] = 1.0
    Coe_Matrix[26] = Coe_Matrix[5]   ### Diagonal elements [3,3]: F1[1] * (4 * nu[1] - 1 - 2 * F3[1])
    Coe_Matrix[27] = 0.0
    Coe_Matrix[28] = 2.0
    Coe_Matrix[29] = 0.0
    Coe_Matrix[30] = Coe_Matrix[22]
    Coe_Matrix[31] = 0.0
    ### 5th row
    Coe_Matrix[32] = - Coe_Matrix[5] + 3
    Coe_Matrix[33] = 0.0
    Coe_Matrix[34] = 0.0   ### Diagonal elements [4,4]: 
    Coe_Matrix[35] = 2.0 - 2.0 * F2[1]
    Coe_Matrix[36] = 4.0 * F2[1] * nu[2] - 3 * F2[1] - 1.0
    Coe_Matrix[37] = 1.0 - F2[1] - 4 * nu[2] + 4 * F2[1] * nu[2]
    Coe_Matrix[38] = 0.0
    Coe_Matrix[39] = 0.0
    ### 6th row
    Coe_Matrix[40] = Coe_Matrix[5] - 3
    Coe_Matrix[41] = 2 - 2 * F2[1]
    Coe_Matrix[42] = 0.0   ### Diagonal elements [5,5]: 
    Coe_Matrix[43] = Coe_Matrix[37]
    Coe_Matrix[44] = - Coe_Matrix[36]
    Coe_Matrix[45] = 0.0
    Coe_Matrix[46] = 0.0
    Coe_Matrix[47] = 0.0
    ### 7th row
    Coe_Matrix[48] = 2.0
    Coe_Matrix[49] = 0.0
    Coe_Matrix[50] = - Coe_Matrix[22]   ### Diagonal elements [6,6]: 
    Coe_Matrix[51] = 0.0
    Coe_Matrix[52] = 0.0
    Coe_Matrix[53] = 0.0
    Coe_Matrix[54] = 1.0 - 4 * nu[3]
    Coe_Matrix[55] = - 1.0
    ### 8th row
    Coe_Matrix[56] = 0.0
    Coe_Matrix[57] = 1.0
    Coe_Matrix[58] = - Coe_Matrix[22]   ### Diagonal elements [7,7]: 
    Coe_Matrix[59] = 0.0
    Coe_Matrix[60] = 2.0
    Coe_Matrix[61] = 0.0
    Coe_Matrix[62] = Coe_Matrix[54]
    Coe_Matrix[63] = 0.0
    ### 9th row
    Coe_Matrix[64] = Coe_Matrix[22] + 3
    Coe_Matrix[65] = 0.0
    Coe_Matrix[66] = 0.0   ### Diagonal elements [8,8]: 
    Coe_Matrix[67] = 2.0 - 2.0 * F2[2]
    Coe_Matrix[68] = 4.0 * F2[2] * nu[3] - 3 * F2[2] - 1.0
    Coe_Matrix[69] = 1.0 - F2[2] - 4 * nu[3] + 4 * F2[2] * nu[3]
    Coe_Matrix[70] = 0.0
    Coe_Matrix[71] = 0.0
    ### 10th row
    Coe_Matrix[72] = - Coe_Matrix[22] - 3
    Coe_Matrix[73] = 2 - 2 * F2[2]
    Coe_Matrix[74] = 0.0   ### Diagonal elements [9,9]: 
    Coe_Matrix[75] = Coe_Matrix[69]
    Coe_Matrix[76] = - Coe_Matrix[68]
    Coe_Matrix[77] = 0.0
    Coe_Matrix[78] = 0.0
    Coe_Matrix[79] = 0.0
    ### 11th row
    Coe_Matrix[80] = 2.0
    Coe_Matrix[81] = 0.0
    Coe_Matrix[82] = - Coe_Matrix[54]   ### Diagonal elements [10,10]: 
    Coe_Matrix[83] = 0.0
    Coe_Matrix[84] = 0.0
    Coe_Matrix[85] = 0.0
    Coe_Matrix[86] = 1.0 - 4 * nu[4]
    Coe_Matrix[87] = - 1.0
    ### 12th row
    Coe_Matrix[88] = 0.0
    Coe_Matrix[89] = 1.0
    Coe_Matrix[90] = - Coe_Matrix[54]   ### Diagonal elements [11,11]: 
    Coe_Matrix[91] = 0.0
    Coe_Matrix[92] = 2.0
    Coe_Matrix[93] = 0.0
    Coe_Matrix[94] = Coe_Matrix[86]
    Coe_Matrix[95] = 0.0
    ### 13th row
    Coe_Matrix[96] = Coe_Matrix[54] + 3
    Coe_Matrix[97] = 0.0
    Coe_Matrix[98] = 0.0   ### Diagonal elements [12,12]: 
    Coe_Matrix[99] = 2.0 - 2.0 * F2[3]
    Coe_Matrix[100] = 4.0 * F2[3] * nu[4] - 3 * F2[3] - 1.0
    Coe_Matrix[101] = 1.0 - F2[3] - 4 * nu[4] + 4 * F2[3] * nu[4]
    Coe_Matrix[102] = 0.0
    Coe_Matrix[103] = 0.0
    ### 14th row
    Coe_Matrix[104] = - Coe_Matrix[54] - 3
    Coe_Matrix[105] = 2 - 2 * F2[3]
    Coe_Matrix[106] = 0.0   ### Diagonal elements [13,13]: 
    Coe_Matrix[107] = Coe_Matrix[101]
    Coe_Matrix[108] = - Coe_Matrix[100]
    Coe_Matrix[109] = 0.0
    Coe_Matrix[110] = 0.0
    Coe_Matrix[111] = 0.0   ### Placeholder 0
    ### 15th row
    Coe_Matrix[112] = 2.0
    Coe_Matrix[113] = 0.0
    Coe_Matrix[114] = - Coe_Matrix[86]   ### Diagonal elements [14,14]: 
    Coe_Matrix[115] = 0.0
    Coe_Matrix[116] = 0.0
    Coe_Matrix[117] = - 1.0
    Coe_Matrix[118] = 0.0   ### Placeholder 0
    Coe_Matrix[119] = 0.0   ### Placeholder 0
    ### 16th row
    Coe_Matrix[120] = 0.0
    Coe_Matrix[121] = 1.0
    Coe_Matrix[122] = - Coe_Matrix[86]   ### Diagonal elements [15,15]: 
    Coe_Matrix[123] = 2.0
    Coe_Matrix[124] = 1.0 - 4.0 * nu[5]
    Coe_Matrix[125] = 0.0   ### Placeholder 0
    Coe_Matrix[126] = 0.0   ### Placeholder 0
    Coe_Matrix[127] = 0.0   ### Placeholder 0
    ### 17th row
    Coe_Matrix[128] = Coe_Matrix[86] + 3
    Coe_Matrix[129] = 0.0
    Coe_Matrix[130] = 2.0 - 2.0 * F2[4]   ### Diagonal elements [16,16]: 
    Coe_Matrix[131] = 1.0 - F2[4] - 4 * nu[5] + 4 * F2[4] * nu[5]
    Coe_Matrix[132] = 0.0   ### Placeholder 0
    Coe_Matrix[133] = 0.0   ### Placeholder 0
    Coe_Matrix[134] = 0.0   ### Placeholder 0
    Coe_Matrix[135] = 0.0   ### Placeholder 0
    ### 18th row
    Coe_Matrix[136] = - Coe_Matrix[86] - 3
    Coe_Matrix[137] = 0.0
    Coe_Matrix[138] = 1.0 + 3 * F2[4] - 4 * F2[4] * nu[5]   ### Diagonal elements [17,17]: 
    Coe_Matrix[139] = 0.0   ### Placeholder 0
    Coe_Matrix[140] = 0.0   ### Placeholder 0
    Coe_Matrix[141] = 0.0   ### Placeholder 0
    Coe_Matrix[142] = 0.0   ### Placeholder 0
    Coe_Matrix[143] = 0.0   ### Placeholder 0


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
    

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.inline    
cdef inline void Coefficient_52(double[64] points, double[64] weights, int point_iter, double[6] F1, double[6] F2, double[6] F3, double* z, double* nu, double* E, int alpha, double H) noexcept nogil:
    cdef double m = points[point_iter]
    cdef int i
    for i in range(1, 5):
        F1[i] = exp(m * (z[i-1] - z[i]) / H)
        F3[i] = m * z[i] / H
    if alpha == 1:
        ###first row
        Coe_Matrix_copy[4] = F1[1]
        #second row
        Coe_Matrix_copy[9] = F1[1]
        Coe_Matrix_copy[11] = F1[1] * Coe_Matrix[11]
        #third row
        Coe_Matrix_copy[18] = 2 * F3[1] + Coe_Matrix[18]
        Coe_Matrix_copy[19] = F1[1]
        Coe_Matrix_copy[20] = - 2 * F1[2]
        Coe_Matrix_copy[22] = Coe_Matrix[22] * F1[2] - 2 * F1[2] * F3[1]
        #4th row
        Coe_Matrix_copy[24] = - 2 * F1[1]
        Coe_Matrix_copy[26] = (Coe_Matrix[26] - 2 * F3[1]) * F1[1]
        Coe_Matrix_copy[29] = - F1[2]
        Coe_Matrix_copy[30] = Coe_Matrix[30] + 2 * F3[1]
        # 5th row
        Coe_Matrix_copy[36] = Coe_Matrix[36] * F1[2]
        Coe_Matrix_copy[37] = Coe_Matrix[37] + 2 * F3[1] - 2 * F2[1] * F3[1]
        #6th row
        Coe_Matrix_copy[40] = Coe_Matrix[40] * F1[1]
        Coe_Matrix_copy[41] = F1[2] * Coe_Matrix[41]
        Coe_Matrix_copy[43] = - Coe_Matrix[43] * F1[2] + 2 * F1[2] * F3[1] * (1 - F2[1])
        #7th row
        Coe_Matrix_copy[50] = Coe_Matrix[50] + 2 * F3[2]
        Coe_Matrix_copy[51] = F1[2]
        Coe_Matrix_copy[52] = - 2 * F1[3]
        Coe_Matrix_copy[54] = Coe_Matrix[54] * F1[3] - 2 * F1[3] * F3[2]
        #8th row
        Coe_Matrix_copy[56] = - 2 * F1[2]
        Coe_Matrix_copy[58] = Coe_Matrix[58] * F1[2] - 2 * F1[2] * F3[2]
        Coe_Matrix_copy[61] = - F1[3]
        Coe_Matrix_copy[62] = Coe_Matrix[62] + 2 * F3[2]
        #9th row
        Coe_Matrix_copy[68] = Coe_Matrix[68] * F1[3]
        Coe_Matrix_copy[69] = Coe_Matrix[69] + 2 * F3[2] - 2 * F2[2] * F3[2]
        #10th row
        Coe_Matrix_copy[72] = Coe_Matrix[72] * F1[2]
        Coe_Matrix_copy[73] = F1[3] * Coe_Matrix[73]
        Coe_Matrix_copy[75] = - Coe_Matrix[75] * F1[3] + 2 * F1[3] * F3[2] * (1 - F2[2])
        #11th row
        Coe_Matrix_copy[82] = Coe_Matrix[82] + 2 * F3[3]
        Coe_Matrix_copy[83] = F1[3]
        Coe_Matrix_copy[84] = - 2 * F1[4]
        Coe_Matrix_copy[86] = Coe_Matrix[86] * F1[4] - 2 * F1[4] * F3[3]
        #12th row
        Coe_Matrix_copy[88] = - 2 * F1[3]
        Coe_Matrix_copy[90] = Coe_Matrix[90] * F1[3] - 2 * F1[3] * F3[3]
        Coe_Matrix_copy[93] = - F1[4]
        Coe_Matrix_copy[94] = Coe_Matrix[94] + 2 * F3[3]
        #13th row
        Coe_Matrix_copy[100] = Coe_Matrix[100] * F1[4]
        Coe_Matrix_copy[101] = Coe_Matrix[101] + 2 * F3[3] - 2 * F2[3] * F3[3]
        #14th row
        Coe_Matrix_copy[104] = Coe_Matrix[104] * F1[3]
        Coe_Matrix_copy[105] = F1[4] * Coe_Matrix[105]
        Coe_Matrix_copy[107] = - Coe_Matrix[107] * F1[4] + 2 * F1[4] * F3[3] * (1 - F2[3])
        #15th row
        Coe_Matrix_copy[114] = Coe_Matrix[114] + 2 * F3[4]
        Coe_Matrix_copy[115] = F1[4]
        #16th row
        Coe_Matrix_copy[120] = - 2 * F1[4]
        Coe_Matrix_copy[122] = Coe_Matrix[122] * F1[4] - 2 * F1[4] * F3[4]
        Coe_Matrix_copy[124] = Coe_Matrix[124] + 2 * F3[4]
        #17th row
        Coe_Matrix_copy[131] = Coe_Matrix[131] + 2 * F3[4] - 2 * F2[4] * F3[4]
        #18th row
        Coe_Matrix_copy[136] = Coe_Matrix[136] * F1[4]

    else:
        pass

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.inline   
cdef inline void Integrand_52(int index, double[64] points, double[64] weights, double* b, double[20] Coe, int[20] ipiv) noexcept nogil:
    cdef int n = 18
    cdef int kl = 2
    cdef int ku = 5
    cdef int bw = 10
    cdef int i, j, k, l, base
    cdef double row_cache ### Temporary array on the stack
    cdef double b_cache
    cdef int[18] row_permutation
    cdef int row_permutation_cache
    cdef double mu
    cdef double temp_y
    cdef double* y = b           
    cdef double tmp
    #memset(&y[0], 0, sizeof(y))   
    # b [1,0,0,...,0]
    b[0] = 1.0
    row_permutation[0] = 1
    for i in range(1,n):
        b[i] = 0.0
        row_permutation[i] = i + 1
    ### col: 0
    ## row:0 switch row:1
    for i in range(4):
        row_cache = Coe_Matrix_copy[2 + i]
        Coe_Matrix_copy[2 + i] = Coe_Matrix_copy[9 + i]
        Coe_Matrix_copy[9 + i] = row_cache
    b_cache = b[0]
    row_permutation_cache = row_permutation[0]
    b[0] = b[1]
    row_permutation[0] = row_permutation[1]
    b[1] = b_cache
    row_permutation[1] = row_permutation_cache
    ## Disposal
    mu = Coe_Matrix_copy[9] / Coe_Matrix_copy[2]
    for i in range(4):
        Coe_Matrix_copy[9 + i] -= mu * Coe_Matrix_copy[2 + i]
    b[1] -= mu * b[0]
    Coe_Matrix_copy[9] = mu  #L
    mu = Coe_Matrix_copy[16] / Coe_Matrix_copy[2]
    for i in range(4):
        Coe_Matrix_copy[16 + i] -= mu * Coe_Matrix_copy[2 + i]
    b[2] -= mu * b[0]
    Coe_Matrix_copy[16] = mu  #L
    ### col:1
    mu = Coe_Matrix_copy[17] / Coe_Matrix_copy[10]
    for i in range(3):
        Coe_Matrix_copy[17 + i] -= mu * Coe_Matrix_copy[10 + i]
    b[2] -= mu * b[1]
    Coe_Matrix_copy[17] = mu  #L
    mu = Coe_Matrix_copy[24] / Coe_Matrix_copy[10]
    for i in range(3):
        Coe_Matrix_copy[24 + i] -= mu * Coe_Matrix_copy[10 + i]
    b[3] -= mu * b[1]
    Coe_Matrix_copy[24] = mu  #L
    ### col:2
    mu = Coe_Matrix_copy[25] / Coe_Matrix_copy[18]
    for i in range(6):
        Coe_Matrix_copy[25 + i] -= mu * Coe_Matrix_copy[18 + i]
    b[3] -= mu * b[2]
    Coe_Matrix_copy[25] = mu  #L
    mu = Coe_Matrix_copy[32] / Coe_Matrix_copy[18]
    for i in range(6):
        Coe_Matrix_copy[32 + i] -= mu * Coe_Matrix_copy[18 + i]
    b[4] -= mu * b[2]
    Coe_Matrix_copy[32] = mu  #L
    ### col:3-14
    for k in range(3):
        ## col: 3 + k * 4----- 3->7->11
        mu = Coe_Matrix_copy[33 + 32 * k] / Coe_Matrix_copy[26 + 32 * k]
        for i in range(5):
            Coe_Matrix_copy[33 + 32 * k + i] -= mu * Coe_Matrix_copy[26 + 32 * k + i]
        b[4 + 4 * k] -= mu * b[3 + 4 * k]
        Coe_Matrix_copy[33 + 32 * k] = mu  #L
        mu = Coe_Matrix_copy[40 + 32 * k] / Coe_Matrix_copy[26 + 32 * k]
        for i in range(5):
            Coe_Matrix_copy[40 + 32 * k + i] -= mu * Coe_Matrix_copy[26 + 32 * k + i]
        b[5 + 4 * k] -= mu * b[3 + 4 * k]
        Coe_Matrix_copy[40 + 32 * k] = mu  #L
        ### col: 4 + k * 4----- 4->8->12
        ## switch row

        mu = Coe_Matrix_copy[41 + 32 * k] / Coe_Matrix_copy[34 + 32 * k]
        for i in range(4):
            Coe_Matrix_copy[41 + 32 * k + i] -= mu * Coe_Matrix_copy[34 + 32 * k + i]
        b[5 + 4 * k] -= mu * b[4 + 4 * k]
        Coe_Matrix_copy[41 + 32 * k] = mu  #L
        mu = Coe_Matrix_copy[48 + 32 * k] / Coe_Matrix_copy[34 + 32 * k]
        for i in range(4):
            Coe_Matrix_copy[48 + 32 * k + i] -= mu * Coe_Matrix_copy[34 + 32 * k + i]
        b[6 + 4 * k] -= mu * b[4 + 4 * k]
        Coe_Matrix_copy[48 + 32 * k] = mu  #L
        ### col: 5 + k * 4----- 5->9->13
        mu = Coe_Matrix_copy[49 + 32 * k] / Coe_Matrix_copy[42 + 32 * k]
        for i in range(3):
            Coe_Matrix_copy[49 + 32 * k + i] -= mu * Coe_Matrix_copy[42 + 32 * k + i]
        b[6 + 4 * k] -= mu * b[5 + 4 * k]
        Coe_Matrix_copy[49 + 32 * k] = mu  #L
        mu = Coe_Matrix_copy[56 + 32 * k] / Coe_Matrix_copy[42 + 32 * k]
        for i in range(3):
            Coe_Matrix_copy[56 + 32 * k + i] -= mu * Coe_Matrix_copy[42 + 32 * k + i]
        b[7 + 4 * k] -= mu * b[5 + 4 * k]
        Coe_Matrix_copy[56 + 32 * k] = mu  #L
        ### col: 6 + k * 4----- 6->10->14
        mu = Coe_Matrix_copy[57 + 32 * k] / Coe_Matrix_copy[50 + 32 * k]
        for i in range(6):
            Coe_Matrix_copy[57 + 32 * k + i] -= mu * Coe_Matrix_copy[50 + 32 * k + i]
        b[7 + 4 * k] -= mu * b[6 + 4 * k]
        Coe_Matrix_copy[57 + 32 * k] = mu  #L
        mu = Coe_Matrix_copy[64 + 32 * k] / Coe_Matrix_copy[50 + 32 * k]
        for i in range(6):
            Coe_Matrix_copy[64 + 32 * k + i] -= mu * Coe_Matrix_copy[50 + 32 * k + i]
        b[8 + 4 * k] -= mu * b[6 + 4 * k]
        Coe_Matrix_copy[64 + 32 * k] = mu  #L
    ### col:15
    mu = Coe_Matrix_copy[129] / Coe_Matrix_copy[122]
    for i in range(3):
        Coe_Matrix_copy[129 + i] -= mu * Coe_Matrix_copy[122 + i]
    b[16] -= mu * b[15]
    Coe_Matrix_copy[129] = mu  #L
    mu = Coe_Matrix_copy[136] / Coe_Matrix_copy[122]
    for i in range(3):
        Coe_Matrix_copy[136 + i] -= mu * Coe_Matrix_copy[122 + i]
    b[17] -= mu * b[15]
    Coe_Matrix_copy[136] = mu  #L
    ### col:16
    mu = Coe_Matrix_copy[137] / Coe_Matrix_copy[130]
    for i in range(2):
        Coe_Matrix_copy[137 + i] -= mu * Coe_Matrix_copy[130 + i]
    b[17] -= mu * b[16]
    Coe_Matrix_copy[137] = mu  #L
    ### Compute LUx = e
    Coe[17] = y[17] / Coe_Matrix_copy[138]

    tmp     = y[16] - Coe_Matrix_copy[131] * Coe[17]
    Coe[16] = tmp   / Coe_Matrix_copy[130]

    tmp     = y[15] - (Coe_Matrix_copy[123] * Coe[16]
                    + Coe_Matrix_copy[124] * Coe[17])
    Coe[15] = tmp   / Coe_Matrix_copy[122]

    tmp     = y[14] - (Coe_Matrix_copy[115] * Coe[15]
                    + Coe_Matrix_copy[116] * Coe[16]
                    + Coe_Matrix_copy[117] * Coe[17])
    Coe[14] = tmp   / Coe_Matrix_copy[114]

    tmp     = y[13] - (Coe_Matrix_copy[107] * Coe[14]
                    + Coe_Matrix_copy[108] * Coe[15]
                    + Coe_Matrix_copy[109] * Coe[16]
                    + Coe_Matrix_copy[110] * Coe[17])
    Coe[13] = tmp   / Coe_Matrix_copy[106]
    # 12->0 
    for k in range(12, -1, -1):            # k = 12 .. 0
        # 12-k ，index = 98 - 8*(12-k)
        base = 98 - 8 * (12 - k)

        tmp  = y[k]                         \
            - Coe_Matrix_copy[base + 1] * Coe[k + 1] \
            - Coe_Matrix_copy[base + 2] * Coe[k + 2] \
            - Coe_Matrix_copy[base + 3] * Coe[k + 3] \
            - Coe_Matrix_copy[base + 4] * Coe[k + 4] \
            - Coe_Matrix_copy[base + 5] * Coe[k + 5]

        Coe[k] = tmp / Coe_Matrix_copy[base]
    Coe[19] = Coe[17]
    Coe[17] = Coe[16]
    Coe[16] = 0.0
    Coe[18] = 0.0
    
cdef bint _initialized = False      
cdef double[4] _cached_Coe
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.inline   
cdef inline void Integrand_IA(int index, double[64] points, double[64] weights, double* b, double[20] Coe, int[20] ipiv) noexcept nogil:
    cdef double cache_D 
    cdef double[2] r
    cdef int iii
    if not _initialized:
        cache_D = Coe_Matrix_copy[5] * Coe_Matrix_copy[10] - Coe_Matrix_copy[3] * Coe_Matrix_copy[12]
        Coe[1] = - Coe_Matrix_copy[12] / cache_D
        Coe[3] = Coe_Matrix_copy[10] / cache_D
        Coe[0] = 0.0
        Coe[2] = 0.0
        for iii in range(18):
            Coe[iii + 4] = 0.0 

        initialized = True
    

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double gaussian_quadrature_integrate(double* z, double* nu, double* E,
                   double* evaluation, double H, double q, double a, int alpha, double[6] F1, double[6] F2, double[6] F3, double* zeros, int Type, int intervals, int index, double[64] points, double[64] weights, double[288] Coe_matrix, double[6] b, double[20] Coe, int[20] ipiv) noexcept nogil:
    """
    Guass-Legendre integral with convergence check
    Type == 0 is displacement; Type == 1 is stress (vertical)
    zeros is zeros points
    """
    cdef double total_integral = 0
    cdef double Total_integral = 0
    cdef double integrand = 0
    cdef long* n_points_array = <long*> malloc(intervals * sizeof(long))
    cdef int i, j, k, ii
    cdef double interval_integral = 0
    cdef double[2] exp_term  
    cdef double A, B, C, D
    
    cdef long n_points_number = <long>sqrt(zeros[1])
    cdef long eva = <long>(2 * (3 + floor(3 / ((evaluation[0] / a) + 1))))
    for i in range (intervals):
        n_points_array[i] = <long>(ceil(n_points_number + 2 / sqrt(i + 1))) * 4
    if Type == 0:
        for i in range(intervals-1):
            interval_integral = 0
            gauss_legendre_points(n_points_array[i], zeros[i], zeros[i+1], points, weights)
            for j in range(n_points_array[i]):
                memcpy(Coe_Matrix_copy, Coe_Matrix, 144 * sizeof(double))
                Coefficient_52(points, weights, j, F1, F2, F3, z, nu, E, alpha, H)
                e_Index = exp(points[j] * (z[0] - z[1]) / H)
                if e_Index > 0.05:
                    Integrand_52(index, points, weights, b, Coe, ipiv)
                else:
                    Integrand_IA(index, points, weights, b, Coe, ipiv)
                exp_term[0] = exp(points[j] * (evaluation[1] / H - z[index] / H))  
                exp_term[1] = exp(- points[j] * (evaluation[1] / H - z[index - 1] / H))
                A = Coe[index * 4 - 4] * exp_term[0]      ### Coe[1] is A1
                B = - Coe[index * 4 - 3] * exp_term[1]    ### Coe[2] is B1
                C = - Coe[index * 4 - 2] * (2 - 4 * nu[index] - points[j] * evaluation[1] / H) * exp_term[0]    ### Coe[3] is C1
                D = - Coe[index * 4 - 1] * (2 - 4 * nu[index] + points[j] * evaluation[1] / H) * exp_term[1]    ### Coe[4] is D1
                integrand = weights[j] / points[j] * (A + B + C + D) * BJ0(points[j] * evaluation[0] / H) * BJ1(points[j] * a / H) 
                interval_integral += integrand 

            if abs(interval_integral) < 0.01 * abs(total_integral) and i % 2 == 1 and i > eva:
                total_integral += interval_integral * 0.5
                
                Total_integral += total_integral 
                return Total_integral * (q * a * (1 + nu[index]) / E[index])
            total_integral += interval_integral
        Total_integral += total_integral * (q * a * (1 + nu[index]) / E[index])

    else:
        for i in range(intervals-1):
            interval_integral = 0
            gauss_legendre_points(n_points_array[i], zeros[i], zeros[i+1], points, weights)
            for j in range(n_points_array[i]):
                memcpy(Coe_Matrix_copy, Coe_Matrix, 144 * sizeof(double))
                Coefficient_52(points, weights, j, F1, F2, F3, z,  nu, E, alpha, H)
                Integrand_52(index, points, weights, b, Coe, ipiv)
                exp_term[0] = exp(points[j] * (evaluation[1] / H - z[index] / H))  
                exp_term[1] = exp(- points[j] * (evaluation[1] / H - z[index - 1] / H))
                A = -Coe[index * 4 - 4] * exp_term[0]
                B = -Coe[index * 4 - 3] * exp_term[1]
                C = Coe[index * 4 - 2] * (1 - 2 * nu[index] - points[j] * evaluation[1] / H) * exp_term[0]
                D = -Coe[index * 4 - 1] * (1 - 2 * nu[index] + points[j] * evaluation[1] / H) * exp_term[1]
                integrand = weights[j] * q * a / H * (A + B + C + D) * BJ0(points[j] * evaluation[0] / H) * BJ1(points[j] * a / H)
                #integrand = weights[j] * q * a  * (A + B + C + D) * gsl_sf_bessel_J0(points[j] * evaluation[0] / H) * gsl_sf_bessel_J1(points[j] * a / H) / H
                interval_integral += integrand 
            if abs(interval_integral) < 0.25 * abs(total_integral) and i % 2 == 1:
                total_integral += interval_integral
                Total_integral += total_integral - 0.5 * interval_integral
                return Total_integral
            total_integral += interval_integral
        Total_integral += total_integral - 0.5 * interval_integral

    free(n_points_array)
    return Total_integral

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def Calculation(double[:, :] Data_array):
    ## define the parameters
    global Coe_Matrix
    global Coe_Matrix_copy
    cdef int num_cols = 10
    cdef int index, i, ii
    cdef int alpha = 1
    cdef double q = Data_array[10, 1]
    cdef double a = Data_array[10, 3]
    cdef double[6] z
    cdef double[6] nu
    cdef double[6] E
    cdef double[2] evaluation
    cdef int num_zeros = 120
    cdef double[288] Coe_matrix 
    memset(&Coe_matrix[0], 0, sizeof(double) * 288)
    cdef double[6] F1
    cdef double[6] F2
    cdef double[6] F3
    cdef double[18] b
    cdef double[20] Coe 
    cdef int[28] ipiv 
    cdef double H
    cdef double *zeros
    cdef double[64] weights
    cdef double[64] points
    cdef double[10] result_displacement
    Input_num(Data_array, z, nu, E, evaluation, num_cols)
    H = z[4]
    index = index_search(z, evaluation)
    ### define parameters
    F2[0] = (E[0] / E[1]) * ((1.0 + nu[1]) / (1.0 + nu[0]))
    F2[1] = (E[1] / E[2]) * ((1.0 + nu[2]) / (1.0 + nu[1]))
    F2[2] = (E[2] / E[3]) * ((1.0 + nu[3]) / (1.0 + nu[2]))
    F2[3] = (E[3] / E[4]) * ((1.0 + nu[4]) / (1.0 + nu[3]))
    F2[4] = (E[4] / E[5]) * ((1.0 + nu[5]) / (1.0 + nu[4]))
    cdef double start = perf_counter()
    for ii in range(num_cols):
        Input_num(Data_array, z, nu, E, evaluation, num_cols)
        H = z[4]
        index = index_search(z, evaluation)
        Input_num(Data_array, z, nu, E, evaluation, ii)
        H = z[4]
        index = index_search(z, evaluation)
        Variable_Assignment(z, nu, E, evaluation, H, q, a, F2)
        zeros = bessel_zeros(num_zeros, evaluation[0] / H, a / H)
        result_displacement[ii] = gaussian_quadrature_integrate(z, nu, E, evaluation, H, q, a, alpha, F1, F2, F3, zeros, 0, 121, index, points, weights, Coe_matrix, b, Coe, ipiv)
    cdef double end = perf_counter()
    cdef double elapsed = end - start
    print(f"Run time: {elapsed:.9f} s")
    cdef double result_stress = gaussian_quadrature_integrate(z, nu, E, evaluation, H, q, a, alpha, F1, F2, F3, zeros, 1, 121, index, points, weights, Coe_matrix, b, Coe, ipiv)
    free(zeros)
    
    return result_stress, result_displacement[0], result_displacement[1], result_displacement[2], result_displacement[3], result_displacement[4], result_displacement[5], result_displacement[6], result_displacement[7], result_displacement[8], result_displacement[9]