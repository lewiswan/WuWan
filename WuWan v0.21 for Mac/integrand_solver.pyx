#interand_solver.pyx

import cython
from cython cimport view
from math_fun cimport gauss_legendre_points, BJ0, BJ1
from libc.string cimport memcpy
from libc.math cimport exp, sqrt, floor, ceil, fabs
from libc.stdlib cimport malloc, free
from processing_function cimport Input_num, index_search, Variable_Assignment, Coefficient_52

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.inline   
cdef inline void Integrand_52(int index, double[64] points, double[64] weights, double* b, double[20] Coe, int[20] ipiv, double[144] Coe_Matrix_copy) noexcept nogil:
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
    
#cdef bint _initialized = False      
#cdef double[4] _cached_Coe
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.inline   
cdef inline void Integrand_IA(int index, double[64] points, double[64] weights, double* b, double[20] Coe, int[20] ipiv, double[144] Coe_Matrix_copy, bint* p_initialized) noexcept nogil:
    cdef double cache_D 
    cdef double[2] r
    cdef int iii
    if not p_initialized[0]:
        cache_D = Coe_Matrix_copy[5] * Coe_Matrix_copy[10] - Coe_Matrix_copy[3] * Coe_Matrix_copy[12]
        Coe[1] = - Coe_Matrix_copy[12] / cache_D
        Coe[3] = Coe_Matrix_copy[10] / cache_D
        Coe[0] = 0.0
        Coe[2] = 0.0
        for iii in range(18):
            Coe[iii + 4] = 0.0 

        p_initialized[0] = True
    

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double gaussian_quadrature_integrate(double* z, double* nu, double* E,
                   double* evaluation, double H, double q, double a, int alpha, double[6] F1, double[6] F2, double[6] F3, double* zeros, int Type, int intervals, int index, double[64] points, double[64] weights, double[144] Coe_Matrix, double[144] Coe_Matrix_copy, double[6] b, double[20] Coe, int[20] ipiv) noexcept nogil:
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
    cdef bint p_initialized = False
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
                Coefficient_52(points, weights, j, F1, F2, F3, z, nu, E, alpha, H, Coe_Matrix, Coe_Matrix_copy)
                e_Index = exp(points[j] * (z[0] - z[1]) / H)
                if e_Index > 0.05:
                    Integrand_52(index, points, weights, b, Coe, ipiv, Coe_Matrix_copy)
                else:
                    Integrand_IA(index, points, weights, b, Coe, ipiv, Coe_Matrix_copy, &p_initialized)
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
                Coefficient_52(points, weights, j, F1, F2, F3, z,  nu, E, alpha, H, Coe_Matrix, Coe_Matrix_copy)
                Integrand_52(index, points, weights, b, Coe, ipiv, Coe_Matrix_copy)
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