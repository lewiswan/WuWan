from cython cimport view

cdef void Integrand_52(int index, double[64] points, double[64] weights, double* b, double[20] Coe, int[20] ipiv, double[144] Coe_Matrix_copy) noexcept nogil
cdef void Integrand_IA(int index, double[64] points, double[64] weights, double* b, double[20] Coe, int[20] ipiv, double[144] Coe_Matrix_copy, bint* p_initialized) noexcept nogil
cdef double gaussian_quadrature_integrate(double* z, double* nu, double* E, double* evaluation, double H, double q, double a, int alpha, double[6] F1, double[6] F2, double[6] F3, double* zeros, int Type, int intervals, int index, double[64] points, double[64] weights, double[144] Coe_Matrix, double[144] Coe_Matrix_copy, double[6] b, double[20] Coe, int[20] ipiv) noexcept nogil
 