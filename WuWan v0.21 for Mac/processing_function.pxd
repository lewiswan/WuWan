# processing_function.pxd

cdef void Input_num(double[:, :] Data_array, double* z, double* nu, double* E, double* evaluation, int index_point) noexcept nogil
cdef int index_search(double *z, double *evaluation) noexcept nogil
cdef void Variable_Assignment(double* z, double* nu, double* E, double* evaluation, double H, double q, double a, double* F2, double[144] Coe_Matrix) noexcept nogil
cdef void Coefficient_52(double[64] points, double[64] weights, int point_iter, double[6] F1, double[6] F2, double[6] F3, double* z, double* nu, double* E, int alpha, double H, double[144] Coe_Matrix,double[144] Coe_Matrix_copy) noexcept nogil
