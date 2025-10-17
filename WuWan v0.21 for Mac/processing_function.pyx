# forwar_solver.pyx
import cython
from cython cimport view
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs, M_PI, cos, sin, pow, cbrt
from libc.string cimport memcpy, memset
from libc.math cimport floor, ceil, exp
from libc.stdlib cimport malloc, free

#from math_fun cimport BJ0, BJ1

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

    evaluation[0] = Data_array[index_point + 1, 6]
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
cdef void Variable_Assignment(double* z, double* nu, double* E, double* evaluation, double H, double q, double a, double* F2, double[144] Coe_Matrix) noexcept nogil:
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
@cython.inline    
cdef inline void Coefficient_52(double[64] points, double[64] weights, int point_iter, double[6] F1, double[6] F2, double[6] F3, double* z, double* nu, double* E, int alpha, double H, double[144] Coe_Matrix,double[144] Coe_Matrix_copy) noexcept nogil:
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
