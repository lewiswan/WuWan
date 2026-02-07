// src/processing_function.h
#ifndef EIGEN_VECTORIZE
#define EIGEN_VECTORIZE  
#endif
#if defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif
#define EIGEN_DONT_ALIGN_STATICALLY 0 
#pragma once
#include <iostream>
#include "structures.h" 
#include <algorithm> 
#include <cstring> // for std::memset


inline void parse_input_data(const Eigen::Ref<const InputMat>& input_arr, ModelParams& params) {
    
    // Scalar loads
    params.q = input_arr(10, 1); 
    params.a = input_arr(10, 3); 

    // Depth Calculation
    params.z.head(5) = input_arr.block<5, 1>(1, 3);
    params.z[5] = 1e10; 

    // Cumulative depth 
    for (int i = 0; i < 4; ++i) {
        params.z[i+1] += params.z[i]; 
    }

    // Material Properties
    params.nu.head(6) = input_arr.block<6, 1>(1, 2);
    params.E.head(6)  = input_arr.block<6, 1>(1, 1);

    // evaluation points Params
    params.evaluation(0) = input_arr(1, 6);
    params.evaluation(1) = 0.0;

    // total depth expect infinate layer
    params.H = params.z(4);

    // pre-calculation F2
    // numerator: E[i] * (1 + nu[i+1])
    auto numerator = params.E.head(5).array() * (1.0 + params.nu.tail(5).array());
    // denominator: E[i+1] * (1 + nu[i])
    auto denominator = params.E.tail(5).array() * (1.0 + params.nu.head(5).array());
    params.F2 = (numerator / denominator).matrix();
}

inline int index_search(const Vec6& z, double target){
    auto it = std::lower_bound(z.data(), z.data() + 6, target); // std::lower_bound --> first element >= target
    int low = std::distance(z.data(), it); // compute index
    if (low <= 0) {
        low = 1;
    }
    if (low > 5) {
        low = 5; 
    }
    return low;
}

inline void Variable_Assignment(ModelParams& params) {
    double* M = params.Coe_Matrix_Template.data();
    const double* nu = params.nu.data();
    const double* F2 = params.F2.data();

    std::memset(M, 0, sizeof(double) * 144);

    // --- compute cache ---
    double nu1_x2     = 2.0 * nu[1];
    double term_nu1   = 2.0 * nu1_x2 - 1.0; // 4*nu[1] - 1
    
    double term_nu2   = 1.0 - 4.0 * nu[2];
    double term_nu3   = 1.0 - 4.0 * nu[3];
    double term_nu4   = 1.0 - 4.0 * nu[4];
    double term_nu5   = 1.0 - 4.0 * nu[5];

    // --- F2 / Block terms  ---
    // Block 1 (Layer 2 interface)
    double f2_1       = F2[1];
    double b1_A       = 2.0 - 2.0 * f2_1;            // 2 - 2*F2
    double b1_B       = f2_1 * (4.0 * nu[2] - 3.0) - 1.0; // 4*F2*nu - 3*F2 - 1
    double b1_C       = 1.0 - f2_1 - 4.0 * nu[2] + 4.0 * f2_1 * nu[2]; // 1 - F2 - 4nu + 4F2nu

    // Block 2 (Layer 3 interface)
    double f2_2       = F2[2];
    double b2_A       = 2.0 - 2.0 * f2_2;
    double b2_B       = f2_2 * (4.0 * nu[3] - 3.0) - 1.0;
    double b2_C       = 1.0 - f2_2 - 4.0 * nu[3] + 4.0 * f2_2 * nu[3];

    // Block 3 (Layer 4 interface)
    double f2_3       = F2[3];
    double b3_A       = 2.0 - 2.0 * f2_3;
    double b3_B       = f2_3 * (4.0 * nu[4] - 3.0) - 1.0;
    double b3_C       = 1.0 - f2_3 - 4.0 * nu[4] + 4.0 * f2_3 * nu[4];

    // Block 4 (Layer 5 interface)
    double f2_4       = F2[4];
    double b4_A       = 2.0 - 2.0 * f2_4;
    double b4_C       = 1.0 - f2_4 - 4.0 * nu[5] + 4.0 * f2_4 * nu[5];
    double b4_D       = 1.0 + 3.0 * f2_4 - 4.0 * f2_4 * nu[5]; // Row 17 特有项


    // --- Row 0 ---
    M[3]  = -2.0;
    M[5]  = term_nu1; // Cached: 4*nu[1]-1

    // --- Row 1 ---
    M[10] = -1.0;     // 1*8 + 2
    M[11] = nu1_x2;
    M[12] = nu1_x2;

    // --- Row 2 ---
    M[16] = 2.0;
    M[18] = term_nu1; // Cached
    M[22] = term_nu2; // Cached: 1-4*nu[2]
    M[23] = -1.0;

    // --- Row 3 ---
    M[25] = 1.0;
    M[26] = term_nu1; // Cached
    M[28] = 2.0;
    M[30] = term_nu2; // Cached

    // --- Row 4 ---
    M[32] = -term_nu1 + 3.0; // Cached: -M(0,5) + 3
    M[35] = b1_A;
    M[36] = b1_B;
    M[37] = b1_C;

    // --- Row 5 ---
    M[40] = term_nu1 - 3.0;  // Cached: M(0,5) - 3
    M[41] = b1_A;
    M[43] = b1_C;            // Avoid Read-After-Write: M(4,5)
    M[44] = -b1_B;           // Avoid Read-After-Write: -M(4,4)

    // --- Row 6 ---
    M[48] = 2.0;
    M[50] = -term_nu2;       // -M(2,6)
    M[54] = term_nu3;
    M[55] = -1.0;

    // --- Row 7 ---
    M[57] = 1.0;
    M[58] = -term_nu2;       // -M(2,6)
    M[60] = 2.0;
    M[62] = term_nu3;        // M(6,6)

    // --- Row 8 ---
    M[64] = term_nu2 + 3.0;  // M(2,6) + 3
    M[67] = b2_A;
    M[68] = b2_B;
    M[69] = b2_C;

    // --- Row 9 ---
    M[72] = -term_nu2 - 3.0; // -M(2,6) - 3
    M[73] = b2_A;
    M[75] = b2_C;            // Cached
    M[76] = -b2_B;           // Cached

    // --- Row 10 ---
    M[80] = 2.0;
    M[82] = -term_nu3;       // -M(6,6)
    M[86] = term_nu4;
    M[87] = -1.0;

    // --- Row 11 ---
    M[89] = 1.0;
    M[90] = -term_nu3;       // -M(6,6)
    M[92] = 2.0;
    M[94] = term_nu4;        // M(10,6)

    // --- Row 12 ---
    M[96] = term_nu3 + 3.0;  // M(6,6) + 3
    M[99] = b3_A;
    M[100] = b3_B;
    M[101] = b3_C;

    // --- Row 13 ---
    M[104] = -term_nu3 - 3.0;// -M(6,6) - 3
    M[105] = b3_A;
    M[107] = b3_C;           // Cached
    M[108] = -b3_B;          // Cached

    // --- Row 14 (Pattern Shift) ---
    M[112] = 2.0;
    M[114] = -term_nu4;      // -M(10,6)
    M[117] = -1.0;

    // --- Row 15 ---
    M[121] = 1.0;
    M[122] = -term_nu4;      // -M(10,6)
    M[123] = 2.0;
    M[124] = term_nu5;

    // --- Row 16 ---
    M[128] = term_nu4 + 3.0; // M(10,6) + 3
    M[130] = b4_A;
    M[131] = b4_C;

    // --- Row 17 ---
    M[136] = -term_nu4 - 3.0;// -M(10,6) - 3
    M[138] = b4_D;
}

inline void Coefficient_52(CalcBuffer& buffer, const ModelParams& params, int j){
    double* RESTRICT M = buffer.Coe_Matrix.data(); 
    const double* RESTRICT T = params.Coe_Matrix_Template.data();

    const double* RESTRICT z = params.z.data(); 
    const double* RESTRICT F2 = params.F2.data();
    double* RESTRICT F1 = buffer.F1.data(); 
    double* RESTRICT F3 = buffer.F3.data();

    double m = buffer.points(j);
    double inv_H = 1.0 / params.H; 
    double m_inv_H = m * inv_H;

    // k=1
    F1[1] = std::exp(m_inv_H * (z[0] - z[1]));
    F3[1] = m_inv_H * z[1];

    // k=2
    F1[2] = std::exp(m_inv_H * (z[1] - z[2]));
    F3[2] = m_inv_H * z[2];

    // k=3
    F1[3] = std::exp(m_inv_H * (z[2] - z[3]));
    F3[3] = m_inv_H * z[3];

    // k=4
    F1[4] = std::exp(m_inv_H * (z[3] - z[4]));
    F3[4] = m_inv_H * z[4];

    double f1_1 = F1[1];
    double f1_2 = F1[2];
    double f1_3 = F1[3];
    double f1_4 = F1[4];

    double f3_1_x2 = 2.0 * F3[1];
    double f3_2_x2 = 2.0 * F3[2];
    double f3_3_x2 = 2.0 * F3[3];
    double f3_4_x2 = 2.0 * F3[4];

    double cross_term_1 = f3_1_x2 * (1.0 - F2[1]);
    double cross_term_2 = f3_2_x2 * (1.0 - F2[2]);
    double cross_term_3 = f3_3_x2 * (1.0 - F2[3]);
    double cross_term_4 = f3_4_x2 * (1.0 - F2[4]);
    
    // --- Row 0 (Offset 0) ---
    M[4] = f1_1;

    // --- Row 1 (Offset 8) ---
    M[9]  = f1_1;
    M[11] = f1_1 * T[11];

    // --- Row 2 (Offset 16) ---
    M[18] = f3_1_x2 + T[18];
    M[19] = f1_1;
    M[20] = -2.0 * f1_2;
    // 优化: T[22]*F1[2] - 2*F1[2]*F3[1]  =>  F1[2] * (T[22] - 2*F3[1])
    M[22] = f1_2 * (T[22] - f3_1_x2); 

    // --- Row 3 (Offset 24) ---
    M[24] = -2.0 * f1_1;
    M[26] = f1_1 * (T[26] - f3_1_x2);
    M[29] = -f1_2;
    M[30] = T[30] + f3_1_x2;

    // --- Row 4 (Offset 32) ---
    M[36] = T[36] * f1_2;
    // 优化: T[37] + 2*F3[1] - 2*F2[1]*F3[1] => T[37] + 2*F3[1]*(1-F2[1])
    M[37] = T[37] + cross_term_1; 

    // --- Row 5 (Offset 40) ---
    M[40] = T[40] * f1_1;
    M[41] = f1_2 * T[41];
    M[43] = -T[43] * f1_2 + f1_2 * cross_term_1; 

    // --- Row 6 (Offset 48) ---
    M[50] = T[50] + f3_2_x2;
    M[51] = f1_2;
    M[52] = -2.0 * f1_3;
    M[54] = f1_3 * (T[54] - f3_2_x2);

    // --- Row 7 (Offset 56) ---
    M[56] = -2.0 * f1_2;
    M[58] = f1_2 * (T[58] - f3_2_x2);
    M[61] = -f1_3;
    M[62] = T[62] + f3_2_x2;

    // --- Row 8 (Offset 64) ---
    M[68] = T[68] * f1_3;
    M[69] = T[69] + cross_term_2;

    // --- Row 9 (Offset 72) ---
    M[72] = T[72] * f1_2;
    M[73] = f1_3 * T[73];
    M[75] = -T[75] * f1_3 + f1_3 * cross_term_2;

    // --- Row 10 (Offset 80) ---
    M[82] = T[82] + f3_3_x2;
    M[83] = f1_3;
    M[84] = -2.0 * f1_4;
    M[86] = f1_4 * (T[86] - f3_3_x2);

    // --- Row 11 (Offset 88) ---
    M[88] = -2.0 * f1_3;
    M[90] = f1_3 * (T[90] - f3_3_x2);
    M[93] = -f1_4;
    M[94] = T[94] + f3_3_x2;

    // --- Row 12 (Offset 96) ---
    M[100] = T[100] * f1_4;
    M[101] = T[101] + cross_term_3;

    // --- Row 13 (Offset 104) ---
    M[104] = T[104] * f1_3;
    M[105] = f1_4 * T[105];
    M[107] = -T[107] * f1_4 + f1_4 * cross_term_3;

    // --- Row 14 (Offset 112) ---
    M[114] = T[114] + f3_4_x2;
    M[115] = f1_4;

    // --- Row 15 (Offset 120) ---
    M[120] = -2.0 * f1_4;
    M[122] = f1_4 * (T[122] - f3_4_x2);
    M[124] = T[124] + f3_4_x2;

    // --- Row 16 (Offset 128) ---
    M[131] = T[131] + cross_term_4;

    // --- Row 17 (Offset 136) ---
    M[136] = T[136] * f1_4;
}

inline void Derivative_E(CalcBuffer& buffer, const ModelParams& params){
    Vec42& D = buffer.DCoefficientDE;
    const Vec6& E = params.E;
    const Vec6& nu = params.nu;
    const Vec5& F2 = params.F2;
    Vec5& F1 = buffer.F1;
    Vec5& F3 = buffer.F3;

    double F21_E1 =  F2[1] / E[1];
    double F21_E2 = -F2[1] / E[2];
    
    double F22_E2 =  F2[2] / E[2];
    double F22_E3 = -F2[2] / E[3];
    
    double F23_E3 =  F2[3] / E[3];
    double F23_E4 = -F2[3] / E[4];
    
    double F24_E4 =  F2[4] / E[4];
    double F24_E5 = -F2[4] / E[5];

    // ### for E1 (Indices 0-5)
    D[0] = -2.0 * F21_E1;
    D[1] = (4.0 * F1[2] * nu[2] - 3.0 * F1[2]) * F21_E1;
    D[2] = (4.0 * nu[2] - 2.0 * F3[1] - 1.0) * F21_E1;
    D[3] = -2.0 * F1[2] * F21_E1;
    D[4] = (-2.0 * F1[2] * F3[1] - 4.0 * F1[2] * nu[2] + F1[2]) * F21_E1;
    D[5] = (3.0 - 4.0 * nu[2]) * F21_E1;

    // ### for E2 (Indices 6-17)
    // Part A: Interaction with Layer 1 interface
    D[6]  = -2.0 * F21_E2;
    D[7]  = (4.0 * F1[2] * nu[2] - 3.0 * F1[2]) * F21_E2;
    D[8]  = (4.0 * nu[2] - 2.0 * F3[1] - 1.0) * F21_E2;
    D[9]  = -2.0 * F1[2] * F21_E2;
    D[10] = (-2.0 * F1[2] * F3[1] - 4.0 * F1[2] * nu[2] + F1[2]) * F21_E2;
    D[11] = (3.0 - 4.0 * nu[2]) * F21_E2;
    
    // Part B: Interaction with Layer 2 interface
    D[12] = -2.0 * F22_E2;
    D[13] = (4.0 * F1[3] * nu[3] - 3.0 * F1[3]) * F22_E2;
    D[14] = (4.0 * nu[3] - 2.0 * F3[2] - 1.0) * F22_E2;
    D[15] = -2.0 * F1[3] * F22_E2;
    D[16] = (-2.0 * F1[3] * F3[2] - 4.0 * F1[3] * nu[3] + F1[3]) * F22_E2;
    D[17] = (3.0 - 4.0 * nu[3]) * F22_E2;

    // ### for E3 (Indices 18-29)
    // Part A
    D[18] = -2.0 * F22_E3;
    D[19] = (4.0 * F1[3] * nu[3] - 3.0 * F1[3]) * F22_E3;
    D[20] = (4.0 * nu[3] - 2.0 * F3[2] - 1.0) * F22_E3;
    D[21] = -2.0 * F1[3] * F22_E3;
    D[22] = (-2.0 * F1[3] * F3[2] - 4.0 * F1[3] * nu[3] + F1[3]) * F22_E3;
    D[23] = (3.0 - 4.0 * nu[3]) * F22_E3;
    
    // Part B
    D[24] = -2.0 * F23_E3;
    D[25] = (4.0 * F1[4] * nu[4] - 3.0 * F1[4]) * F23_E3;
    D[26] = (4.0 * nu[4] - 2.0 * F3[3] - 1.0) * F23_E3;
    D[27] = -2.0 * F1[4] * F23_E3;
    D[28] = (-2.0 * F1[4] * F3[3] - 4.0 * F1[4] * nu[4] + F1[4]) * F23_E3;
    D[29] = (3.0 - 4.0 * nu[4]) * F23_E3;

    // ### for E4 (Indices 30-38)
    // Part A
    D[30] = -2.0 * F23_E4;
    D[31] = (4.0 * F1[4] * nu[4] - 3.0 * F1[4]) * F23_E4;
    D[32] = (4.0 * nu[4] - 2.0 * F3[3] - 1.0) * F23_E4;
    D[33] = -2.0 * F1[4] * F23_E4;
    D[34] = (-2.0 * F1[4] * F3[3] - 4.0 * F1[4] * nu[4] + F1[4]) * F23_E4;
    D[35] = (3.0 - 4.0 * nu[4]) * F23_E4;
    
    // Part B 
    D[36] = -2.0 * F24_E4;
    D[37] = (4.0 * nu[5] - 2.0 * F3[4] - 1.0) * F24_E4;
    D[38] = (3.0 - 4.0 * nu[5]) * F24_E4;

    // ### for E5 (Indices 39-41)
    D[39] = -2.0 * F24_E5;
    D[40] = (4.0 * nu[5] - 2.0 * F3[4] - 1.0) * F24_E5;
    D[41] = (3.0 - 4.0 * nu[5]) * F24_E5;
}

inline void Coefficient_IA(CalcBuffer& buffer, const ModelParams& params, int j){
    BandMat18x8& M = buffer.Coe_Matrix;
    const BandMat18x8& M_Template = params.Coe_Matrix_Template;
    Vec5& F1 = buffer.F1; 
    Vec5& F3 = buffer.F3; 
    const Vec6& z = params.z;  
    double H = params.H;
    double m = buffer.points(j);
    for (int k = 1; k <= 2; ++k) {
        F1[k] = std::exp(m * (z(k - 1) - z(k)) / H);
        F3[k] = m * z(k) / H;
    }
    // --- Row 0 ---
    M(0, 4) = F1[1];

    // --- Row 1 ---
    M(1, 1) = F1[1];
    M(1, 3) = F1[1] * M_Template(1, 3);

    // --- Row 2 ---
    M(2, 2) = 2.0 * F3[1] + M_Template(2, 2);
    M(2, 3) = F1[1];
    M(2, 4) = -2.0 * F1[2]; 
    M(2, 6) = M_Template(2, 6) * F1[2] - 2.0 * F1[2] * F3[1];

    // --- Row 3 ---
    M(3, 0) = -2.0 * F1[1];
    M(3, 2) = (M_Template(3, 2) - 2.0 * F3[1]) * F1[1];
    M(3, 5) = -F1[2]; // F1[k+1] -> F1[2]
    M(3, 6) = M_Template(3, 6) + 2.0 * F3[1];
}