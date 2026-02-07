// src/math_fun.h
#ifndef EIGEN_VECTORIZE
#define EIGEN_VECTORIZE  
#endif

#define EIGEN_DONT_ALIGN_STATICALLY 0 
#pragma once
#include <iostream>
#include "structures.h" 
#include "bessel_table.h"
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <cmath>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


template <int N>
inline void compute_gauss_fixed_n(double a, double b, Vec64& points, Vec64& weights) {
    static const auto gl = boost::math::quadrature::gauss<double, N>();
    
    double half_len = 0.5 * (b - a);
    double mid_pt   = 0.5 * (b + a);
    
    const auto& abs = gl.abscissa();
    const auto& wts = gl.weights();
    
    int half_n = static_cast<int>(abs.size());
    int idx = 0;
    
    if (N % 2 == 0) {
        for (int i = half_n - 1; i >= 0; --i) {
            points[idx]  = mid_pt - half_len * abs[i];
            weights[idx] = half_len * wts[i];
            ++idx;
        }
        for (int i = 0; i < half_n; ++i) {
            points[idx]  = mid_pt + half_len * abs[i];
            weights[idx] = half_len * wts[i];
            ++idx;
        }
    } else {
        for (int i = half_n - 1; i >= 1; --i) {
            points[idx]  = mid_pt - half_len * abs[i];
            weights[idx] = half_len * wts[i];
            ++idx;
        }
        points[idx]  = mid_pt;  // abs[0] = 0
        weights[idx] = half_len * wts[0];
        ++idx;
        for (int i = 1; i < half_n; ++i) {
            points[idx]  = mid_pt + half_len * abs[i];
            weights[idx] = half_len * wts[i];
            ++idx;
        }
    }
}

inline void gauss_legendre_point(int n, double a, double b, Vec64& points, Vec64& weights) {
    switch (n) {
        case 4:  compute_gauss_fixed_n<4> (a, b, points, weights); break;
        case 8:  compute_gauss_fixed_n<8> (a, b, points, weights); break;
        case 12: compute_gauss_fixed_n<12>(a, b, points, weights); break;
        case 16: compute_gauss_fixed_n<16>(a, b, points, weights); break;
        case 20: compute_gauss_fixed_n<20>(a, b, points, weights); break;
        case 24: compute_gauss_fixed_n<24>(a, b, points, weights); break;
        case 28: compute_gauss_fixed_n<28>(a, b, points, weights); break;
        case 32: compute_gauss_fixed_n<32>(a, b, points, weights); break;
        case 36: compute_gauss_fixed_n<36>(a, b, points, weights); break;
        case 40: compute_gauss_fixed_n<40>(a, b, points, weights); break;
        case 44: compute_gauss_fixed_n<44>(a, b, points, weights); break;
        case 48: compute_gauss_fixed_n<48>(a, b, points, weights); break;
        case 52: compute_gauss_fixed_n<52>(a, b, points, weights); break;
        case 56: compute_gauss_fixed_n<56>(a, b, points, weights); break;
        case 60: compute_gauss_fixed_n<60>(a, b, points, weights); break;
        case 64: compute_gauss_fixed_n<64>(a, b, points, weights); break;
        
        default:
            std::cerr << "Error: unsupported n = " << n << ", fallback to 64." << std::endl;
            compute_gauss_fixed_n<64>(a, b, points, weights);
            break;
    }
}


inline double BJ0(double value){
    double cache_1 = value * value;
    double cache_r;
    double cache_p = -0.25 * cache_1;
    double cache_2;
    double cache_3;
    if (value > 6.0){
        cache_2 = value - 0.25 * M_PI;
        cache_3 = 1.0 / (4.0 * cache_1);
        cache_r = std::sqrt(2.0 / M_PI / value) * (
            std::cos(cache_2) * (1.0 - 0.28125 * cache_3 + 1.79443359375 * cache_3 * cache_3) 
            - std::sin(cache_2) * (-0.125 + 0.5859375 * cache_3 * 2.0 - 7.2674560546875 * cache_3 * cache_3 * 2.0) / value
        );
    } else {
        cache_r = 1.0 + cache_p;
        
        cache_p *= -cache_1 / 16.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 36.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 64.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 100.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 144.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 196.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 256.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 324.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 400.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 484.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 576.0;
        cache_r += cache_p;
    }
    return cache_r;
}

inline double BJ1(double value) {
    double cache_1 = value * value;
    double cache_r;
    double cache_p = 0.5 * value;
    double cache_2;
    double cache_3;

    if (value > 6.0) {
        cache_2 = value - 0.75 * M_PI;
        cache_3 = 1.0 / (4.0 * cache_1);
        cache_r = std::sqrt(2.0 / M_PI / value) * (
            std::cos(cache_2) * (1.0 + 0.46875 * cache_3 - 2.30712890625 * cache_3 * cache_3) 
            - std::sin(cache_2) * (0.375 - 0.8203125 * cache_3 / 2.0 + 8.8824462890625 * cache_3 * cache_3 / 2.0) / value
        );
    } else {
        cache_r = cache_p;
        
        cache_p *= -cache_1 / 8.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 24.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 48.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 80.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 120.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 168.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 224.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 288.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 360.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 440.0;
        cache_r += cache_p;
        
        cache_p *= -cache_1 / 528.0;
        cache_r += cache_p;
    }
    return cache_r;
}


inline void bessel_zeros(CalcBuffer& buffer, double r, double a) {
    constexpr int num_zeros = 120;
    
    // 获取编译时/首次运行时生成的表
    const auto& j0_table = BesselZeros::J0();
    const auto& j1_table = BesselZeros::J1();
    
    buffer.zeros[0] = 0.0;
    
    if (r > 1e-10) {
        const double inv_r = 1.0 / r;
        const double inv_a = 1.0 / a;
        
        int j = 0, k = 0;
        for (int i = 0; i < num_zeros; ++i) {
            double z0_scaled = j0_table[k] * inv_r;
            double z1_scaled = j1_table[j] * inv_a;
            
            if (z0_scaled > z1_scaled) {
                buffer.zeros[i + 1] = z1_scaled;
                ++j;
            } else {
                buffer.zeros[i + 1] = z0_scaled;
                ++k;
            }
        }
    } else {
        const double inv_a = 1.0 / a;
        for (int i = 0; i < num_zeros; ++i) {
            buffer.zeros[i + 1] = j1_table[i] * inv_a;
        }
    }
}

inline void Integrand_52(CalcBuffer& buffer, const ModelParams& params) {
    double* __restrict__ M = buffer.Coe_Matrix.data(); 
    double* __restrict__ vb = buffer.b.data();
    double* __restrict__ Coe = buffer.Coe.data();
    
    for (int i = 0; i < 18; ++i) vb[i] = 0.0;
    vb[0] = 1.0;
    
    // =========================================================
    // Part 1: Forward Elimination 
    // =========================================================
    
    // Swap
    double t0 = M[2], t1 = M[3], t2 = M[4], t3 = M[5];
    M[2] = M[9];  M[3] = M[10]; M[4] = M[11]; M[5] = M[12];
    M[9] = t0;    M[10] = t1;   M[11] = t2;   M[12] = t3;
    double tb = vb[0]; vb[0] = vb[1]; vb[1] = tb;
    
    // Col 0
    double mu = M[9] / M[2];
    M[10] -= mu * M[3]; M[11] -= mu * M[4]; M[12] -= mu * M[5];
    vb[1] -= mu * vb[0]; M[9] = mu;
    
    mu = M[16] / M[2];
    M[17] -= mu * M[3]; M[18] -= mu * M[4]; M[19] -= mu * M[5];
    vb[2] -= mu * vb[0]; M[16] = mu;

    // Col 1
    mu = M[17] / M[10];
    M[18] -= mu * M[11]; M[19] -= mu * M[12];
    vb[2] -= mu * vb[1]; M[17] = mu;
    
    mu = M[24] / M[10];
    M[25] -= mu * M[11]; M[26] -= mu * M[12];
    vb[3] -= mu * vb[1]; M[24] = mu;

    // Col 2
    mu = M[25] / M[18];
    M[26] -= mu * M[19]; M[27] -= mu * M[20]; M[28] -= mu * M[21];
    M[29] -= mu * M[22]; M[30] -= mu * M[23];
    vb[3] -= mu * vb[2]; M[25] = mu;
    
    mu = M[32] / M[18];
    M[33] -= mu * M[19]; M[34] -= mu * M[20]; M[35] -= mu * M[21];
    M[36] -= mu * M[22]; M[37] -= mu * M[23];
    vb[4] -= mu * vb[2]; M[32] = mu;

    // Cols 3-14 (k=0,1,2)
    #define DO_BLOCK(K) do { \
        int bp = 26 + 32*(K), bc = 33 + 32*(K), bn = 40 + 32*(K); \
        mu = M[bc] / M[bp]; \
        M[bc+1] -= mu*M[bp+1]; M[bc+2] -= mu*M[bp+2]; M[bc+3] -= mu*M[bp+3]; M[bc+4] -= mu*M[bp+4]; \
        vb[4+4*(K)] -= mu*vb[3+4*(K)]; M[bc] = mu; \
        mu = M[bn] / M[bp]; \
        M[bn+1] -= mu*M[bp+1]; M[bn+2] -= mu*M[bp+2]; M[bn+3] -= mu*M[bp+3]; M[bn+4] -= mu*M[bp+4]; \
        vb[5+4*(K)] -= mu*vb[3+4*(K)]; M[bn] = mu; \
        int r1 = 41+32*(K), r2 = 48+32*(K), pv = 34+32*(K); \
        mu = M[r1] / M[pv]; \
        M[r1+1] -= mu*M[pv+1]; M[r1+2] -= mu*M[pv+2]; M[r1+3] -= mu*M[pv+3]; \
        vb[5+4*(K)] -= mu*vb[4+4*(K)]; M[r1] = mu; \
        mu = M[r2] / M[pv]; \
        M[r2+1] -= mu*M[pv+1]; M[r2+2] -= mu*M[pv+2]; M[r2+3] -= mu*M[pv+3]; \
        vb[6+4*(K)] -= mu*vb[4+4*(K)]; M[r2] = mu; \
        r1 = 49+32*(K); r2 = 56+32*(K); pv = 42+32*(K); \
        mu = M[r1] / M[pv]; \
        M[r1+1] -= mu*M[pv+1]; M[r1+2] -= mu*M[pv+2]; \
        vb[6+4*(K)] -= mu*vb[5+4*(K)]; M[r1] = mu; \
        mu = M[r2] / M[pv]; \
        M[r2+1] -= mu*M[pv+1]; M[r2+2] -= mu*M[pv+2]; \
        vb[7+4*(K)] -= mu*vb[5+4*(K)]; M[r2] = mu; \
        r1 = 57+32*(K); r2 = 64+32*(K); pv = 50+32*(K); \
        mu = M[r1] / M[pv]; \
        M[r1+1] -= mu*M[pv+1]; M[r1+2] -= mu*M[pv+2]; M[r1+3] -= mu*M[pv+3]; \
        M[r1+4] -= mu*M[pv+4]; M[r1+5] -= mu*M[pv+5]; \
        vb[7+4*(K)] -= mu*vb[6+4*(K)]; M[r1] = mu; \
        mu = M[r2] / M[pv]; \
        M[r2+1] -= mu*M[pv+1]; M[r2+2] -= mu*M[pv+2]; M[r2+3] -= mu*M[pv+3]; \
        M[r2+4] -= mu*M[pv+4]; M[r2+5] -= mu*M[pv+5]; \
        vb[8+4*(K)] -= mu*vb[6+4*(K)]; M[r2] = mu; \
    } while(0)
    
    DO_BLOCK(0);
    DO_BLOCK(1);
    DO_BLOCK(2);
    #undef DO_BLOCK

    // Col 15
    mu = M[129] / M[122];
    M[130] -= mu * M[123]; M[131] -= mu * M[124];
    vb[16] -= mu * vb[15]; M[129] = mu;
    mu = M[136] / M[122];
    M[137] -= mu * M[123]; M[138] -= mu * M[124];
    vb[17] -= mu * vb[15]; M[136] = mu;

    // Col 16
    mu = M[137] / M[130];
    M[138] -= mu * M[131];
    vb[17] -= mu * vb[16]; M[137] = mu;

    // =========================================================
    // Part 2: Back Substitution 
    // =========================================================

    double inv17 = 1.0 / M[138];
    double inv16 = 1.0 / M[130];
    double inv15 = 1.0 / M[122];
    double inv14 = 1.0 / M[114];
    double inv13 = 1.0 / M[106];
    double inv12 = 1.0 / M[98];
    double inv11 = 1.0 / M[90];
    double inv10 = 1.0 / M[82];
    double inv9  = 1.0 / M[74];
    double inv8  = 1.0 / M[66];
    double inv7  = 1.0 / M[58];
    double inv6  = 1.0 / M[50];
    double inv5  = 1.0 / M[42];
    double inv4  = 1.0 / M[34];
    double inv3  = 1.0 / M[26];
    double inv2  = 1.0 / M[18];
    double inv1  = 1.0 / M[10];
    double inv0  = 1.0 / M[2];
    
    
    // 完全展开的回代 - 直接用局部变量
    double c17 = vb[17] * inv17;
    double c16 = (vb[16] - M[131] * c17) * inv16;
    double c15 = (vb[15] - M[123] * c16 - M[124] * c17) * inv15;
    double c14 = (vb[14] - M[115] * c15 - M[116] * c16 - M[117] * c17) * inv14;
    double c13 = (vb[13] - M[107] * c14 - M[108] * c15 - M[109] * c16 - M[110] * c17) * inv13;
    double c12 = (vb[12] - M[99]*c13 - M[100]*c14 - M[101]*c15 - M[102]*c16 - M[103]*c17) * inv12;
    double c11 = (vb[11] - M[91]*c12 - M[92]*c13 - M[93]*c14 - M[94]*c15 - M[95]*c16) * inv11;
    double c10 = (vb[10] - M[83]*c11 - M[84]*c12 - M[85]*c13 - M[86]*c14 - M[87]*c15) * inv10;
    double c9  = (vb[9] - M[75]*c10 - M[76]*c11 - M[77]*c12 - M[78]*c13 - M[79]*c14) * inv9;
    double c8  = (vb[8] - M[67]*c9 - M[68]*c10 - M[69]*c11 - M[70]*c12 - M[71]*c13) * inv8;
    double c7  = (vb[7] - M[59]*c8 - M[60]*c9 - M[61]*c10 - M[62]*c11 - M[63]*c12) * inv7;
    double c6  = (vb[6] - M[51]*c7 - M[52]*c8 - M[53]*c9 - M[54]*c10 - M[55]*c11) * inv6;
    double c5  = (vb[5] - M[43]*c6 - M[44]*c7 - M[45]*c8 - M[46]*c9 - M[47]*c10) * inv5;
    double c4  = (vb[4] - M[35]*c5 - M[36]*c6 - M[37]*c7 - M[38]*c8 - M[39]*c9) * inv4;
    double c3  = (vb[3] - M[27]*c4 - M[28]*c5 - M[29]*c6 - M[30]*c7 - M[31]*c8) * inv3;
    double c2  = (vb[2] - M[19]*c3 - M[20]*c4 - M[21]*c5 - M[22]*c6 - M[23]*c7) * inv2;
    double c1  = (vb[1] - M[11]*c2 - M[12]*c3 - M[13]*c4 - M[14]*c5 - M[15]*c6) * inv1;
    double c0  = (vb[0] - M[3]*c1 - M[4]*c2 - M[5]*c3 - M[6]*c4 - M[7]*c5) * inv0;
    
    // 写回结果
    Coe[0] = c0;   Coe[1] = c1;   Coe[2] = c2;   Coe[3] = c3;
    Coe[4] = c4;   Coe[5] = c5;   Coe[6] = c6;   Coe[7] = c7;
    Coe[8] = c8;   Coe[9] = c9;   Coe[10] = c10; Coe[11] = c11;
    Coe[12] = c12; Coe[13] = c13; Coe[14] = c14; Coe[15] = c15;
    Coe[16] = c16; Coe[17] = c17;

    Coe[19] = c17;
    Coe[17] = c16; 
    Coe[16] = 0.0;
    Coe[18] = 0.0;
}




inline void compute_dKdE_times_X(CalcBuffer& buffer, int k){
    const double* C = buffer.Coe.data();            // Coe
    const double* D = buffer.DCoefficientDE.data(); // DCoefficientDE
    double* Y_ptr = buffer.Y.data();                // Y
    buffer.Y.setZero();
    switch (k) {
    case 1:
        Y_ptr[4] = - (C[5] * D[0] + C[6] * D[1] + C[7] * D[2]);
        Y_ptr[5] = - (C[4] * D[3] + C[6] * D[4] + C[7] * D[5]);
        break;

    case 2:
        Y_ptr[4] = - (C[5] * D[6] + C[6] * D[7] + C[7] * D[8]);
        Y_ptr[5] = - (C[4] * D[9] + C[6] * D[10] + C[7] * D[11]);
        Y_ptr[8] = - (C[9] * D[12] + C[10] * D[13] + C[11] * D[14]);
        Y_ptr[9] = - (C[8] * D[15] + C[10] * D[16] + C[11] * D[17]);
        break;

    case 3:
        Y_ptr[8]  = - (C[9] * D[18] + C[10] * D[19] + C[11] * D[20]);
        Y_ptr[9]  = - (C[8] * D[21] + C[10] * D[22] + C[11] * D[23]);
        Y_ptr[12] = - (C[13] * D[24] + C[14] * D[25] + C[15] * D[26]);
        Y_ptr[13] = - (C[12] * D[27] + C[14] * D[28] + C[15] * D[29]);
        break;

    case 4:
        Y_ptr[12] = - (C[13] * D[30] + C[14] * D[31] + C[15] * D[32]);
        Y_ptr[13] = - (C[12] * D[33] + C[14] * D[34] + C[15] * D[35]);
        Y_ptr[16] = - (C[17] * D[36] + C[19] * D[37]);
        Y_ptr[17] = - (C[19] * D[38]);
        break;

    case 5:
        Y_ptr[16] = - (C[17] * D[39] + C[19] * D[40]);
        Y_ptr[17] = - (C[19] * D[41]);
        break;
    }     
}

inline void Solve_dXdE(CalcBuffer& buffer) {
    double* __restrict__ Y = buffer.Y.data(); 
    const double* __restrict__ M = buffer.Coe_Matrix.data(); 
    double* __restrict__ Z = buffer.Z.data(); 
    
    // =========================================================
    // Part 1: Forward Substitution (Ly = b)
    // =========================================================
    
    // --- Swap Row 0 and Row 1 (Partial Pivoting) ---
    double y0 = Y[1];
    double y1 = Y[0];
    double y2 = Y[2];
    double y3 = Y[3];
    double y4 = Y[4];
    double y5 = Y[5];
    double y6 = Y[6];
    double y7 = Y[7];
    double y8 = Y[8];
    double y9 = Y[9];
    double y10 = Y[10];
    double y11 = Y[11];
    double y12 = Y[12];
    double y13 = Y[13];
    double y14 = Y[14];
    double y15 = Y[15];
    double y16 = Y[16];
    double y17 = Y[17];
    
    // --- Col 0 ---
    y1 -= M[9] * y0;
    y2 -= M[16] * y0;

    // --- Col 1 ---
    y2 -= M[17] * y1;
    y3 -= M[24] * y1;

    // --- Col 2 ---
    y3 -= M[25] * y2;
    y4 -= M[32] * y2;

    // --- Cols 3-14: Block k=0 ---
    y4 -= M[33] * y3;
    y5 -= M[40] * y3;
    y5 -= M[41] * y4;
    y6 -= M[48] * y4;
    y6 -= M[49] * y5;
    y7 -= M[56] * y5;
    y7 -= M[57] * y6;
    y8 -= M[64] * y6;

    // --- Cols 3-14: Block k=1 ---
    y8  -= M[65] * y7;
    y9  -= M[72] * y7;
    y9  -= M[73] * y8;
    y10 -= M[80] * y8;
    y10 -= M[81] * y9;
    y11 -= M[88] * y9;
    y11 -= M[89] * y10;
    y12 -= M[96] * y10;

    // --- Cols 3-14: Block k=2 ---
    y12 -= M[97] * y11;
    y13 -= M[104] * y11;
    y13 -= M[105] * y12;
    y14 -= M[112] * y12;
    y14 -= M[113] * y13;
    y15 -= M[120] * y13;
    y15 -= M[121] * y14;
    y16 -= M[128] * y14;

    // --- Col 15 ---
    y16 -= M[129] * y15;
    y17 -= M[136] * y15;

    // --- Col 16 ---
    y17 -= M[137] * y16;

    // =========================================================
    // Part 2: Backward Substitution (Uz = y)
    // =========================================================
    
    // 预计算倒数 - 避免重复除法
    double inv17 = 1.0 / M[138];
    double inv16 = 1.0 / M[130];
    double inv15 = 1.0 / M[122];
    double inv14 = 1.0 / M[114];
    double inv13 = 1.0 / M[106];
    double inv12 = 1.0 / M[98];
    double inv11 = 1.0 / M[90];
    double inv10 = 1.0 / M[82];
    double inv9  = 1.0 / M[74];
    double inv8  = 1.0 / M[66];
    double inv7  = 1.0 / M[58];
    double inv6  = 1.0 / M[50];
    double inv5  = 1.0 / M[42];
    double inv4  = 1.0 / M[34];
    double inv3  = 1.0 / M[26];
    double inv2  = 1.0 / M[18];
    double inv1  = 1.0 / M[10];
    double inv0  = 1.0 / M[2];

    // 完全展开的回代 - 全部使用局部变量，最大化寄存器利用
    double z17 = y17 * inv17;
    double z16 = (y16 - M[131] * z17) * inv16;
    double z15 = (y15 - M[123] * z16 - M[124] * z17) * inv15;
    double z14 = (y14 - M[115] * z15 - M[116] * z16 - M[117] * z17) * inv14;
    double z13 = (y13 - M[107] * z14 - M[108] * z15 - M[109] * z16 - M[110] * z17) * inv13;
    double z12 = (y12 - M[99]*z13 - M[100]*z14 - M[101]*z15 - M[102]*z16 - M[103]*z17) * inv12;
    double z11 = (y11 - M[91]*z12 - M[92]*z13 - M[93]*z14 - M[94]*z15 - M[95]*z16) * inv11;
    double z10 = (y10 - M[83]*z11 - M[84]*z12 - M[85]*z13 - M[86]*z14 - M[87]*z15) * inv10;
    double z9  = (y9 - M[75]*z10 - M[76]*z11 - M[77]*z12 - M[78]*z13 - M[79]*z14) * inv9;
    double z8  = (y8 - M[67]*z9 - M[68]*z10 - M[69]*z11 - M[70]*z12 - M[71]*z13) * inv8;
    double z7  = (y7 - M[59]*z8 - M[60]*z9 - M[61]*z10 - M[62]*z11 - M[63]*z12) * inv7;
    double z6  = (y6 - M[51]*z7 - M[52]*z8 - M[53]*z9 - M[54]*z10 - M[55]*z11) * inv6;
    double z5  = (y5 - M[43]*z6 - M[44]*z7 - M[45]*z8 - M[46]*z9 - M[47]*z10) * inv5;
    double z4  = (y4 - M[35]*z5 - M[36]*z6 - M[37]*z7 - M[38]*z8 - M[39]*z9) * inv4;
    double z3  = (y3 - M[27]*z4 - M[28]*z5 - M[29]*z6 - M[30]*z7 - M[31]*z8) * inv3;
    double z2  = (y2 - M[19]*z3 - M[20]*z4 - M[21]*z5 - M[22]*z6 - M[23]*z7) * inv2;
    double z1  = (y1 - M[11]*z2 - M[12]*z3 - M[13]*z4 - M[14]*z5 - M[15]*z6) * inv1;
    double z0  = (y0 - M[3]*z1 - M[4]*z2 - M[5]*z3 - M[6]*z4 - M[7]*z5) * inv0;

    // 写回结果 - 一次性写入
    Z[0] = z0;   Z[1] = z1;   Z[2] = z2;   Z[3] = z3;
    Z[4] = z4;   Z[5] = z5;   Z[6] = z6;   Z[7] = z7;
    Z[8] = z8;   Z[9] = z9;   Z[10] = z10; Z[11] = z11;
    Z[12] = z12; Z[13] = z13; Z[14] = z14; Z[15] = z15;
    Z[16] = 0.0; Z[17] = z16; Z[18] = 0.0; Z[19] = z17;
}


inline void Integrand_IA(CalcBuffer& buffer, const ModelParams& params, bool& p_initialized){
    const double* M = buffer.Coe_Matrix.data(); 
    double* x = buffer.Coe.data();
    if (!p_initialized) {
        double cache_D = M[5] * M[10] - M[3] * M[12];

        x[1] = -M[12] / cache_D;
        x[3] =  M[10] / cache_D;

        x[0] = 0.0;
        x[2] = 0.0;

        for (int i = 4; i < 20; ++i) {
            x[i] = 0.0;
        }

        p_initialized = true;
    }
}