// src/interand_solver.cpp
#ifndef EIGEN_VECTORIZE
#define EIGEN_VECTORIZE  
#endif

#define EIGEN_DONT_ALIGN_STATICALLY 0 
#include <iostream>
#include <cmath>
#include <algorithm>
#include "structures.h" 
#include <boost/math/special_functions/bessel.hpp>
#include "processing_function.h"
#include "math_fun.h"

// ===========================================================================
// Adjoint (transposed) solve of the banded system.
//
// Integrand_52 factors K = P * L * U in place.  The band is stored so that
// entry (r, c) lives at M[8*r + c + 1], i.e. the diagonal is M[8*r + 2], and
// P swaps rows 0 and 1.
//
// The forward sensitivity needs, for each layer k, the scalar
//     s_k = c^T Z_k   with   K Z_k = -(dK/dE_k) X.
// The current code obtains every Z_k with its own triangular solve (five in
// total).  Because c is a single fixed vector, transpose instead:
//     c^T Z_k = (K^-T c)^T (-(dK/dE_k) X) = lambda^T Y_k
// so ONE transposed solve replaces all five.  K^-T c follows from
// K^T = U^T L^T P: solve U^T z = c, then L^T w = z, then lambda = P w.
// ===========================================================================
inline void Solve_Transposed(const double* RESTRICT M, const double* RESTRICT c,
                             double* RESTRICT lambda) {
    double inv[18];
    for (int i = 0; i < 18; ++i) inv[i] = 1.0 / M[8 * i + 2];

    // ---- U^T z = c  (forward substitution, upper band width 5) ----
    const double z0  = c[0] * inv[0];
    const double z1  = (c[1]  - M[3]  * z0) * inv[1];
    const double z2  = (c[2]  - M[4]  * z0 - M[11] * z1) * inv[2];
    const double z3  = (c[3]  - M[5]  * z0 - M[12] * z1 - M[19] * z2) * inv[3];
    const double z4  = (c[4]  - M[6]  * z0 - M[13] * z1 - M[20] * z2 - M[27] * z3) * inv[4];
    const double z5  = (c[5]  - M[7]  * z0 - M[14] * z1 - M[21] * z2 - M[28] * z3 - M[35] * z4) * inv[5];
    const double z6  = (c[6]  - M[15] * z1 - M[22] * z2 - M[29] * z3 - M[36] * z4 - M[43] * z5) * inv[6];
    const double z7  = (c[7]  - M[23] * z2 - M[30] * z3 - M[37] * z4 - M[44] * z5 - M[51] * z6) * inv[7];
    const double z8  = (c[8]  - M[31] * z3 - M[38] * z4 - M[45] * z5 - M[52] * z6 - M[59] * z7) * inv[8];
    const double z9  = (c[9]  - M[39] * z4 - M[46] * z5 - M[53] * z6 - M[60] * z7 - M[67] * z8) * inv[9];
    const double z10 = (c[10] - M[47] * z5 - M[54] * z6 - M[61] * z7 - M[68] * z8 - M[75] * z9) * inv[10];
    const double z11 = (c[11] - M[55] * z6 - M[62] * z7 - M[69] * z8 - M[76] * z9 - M[83] * z10) * inv[11];
    const double z12 = (c[12] - M[63] * z7 - M[70] * z8 - M[77] * z9 - M[84] * z10 - M[91] * z11) * inv[12];
    const double z13 = (c[13] - M[71] * z8 - M[78] * z9 - M[85] * z10 - M[92] * z11 - M[99] * z12) * inv[13];
    const double z14 = (c[14] - M[79] * z9 - M[86] * z10 - M[93] * z11 - M[100] * z12 - M[107] * z13) * inv[14];
    const double z15 = (c[15] - M[87] * z10 - M[94] * z11 - M[101] * z12 - M[108] * z13 - M[115] * z14) * inv[15];
    const double z16 = (c[16] - M[95] * z11 - M[102] * z12 - M[109] * z13 - M[116] * z14 - M[123] * z15) * inv[16];
    const double z17 = (c[17] - M[103] * z12 - M[110] * z13 - M[117] * z14 - M[124] * z15 - M[131] * z16) * inv[17];

    // ---- L^T w = z  (back substitution, unit diagonal) ----
    // L(r, r-1) is at 8r+9 and L(r, r-2) at 8r+16.
    const double w17 = z17;
    const double w16 = z16 - M[137] * w17;
    const double w15 = z15 - M[129] * w16 - M[136] * w17;
    const double w14 = z14 - M[121] * w15 - M[128] * w16;
    const double w13 = z13 - M[113] * w14 - M[120] * w15;
    const double w12 = z12 - M[105] * w13 - M[112] * w14;
    const double w11 = z11 - M[97]  * w12 - M[104] * w13;
    const double w10 = z10 - M[89]  * w11 - M[96]  * w12;
    const double w9  = z9  - M[81]  * w10 - M[88]  * w11;
    const double w8  = z8  - M[73]  * w9  - M[80]  * w10;
    const double w7  = z7  - M[65]  * w8  - M[72]  * w9;
    const double w6  = z6  - M[57]  * w7  - M[64]  * w8;
    const double w5  = z5  - M[49]  * w6  - M[56]  * w7;
    const double w4  = z4  - M[41]  * w5  - M[48]  * w6;
    const double w3  = z3  - M[33]  * w4  - M[40]  * w5;
    const double w2  = z2  - M[25]  * w3  - M[32]  * w4;
    const double w1  = z1  - M[17]  * w2  - M[24]  * w3;
    const double w0  = z0  - M[9]   * w1  - M[16]  * w2;

    // lambda = P w
    lambda[0] = w1;   lambda[1] = w0;
    lambda[2] = w2;   lambda[3] = w3;   lambda[4] = w4;   lambda[5] = w5;
    lambda[6] = w6;   lambda[7] = w7;   lambda[8] = w8;   lambda[9] = w9;
    lambda[10] = w10; lambda[11] = w11; lambda[12] = w12; lambda[13] = w13;
    lambda[14] = w14; lambda[15] = w15; lambda[16] = w16; lambda[17] = w17;
}

// lambda^T Y_k for Y_k = -(dK/dE_k) X, exactly the sparsity pattern that
// compute_dKdE_times_X fills.  `lam` is indexed like Coe (20 slots): the
// solved 18-vector is expanded as Coe[16]=0, Coe[17]=c16, Coe[18]=0,
// Coe[19]=c17 -- the same convention Solve_dXdE uses for Z.
inline double Adjoint_Dot(const double* RESTRICT lam, const double* RESTRICT C,
                          const double* RESTRICT D, int k) {
    switch (k) {
    case 1:
        return lam[4] * (C[5] * D[0] + C[6] * D[1] + C[7] * D[2])
             + lam[5] * (C[4] * D[3] + C[6] * D[4] + C[7] * D[5]);
    case 2:
        return lam[4] * (C[5] * D[6] + C[6] * D[7] + C[7] * D[8])
             + lam[5] * (C[4] * D[9] + C[6] * D[10] + C[7] * D[11])
             + lam[8] * (C[9] * D[12] + C[10] * D[13] + C[11] * D[14])
             + lam[9] * (C[8] * D[15] + C[10] * D[16] + C[11] * D[17]);
    case 3:
        return lam[8]  * (C[9] * D[18] + C[10] * D[19] + C[11] * D[20])
             + lam[9]  * (C[8] * D[21] + C[10] * D[22] + C[11] * D[23])
             + lam[12] * (C[13] * D[24] + C[14] * D[25] + C[15] * D[26])
             + lam[13] * (C[12] * D[27] + C[14] * D[28] + C[15] * D[29]);
    case 4:
        return lam[12] * (C[13] * D[30] + C[14] * D[31] + C[15] * D[32])
             + lam[13] * (C[12] * D[33] + C[14] * D[34] + C[15] * D[35])
             + lam[16] * (C[17] * D[36] + C[19] * D[37])
             + lam[17] * (C[19] * D[38]);
    case 5:
        return lam[16] * (C[17] * D[39] + C[19] * D[40])
             + lam[17] * (C[19] * D[41]);
    }
    return 0.0;
}

void gaussian_quadrature_integrate(ModelParams& params, CalcBuffer& buffer, SimResults& results, bool calc_grad){
    double total_integral = 0;
    double common_term_1, e_Index;
    common_term_1 = 2 - 4 * params.nu(params.index);
    double A, B, C, D;
    double bj0_val, bj1_val;
    double common_factor, factor_common_2;
    double integrand;
    int method_52_count = 0;
    int method_IA_count = 0;
    Vec2 exp_term = Vec2::Zero();
    
    Vec5 total_integral_gradient = Vec5::Zero();
    Vec5 integrand_gradient = Vec5::Zero();
    Vec5 Total_integral_gradient = Vec5::Zero();
    Vec5 interval_integral_gradient = Vec5::Zero();
    Vec5 final_gradient = Vec5::Zero();
    buffer.points.setZero();
    buffer.weights.setZero();
    int n_points_number = static_cast<int>(std::sqrt(buffer.zeros[1])); // pre-compute
    long eva = static_cast<long>(2.0 * (3.0 + std::floor(3.0 / ((1.0 + params.evaluation(0) / params.a)))));
    int base_idx = params.index * 4 - 4;
    int MAX_CAPACITY = 128;
    bool p_initialized = false;
    double interval_integral = 0.0;

    // ---- loop-invariant terms, hoisted out of the node loop ---------------
    const double inv_H = 1.0 / params.H;
    const double h_exp_1 = (params.z[0] - params.z[1]) * inv_H;
    const double z_eval = params.evaluation(1);
    const double et0_offset = (z_eval - params.z[params.index]) * inv_H;
    const double et1_offset = (params.z[params.index - 1] - z_eval) * inv_H;
    const double ct_scale   = z_eval * inv_H;
    // FWD sensors sit on the surface (z = 0), which pins index_search to 1, so
    // the two observation exponentials collapse onto values already at hand:
    //   exp(m (0 - z_1)/H) == exp(m (z_0 - z_1)/H)  (the e_Index test value)
    //   exp(-m (0 - z_0)/H) == 1, and common_term_2 == 0.
    const bool surface_sensor = (z_eval == 0.0) && (params.index == 1);
    double cvec[18];
    double lam[20];

    for (int i = 0; i < 119; ++i){
        int calculated_n = static_cast<int>(std::ceil(n_points_number / (4.0 * i + 1.0) + 2.0 / std::sqrt(i + 1.0))) * 4;
        int n_points = std::min(calculated_n, MAX_CAPACITY);
        interval_integral = 0.0; 
        if (calc_grad) {
            interval_integral_gradient.setZero();
        }

        gauss_legendre_point(n_points, buffer.zeros[i], buffer.zeros[i + 1], buffer.points, buffer.weights);
        for (int j = 0; j < n_points; ++j){
            const double m = buffer.points(j);
            buffer.Coe_Matrix = params.Coe_Matrix_Template;
            e_Index = std::exp(m * h_exp_1);

            double et0, et1, common_term_2;
            if (surface_sensor) {
                et0 = e_Index;
                et1 = 1.0;
                common_term_2 = 0.0;
            } else {
                et0 = std::exp(m * et0_offset);
                et1 = std::exp(m * et1_offset);
                common_term_2 = m * ct_scale;
            }
            exp_term(0) = et0;
            exp_term(1) = et1;

            if (e_Index > 0.05){
                Coefficient_52(buffer, params, j);
                method_52_count++;
                Integrand_52(buffer, params); 

                A =   buffer.Coe(base_idx)     * et0;
                B = - buffer.Coe(base_idx + 1) * et1;
                C = - buffer.Coe(base_idx + 2) * (common_term_1 - common_term_2) * et0;
                D = - buffer.Coe(base_idx + 3) * (common_term_1 + common_term_2) * et1;

                if (calc_grad) {
                    Derivative_E(buffer, params); 
                    // RHS of the adjoint system in the solver's 18-unknown
                    // ordering.  The packed Coe layout used below puts the
                    // 17th unknown at Coe[17] and the 18th at Coe[19] (with
                    // Coe[16] and Coe[18] structurally zero), which is exactly
                    // how Solve_dXdE names its Z slots, so a straight
                    // 0..17 -> 0..17 mapping is all that is needed.
                    for (int q = 0; q < 18; ++q) cvec[q] = 0.0;
                    cvec[base_idx]     =  et0;
                    cvec[base_idx + 1] = -et1;
                    cvec[base_idx + 2] = -(common_term_1 - common_term_2) * et0;
                    cvec[base_idx + 3] = -(common_term_1 + common_term_2) * et1;
                    Solve_Transposed(buffer.Coe_Matrix.data(), cvec, lam);
                    for (int k = 1; k < 6; ++k){
                        // Adjoint_Dot returns lambda^T A_k where A_k is the
                        // positive bracket in compute_dKdE_times_X; the kernel
                        // there uses Y_k = -A_k, hence the negation.
                        buffer.dsum(k - 1) = -Adjoint_Dot(lam, buffer.Coe.data(),
                                                          buffer.DCoefficientDE.data(), k);
                    }
                }
            } else {
                method_IA_count++;
                Coefficient_IA(buffer, params, j);
                Integrand_IA(buffer, params, p_initialized);
                
                A =   buffer.Coe(base_idx)     * et0;
                B = - buffer.Coe(base_idx + 1) * et1;
                C = - buffer.Coe(base_idx + 2) * (common_term_1 - common_term_2) * et0;
                D = - buffer.Coe(base_idx + 3) * (common_term_1 + common_term_2) * et1;
                
                if (calc_grad) {
                    for (int k = 1; k < 6; ++k){
                        buffer.dsum(k - 1) = 0.0;
                    } 
                }
            }

            bj0_val = BJ0(m * params.evaluation(0) * inv_H);
            bj1_val = BJ1(m * params.a * inv_H);
            common_factor = buffer.weights(j) / m * bj0_val * bj1_val;
            
            integrand = common_factor * (A + B + C + D);
            interval_integral += integrand; 

            if (calc_grad) {
                for (int l = 0; l < 5; ++l){
                    integrand_gradient(l) = common_factor * buffer.dsum(l); 
                }
                interval_integral_gradient += integrand_gradient;
            }
        }

        factor_common_2 = params.q * params.a * (1.0 + params.nu(params.index)) / params.E(params.index);

        if (std::abs(interval_integral) < 0.01 * std::abs(total_integral) && (i % 2 == 1) && (i > eva)) {
            total_integral += interval_integral * 0.5;
            double final_displacement = total_integral; 
            results.deflection = final_displacement * factor_common_2;

            if (calc_grad) {
                total_integral_gradient += interval_integral_gradient * 0.5; 
                final_gradient = Total_integral_gradient + total_integral_gradient;
                results.gradient_E(0) = (-final_displacement / params.E(1) + final_gradient(0)) * factor_common_2;
                results.gradient_E(1) = final_gradient(1) * factor_common_2;
                results.gradient_E(2) = final_gradient(2) * factor_common_2;
                results.gradient_E(3) = final_gradient(3) * factor_common_2;
                results.gradient_E(4) = final_gradient(4) * factor_common_2;
            } else {
                 results.gradient_E.setZero();
            }
            return; 
        }

        total_integral += interval_integral;
        results.count_number(0) += method_52_count;
        results.count_number(1) += method_IA_count;
        
        if (calc_grad) {
            total_integral_gradient += interval_integral_gradient;
        }
    }

    factor_common_2 = params.q * params.a * (1.0 + params.nu(params.index)) / params.E(params.index);
    results.deflection = total_integral * factor_common_2;

    if (calc_grad) {
        results.gradient_E(0) = (-total_integral / params.E(1) + total_integral_gradient(0)) * factor_common_2;
        results.gradient_E(1) = total_integral_gradient(1) * factor_common_2;
        results.gradient_E(2) = total_integral_gradient(2) * factor_common_2;
        results.gradient_E(3) = total_integral_gradient(3) * factor_common_2;
        results.gradient_E(4) = total_integral_gradient(4) * factor_common_2;
    } else {
        results.gradient_E.setZero();
    }
    
    return;
}
