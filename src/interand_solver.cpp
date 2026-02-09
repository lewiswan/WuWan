// src/interand_solver.cpp
#ifndef EIGEN_VECTORIZE
#define EIGEN_VECTORIZE  
#endif

#define EIGEN_DONT_ALIGN_STATICALLY 0 
#include <iostream>
#include "structures.h" 
#include <boost/math/special_functions/bessel.hpp>
#include <cmath>
#include "processing_function.h"
#include "math_fun.h"

void gaussian_quadrature_integrate(ModelParams& params, CalcBuffer& buffer, SimResults& results, bool calc_grad){
    double total_integral = 0;
    // double Total_integral = 0; // Unused variable
    double common_term_1, common_term_2, e_Index;
    common_term_1 = 2 - 4 * params.nu(params.index);
    double A, B, C, D;
    double bj0_val, bj1_val;
    double common_factor, factor_common_2;
    double integrand;
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
    // double* tpl = params.Coe_Matrix_Template.data(); // Unused variable

    for (int i = 0; i < 119; ++i){
        int calculated_n = static_cast<int>(std::ceil(n_points_number / (4.0 * i + 1.0) + 2.0 / std::sqrt(i + 1.0))) * 4;
        int n_points = std::min(calculated_n, MAX_CAPACITY);
        interval_integral = 0.0; 
        if (calc_grad) {
            interval_integral_gradient.setZero();
        }

        gauss_legendre_point(n_points, buffer.zeros[i], buffer.zeros[i + 1], buffer.points, buffer.weights);
        for (int j = 0; j < n_points; ++j){
            buffer.Coe_Matrix = params.Coe_Matrix_Template;
            e_Index = std::exp(buffer.points(j) * (params.z(0) - params.z(1)) / params.H);
            if (e_Index > 0.05){
                Coefficient_52(buffer, params, j);
                
                Integrand_52(buffer, params); 

                exp_term(0) = std::exp(buffer.points(j) * (params.evaluation(1) / params.H - params.z(params.index) / params.H));  
                exp_term(1) = std::exp(- buffer.points(j) * (params.evaluation(1) / params.H - params.z(params.index - 1) / params.H));
                common_term_2 = buffer.points(j) * params.evaluation(1) / params.H;
                
                A =   buffer.Coe(base_idx)     * exp_term(0);
                B = - buffer.Coe(base_idx + 1) * exp_term(1);
                C = - buffer.Coe(base_idx + 2) * (common_term_1 - common_term_2) * exp_term(0);
                D = - buffer.Coe(base_idx + 3) * (common_term_1 + common_term_2) * exp_term(1);

                if (calc_grad) {
                    Derivative_E(buffer, params); 
                    for (int k = 1; k < 6; ++k){
                        compute_dKdE_times_X(buffer, k); 
                        Solve_dXdE(buffer);            
                        buffer.dA(k - 1) = buffer.Z(base_idx) * exp_term(0);
                        buffer.dB(k - 1) = - buffer.Z(base_idx + 1) * exp_term(1);
                        buffer.dC(k - 1) = - buffer.Z(base_idx + 2) * (common_term_1 - common_term_2) * exp_term(0);
                        buffer.dD(k - 1) = - buffer.Z(base_idx + 3) * (common_term_1 + common_term_2) * exp_term(1);
                        buffer.dsum(k - 1) = buffer.dA(k - 1) + buffer.dB(k - 1) + buffer.dC(k - 1) + buffer.dD(k - 1);
                    } 
                }
            } else {
                Coefficient_IA(buffer, params, j);
                Integrand_IA(buffer, params, p_initialized);
                
                exp_term(0) = std::exp(buffer.points(j) * (params.evaluation(1) / params.H - params.z(params.index) / params.H));  
                exp_term(1) = std::exp(- buffer.points(j) * (params.evaluation(1) / params.H - params.z(params.index - 1) / params.H));
                common_term_2 = buffer.points(j) * params.evaluation(1) / params.H;
                
                A =   buffer.Coe(base_idx)     * exp_term(0);
                B = - buffer.Coe(base_idx + 1) * exp_term(1);
                C = - buffer.Coe(base_idx + 2) * (common_term_1 - common_term_2) * exp_term(0);
                D = - buffer.Coe(base_idx + 3) * (common_term_1 + common_term_2) * exp_term(1);
                
                if (calc_grad) {
                    for (int k = 1; k < 6; ++k){
                        buffer.dsum(k - 1) = 0.0;
                    } 
                }
            }

            bj0_val = BJ0(buffer.points(j) * params.evaluation(0) / params.H);
            bj1_val = BJ1(buffer.points(j) * params.a / params.H);
            common_factor = buffer.weights(j) / buffer.points(j) * bj0_val * bj1_val;
            
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