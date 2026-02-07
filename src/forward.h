// forward.h
#pragma once
#include <Eigen/Core>
#include <vector>
#include "structures.h"
#include "processing_function.h"
#include "math_fun.h"
#include "bessel_table.h"

inline void RunSimulationKernel(ModelParams& params, CalcBuffer& buffer, SimResults& results, const Eigen::Ref<const Eigen::MatrixXd>& input_arr, bool calc_grad) {
    Variable_Assignment(params);

    for (int i = 0; i < 10; ++i) {
        params.evaluation(0) = input_arr(i + 1, 6);
        bessel_zeros(buffer, params.evaluation(0) / params.H, params.a / params.H);
        gaussian_quadrature_integrate(params, buffer, results, calc_grad);
        results.result_displacement(i) = results.deflection;
        results.J_E.row(i) = results.gradient_E.transpose(); 
    }

}