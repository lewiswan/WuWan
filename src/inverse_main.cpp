#ifndef EIGEN_VECTORIZE
#define EIGEN_VECTORIZE  
#endif

#define EIGEN_DONT_ALIGN_STATICALLY 0 
// ----------------- 1. Header file references -----------------
#include <ceres/ceres.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <pybind11/eigen.h> 
#include <Eigen/Dense>
#include "structures.h"
#include "processing_function.h"
#include "math_fun.h"
#include "bessel_table.h"
#include "forward.h"
#include <chrono>
#include <pybind11/stl.h>

namespace py = pybind11;

class PavementCostFunction : public ceres::SizedCostFunction<10, 5>{
public:
    PavementCostFunction(const Eigen::Ref<InputMat> input_arr, Eigen::Ref<Vec10> obs)
        : input_(input_arr), obs_(obs) {
        parse_input_data(input_, params_);
        params_.index = index_search(params_.z, params_.evaluation(1));
        
        
        
    }
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override {
        
        double current_E[5]; 
        ModelParams params = params_;
        current_E[0] = std::exp(parameters[0][0]);
        current_E[1] = std::exp(parameters[0][1]);
        current_E[2] = std::exp(parameters[0][2]);
        current_E[3] = std::exp(parameters[0][3]);
        current_E[4] = std::exp(parameters[0][4]);
        params.E(1) = current_E[0];     
        params.E(2) = current_E[1];
        params.E(3) = current_E[2];
        params.E(4) = current_E[3];
        params.E(5) = current_E[4];   
        auto numerator = params.E.head(5).array() * (1.0 + params.nu.tail(5).array());
        auto denominator = params.E.tail(5).array() * (1.0 + params.nu.head(5).array());
        params.F2 = (numerator / denominator).matrix();      

        RunSimulationKernel(params, buffer_, results_, input_);

        for(int i=0; i<10; ++i) {
            residuals[i] = results_.result_displacement(i) - obs_(i);
        }

        if (jacobians != NULL && jacobians[0] != NULL) {
            for(int r = 0; r < 10; ++r) { 
                jacobians[0][r * 5 + 0] = results_.J_E(r, 0) * current_E[0]; 
                jacobians[0][r * 5 + 1] = results_.J_E(r, 1) * current_E[1];
                jacobians[0][r * 5 + 2] = results_.J_E(r, 2) * current_E[2];
                jacobians[0][r * 5 + 3] = results_.J_E(r, 3) * current_E[3];
                jacobians[0][r * 5 + 4] = results_.J_E(r, 4) * current_E[4];
            }
        }
        return true;
    }
private:
    const InputMat input_;
    const Vec10 obs_;
    ModelParams params_;
    
    mutable CalcBuffer buffer_;
    mutable SimResults results_;
};


// ----------------- Main Function -----------------
std::vector<double> BackCalculation(Eigen::Ref<InputMat> input_arr, Eigen::Ref<Vec10> obs, Eigen::Ref<Vec10> phys_bounds) {
    
    ModelParams params;
    CalcBuffer buffer;
    SimResults results;
    ceres::Problem problem;
    std::vector<double> x_params(5);

    for (int i = 0; i < 5; ++i){
        double lower_E = phys_bounds(2 * i);
        double upper_E = phys_bounds(2 * i + 1);
        double mid_E = (lower_E + upper_E) * 0.5;
        
        x_params[i] = std::log(mid_E); 
    }
    problem.AddResidualBlock(
        new PavementCostFunction(input_arr, obs),
        nullptr, 
        x_params.data()
    );
    for (int i = 0; i < 5; ++i) {
        // E_min <= E <= E_max  --->  ln(E_min) <= x <= ln(E_max)
        problem.SetParameterLowerBound(x_params.data(), i, std::log(phys_bounds(2 * i)));     
        problem.SetParameterUpperBound(x_params.data(), i, std::log(phys_bounds(2 * i + 1))); 
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.trust_region_strategy_type = ceres::DOGLEG; // similar to dogbox
    options.minimizer_progress_to_stdout = false; 
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-8;
    options.num_threads = 1; 

    auto start = std::chrono::high_resolution_clock::now(); 

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // std::vector<double> final_E_values(5);
    for(int i=0; i<5; ++i) {
        final_E_values[i] = std::exp(x_params[i]);
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Function took " << duration.count() << " milliseconds to execute." << std::endl;
    
    return final_E_values;

}


PYBIND11_MODULE(WuWan_pavement_inverse, m) {
    m.doc() = "5 - layer deflections inverse model"; 
    m.def("BackCalculation", &BackCalculation, "Main calculation function returning deflection array");
}

