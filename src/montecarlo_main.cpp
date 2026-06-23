// montecarlo_main.cpp
#ifndef EIGEN_VECTORIZE
#define EIGEN_VECTORIZE  
#endif

#define EIGEN_DONT_ALIGN_STATICALLY 0 
// ----------------- 1. Header file references -----------------
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <pybind11/eigen.h> 
#include <Eigen/Dense>
#include <omp.h>
#include "structures.h"
#include "processing_function.h"
#include "math_fun.h"
#include "bessel_table.h"
#include "forward.h"
#include "inverse.h"
#include <chrono>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

Eigen::VectorXd CalculateMedians(const Eigen::MatrixXd& mat, int num_rows) {
    int cols = mat.cols();
    Eigen::VectorXd medians(cols);
    
    for (int c = 0; c < cols; ++c) {
        std::vector<double> vals(num_rows);
        for (int r = 0; r < num_rows; ++r) {
            vals[r] = mat(r, c);
        }
        std::sort(vals.begin(), vals.end());
        
        if (num_rows % 2 == 0) {
            medians(c) = (vals[num_rows / 2 - 1] + vals[num_rows / 2]) / 2.0;
        } else {
            medians(c) = vals[num_rows / 2];
        }
    }
    return medians;
}

Eigen::MatrixXd ParalleMonteCarlo(Eigen::Ref<InputMat> input_arr, Eigen::Ref<InputMat2> input_arr2, int num_threads = -1) {
    int n_samples_min = 800; 
    int batch_size = 200; 
    int n_samples_max = 2000;
    int n_cols_result = 5;            
    Eigen::MatrixXd all_results(n_samples_max, n_cols_result);

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    #pragma omp parallel for
    for (int i = 0; i < n_samples_min; ++i) {
        BackCalcResult result = BackCalculation(input_arr, input_arr2, 0);
        all_results.row(i) = result.final_moduli.transpose();
    }

    int current_samples = n_samples_min;
    Eigen::VectorXd prev_medians = CalculateMedians(all_results, current_samples);

    while (current_samples < n_samples_max){
        int next_samples = std::min(current_samples + batch_size, n_samples_max);
        #pragma omp parallel for
        for (int i = current_samples; i < next_samples; ++i) {
            BackCalcResult result = BackCalculation(input_arr, input_arr2, 0);
            all_results.row(i) = result.final_moduli.transpose();
        }
        current_samples = next_samples;
        Eigen::VectorXd current_medians = CalculateMedians(all_results, current_samples);

        bool is_converged = true;
        for (int c = 0; c < n_cols_result; ++c) {
            double diff = std::abs(current_medians(c) - prev_medians(c));
            double denom = std::abs(prev_medians(c)) > 1e-9 ? std::abs(prev_medians(c)) : 1.0; 
            
            if ((diff / denom) > 0.01) { // > 1%
                is_converged = false;
                break; 
            }
        }
        if (is_converged) {
            std::cout << "Converged at " << current_samples << " samples." << std::endl;
            break;
        }
        prev_medians = current_medians;
    }


    return all_results.topRows(current_samples);
}

PYBIND11_MODULE(WuWan_pavement_montecarlo, m) {
    m.doc() = "5 - layer moduli uncertainty monte carlo simulation model"; 
    m.def("ParalleMonteCarlo", &ParalleMonteCarlo, "Moduli uncertainty monte carlo simulation", py::arg("input_arr"), py::arg("input_arr2"), py::arg("num_threads") = -1, py::call_guard<py::gil_scoped_release>());   
}