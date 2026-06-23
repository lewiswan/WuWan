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
#include "structures.h"
#include "processing_function.h"
#include "math_fun.h"
#include "bessel_table.h"
#include "forward.h"
#include "inverse.h"
#include <chrono>
#include <pybind11/stl.h>

namespace py = pybind11;


// ----------------- Main Function -----------------
BackCalcResult BackCalculation(Eigen::Ref<InputMat> input_arr, Eigen::Ref<InputMat2> input_arr2, int verbose) {
    BackCalcParams back_params;
    BackCalcBuffer back_buffer;
    BackCalcResult back_result;

    parse_input_data_backcal(input_arr, input_arr2, back_params, back_buffer);
    // std::cout << back_buffer.x0_phys.transpose() << std::endl;
    InputMat noise_input = input_arr;
    add_err(input_arr, noise_input, back_params, back_buffer);
    int accuracy = input_arr2(10, 2);
    RunBackCalKernel(noise_input, back_result, back_buffer, back_params, verbose, accuracy);
    
    return back_result;

}


PYBIND11_MODULE(WuWan_pavement_inverse, m) {
    py::class_<BackCalcResult>(m, "BackCalcResult")
        .def(py::init<>()) 
        .def_readwrite("final_moduli", &BackCalcResult::final_moduli, "Optimized moduli values (Vec5)")
        .def_readwrite("elapsed_seconds", &BackCalcResult::elapsed_seconds, "Total time taken in seconds")
        .def_readwrite("call_count", &BackCalcResult::call_count, "Number of function evaluations")
        .def_readwrite("grad_count", &BackCalcResult::grad_count, "Number of gradient evaluations")
        .def_readwrite("iterations", &BackCalcResult::iterations, "Number of LM iterations")
        .def_readwrite("final_cost", &BackCalcResult::final_cost, "Final 0.5 * ||r||^2")
        .def_readwrite("status", &BackCalcResult::status, "Convergence status code");
    m.doc() = "5 - layer deflections inverse model"; 
    m.def("BackCalculation", &BackCalculation, "Main calculation function returning deflection array", py::arg("input_arr"), py::arg("input_arr2"), py::arg("verbose") = 0);   
}

