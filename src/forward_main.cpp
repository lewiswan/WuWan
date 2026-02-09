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
#include <chrono>

namespace py = pybind11;


// ----------------- Main Function -----------------
SimResults Calculation(Eigen::Ref<InputMat> input_arr, bool calc_grad) {
    
    ModelParams params;
    CalcBuffer buffer;
    SimResults results;

    parse_input_data(input_arr, params);
    params.index = index_search(params.z, params.evaluation(1));

    RunSimulationKernel(params, buffer, results, input_arr, calc_grad);
    return results;
}


PYBIND11_MODULE(WuWan_pavement_forward, m) {
    py::class_<SimResults>(m, "SimResults")
        .def_readonly("result_displacement", &SimResults::result_displacement)
        .def_readonly("J_E", &SimResults::J_E);
    m.doc() = "5 - layer deflections forward model"; 
    m.def("Calculation", &Calculation, "Main calculation function returning deflection array", py::arg("input_arr"), py::arg("calc_grad"));
}


