// src/structures.h
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <array>

// ----------------- 1. Type Definition  -----------------
// Define some convenient aliases
using Vec2 = Eigen::Matrix<double, 2, 1>;
using Vec4 = Eigen::Matrix<double, 4, 1>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec5 = Eigen::Matrix<double, 5, 1>;
using Vec5_bool = Eigen::Array<bool, 5, 1>;
using Vec10 = Eigen::Matrix<double, 10, 1>; 
using Vec18 = Eigen::Matrix<double, 18, 1>;
using Vec20 = Eigen::Matrix<double, 20, 1>;
using Vec42 = Eigen::Matrix<double, 42, 1>;
using Vec64 = Eigen::Matrix<double, 64, 1>;
using Vec121 = Eigen::Matrix<double, 121, 1>;
using Vec128 = Eigen::Matrix<double, 128, 1>;
using BandMat18x8 = Eigen::Matrix<double, 18, 8, Eigen::RowMajor>;
using Mat5x5 = Eigen::Matrix<double, 5, 5, Eigen::RowMajor>;
using Mat10x5 = Eigen::Matrix<double, 10, 5, Eigen::RowMajor>;
using InputMat = Eigen::Matrix<double, 11, 8, Eigen::RowMajor>;
using InputMat2 = Eigen::Matrix<double, 11, 7, Eigen::RowMajor>;

// ----------------- 2. Structure Definition  -----------------
// const ModelParams& params : Read-only
// CalcBuffer& buffer        : Scratchpad
// SimResults& results       : Output
struct ModelParams {
    // --- 1. model fixed parameters ---
    Vec2 evaluation; // evaluation[0] is r, evaluation[1] is z
    Vec6 z, nu, E;
    double q, a, H;
    int index;
    Vec5 F2;
    // --- 2. The Template of the bandwidth matrix ---
    BandMat18x8 Coe_Matrix_Template;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // Memory alignment support
};

struct CalcBuffer {
    BandMat18x8 Coe_Matrix; 
    Vec121 zeros;
    Vec42 DCoefficientDE;
    Vec128 points;
    Vec128 weights;
    Vec5 F1, F3;
    Vec18 b, Y;
    Vec20 Coe, Z;
    Vec5 dA, dB, dC, dD, dsum;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Memory alignment support
};

struct SimResults {
    double deflection; 
    Vec5 gradient_E;
    Vec10 result_displacement; 
    Mat10x5 J_E; 
    Vec2 count_number; // [method_52_count, method_IA_count]
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct BackCalcParams {
    Vec10 deflections, r, error_deflection_mm, error_r_mm;
    Vec4 thicknesses, error_thickness_mm;
    double load, error_load;
    Vec5 lower_bounds_phys, upper_bounds_phys;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct BackCalcBuffer {
    Vec10 deflections_with_noise, r_with_noise;
    Vec4 thicknesses_with_noise;
    double load_with_noise;
    Vec5 x_prior_log, x0_phys, x0_log;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct BackCalcResult {
    Vec5 final_moduli;
    double elapsed_seconds;
    int call_count;
    int grad_count;
    int iterations;
    double final_cost;
    int status;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW                        
};

// ----------------- 3. Declare the parsing function  -----------------
//double BJ0(double x);
//double BJ1(double x);
//void parse_input_data(const Eigen::Ref<const InputMat>& input_arr, ModelParams& params);
//int index_search(const Vec6& z, double target);
//void Variable_Assignment(ModelParams& params);
//void bessel_zeros(CalcBuffer& buffer, double r, double a);
void gaussian_quadrature_integrate(ModelParams& params, CalcBuffer& buffer, SimResults& results, bool calc_grad);
//void compute_gauss_fixed_n(double a, double b, Vec64& points, Vec64& weights);
//void gauss_legendre_point(int n, double a, double b, Vec64& points, Vec64& weights);
//void Coefficient_52(CalcBuffer& buffer, const ModelParams& params, int j);
//void Derivative_E(CalcBuffer& buffer, const ModelParams& params);
////void Integrand_52(CalcBuffer& buffer, const ModelParams& params);
//void compute_dKdE_times_X(CalcBuffer& buffer, int k);
//void Solve_dXdE(CalcBuffer& buffer);
//void Coefficient_IA(CalcBuffer& buffer, const ModelParams& params, int j);
//void Integrand_IA(CalcBuffer& buffer, const ModelParams& params, bool& p_initialized);
SimResults Calculation(Eigen::Ref<InputMat> input_arr, bool calc_grad);
BackCalcResult BackCalculation(Eigen::Ref<InputMat> input_arr, Eigen::Ref<InputMat2> input_arr2, int verbose);