// inverse.h
#include <Eigen/Dense>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/NumericalDiff>
#include <cmath>
#include <iostream>
#include <chrono>
#include "structures.h"
#include "forward.h"
#include "inverse_functor.h"
#include "projected_lm.h"

inline void parse_input_data_backcal(const Eigen::Ref<const InputMat>& input_arr, const Eigen::Ref<const InputMat2>& input_arr2, BackCalcParams& back_params, BackCalcBuffer& back_buffer) {
    back_params.lower_bounds_phys = input_arr2.block<5, 1>(2, 1);
    back_params.upper_bounds_phys = input_arr2.block<5, 1>(2, 2);
    back_buffer.x0_phys = input_arr.block<5, 1>(2, 1);
    back_buffer.x_prior_log = (back_params.lower_bounds_phys.array().log() + back_params.upper_bounds_phys.array().log()) * 0.5;   
    auto exp_prior = back_buffer.x_prior_log.array().exp();
    auto is_zero = back_buffer.x0_phys.array() == 0.0;
    back_buffer.x0_phys = is_zero.select(exp_prior, back_buffer.x0_phys.array());
    back_buffer.x0_log = back_buffer.x0_phys.array().log();
    
    back_params.error_deflection_mm = input_arr2.block<10, 1>(1, 6) * 1e-3; // convert um to mm
    back_params.error_r_mm = input_arr2.block<10, 1>(1, 5);
    back_params.error_thickness_mm = input_arr2.block<4, 1>(2, 3);
    back_params.error_load = input_arr2(10, 1);

    back_params.deflections = input_arr.block<10, 1>(1, 7);
    back_params.r = input_arr.block<10, 1>(1, 6);
    back_params.thicknesses = input_arr.block<4, 1>(2, 3);
    back_params.load = input_arr(10, 1);

    back_params.deflections *= 1e-3; // convert um to mm

}

inline void add_err(const Eigen::Ref<const InputMat>& input_arr, Eigen::Ref<InputMat> noise_input, BackCalcParams& back_params, BackCalcBuffer& back_buffer) {
    auto rand_tri = [](Eigen::Index) -> double {
        static thread_local std::mt19937_64 rng(std::random_device{}());
        static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(rng) - dist(rng);  
    };
    // deflection error
    Vec10 noise_def = Vec10::NullaryExpr(rand_tri);
    back_buffer.deflections_with_noise = (noise_def).array() * back_params.error_deflection_mm.array() + back_params.deflections.array();
    noise_input.block<10, 1>(1, 7) = back_buffer.deflections_with_noise;

    // thickness error
    Vec4 noise_thi = Vec4::NullaryExpr(rand_tri);
    back_buffer.thicknesses_with_noise = (noise_thi).array() * back_params.error_thickness_mm.array() + back_params.thicknesses.array();
    noise_input.block<4, 1>(2, 3) = back_buffer.thicknesses_with_noise;

    // r error
    Vec10 noise_r = Vec10::NullaryExpr(rand_tri);
    back_buffer.r_with_noise = (noise_r).array() * back_params.error_r_mm.array() + back_params.r.array();
    noise_input.block<10, 1>(1, 6) = back_buffer.r_with_noise;

    // load error
    double noise_load = rand_tri(0);
    back_buffer.load_with_noise = noise_load * back_params.error_load * back_params.load + back_params.load;
    noise_input(10, 1) = back_buffer.load_with_noise;

}


inline void RunBackCalKernel (const Eigen::Ref<const InputMat>& noise_input, BackCalcResult& back_result, BackCalcBuffer& back_buffer, BackCalcParams& back_params, const int verbose, const int accuracy) {
    Vec5& lb_phys = back_params.lower_bounds_phys;
    Vec5& ub_phys = back_params.upper_bounds_phys;
    Vec5 lb_log = lb_phys.array().log();
    Vec5 ub_log = ub_phys.array().log();
    Vec5& x0_phys = back_buffer.x0_phys;
    Vec5& x0_log = back_buffer.x0_log;

    BackCalcFunctor functor;
    functor.n_params = 5;
    functor.n_residuals = 10;
    functor.u_target = back_buffer.deflections_with_noise;
    functor.arr_data = noise_input;

    ProjectedLMOptions opts;
    if (accuracy == 0){
        opts.max_iter = 200;
        opts.max_fev  = 400;
        opts.ftol     = 1e-6;
        opts.xtol     = 1e-6;
        opts.gtol     = 1e-9;
        opts.verbose  = verbose;
        opts.min_iter = 1;              
        opts.cost_threshold = 1e-24;
    } else {
        opts.max_iter = 100;
        opts.max_fev  = 200;
        opts.ftol     = 1e-6;
        opts.xtol     = 1e-6;
        opts.gtol     = 1e-6;
        opts.verbose  = verbose;
        opts.min_iter = 10;              
        opts.cost_threshold = 1e-16;
    }
    

    auto start = std::chrono::high_resolution_clock::now();

    ProjectedLMResult lm_result = projected_lm_solve(functor, x0_log, lb_log, ub_log, opts);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    Vec5 final_phys;
    final_phys = lm_result.x.array().exp();

    back_result.final_moduli    = final_phys;
    back_result.elapsed_seconds = elapsed;
    back_result.call_count      = functor.call_count;
    back_result.grad_count      = functor.grad_count;
    back_result.iterations      = lm_result.iterations;
    back_result.final_cost      = lm_result.final_cost;
    back_result.status          = static_cast<int>(lm_result.status);

}