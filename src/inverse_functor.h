// inverse_functor.h
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "structures.h"
#include "forward.h"

// =====================================================================
// BackCalcFunctor: It operates in log(E) space and directly supports projection methods.
// =====================================================================

struct BackCalcFunctor {
    int n_params;       
    int n_residuals;    
    Vec10 u_target;       
    mutable InputMat arr_data;

    mutable Vec5 cached_x;
    mutable Vec10 cached_u;
    mutable Mat10x5 cached_J_log; 
    mutable bool cache_valid;
    mutable bool cache_has_grad;

    mutable int call_count;
    mutable int grad_count;

    // ============ Core positive computation call (with cache) ============
    void compute_cached(const Vec5& log_x, bool need_grad) const {
        // check cache
        if (cache_valid && cached_x.isApprox(log_x, 1e-15)) {
            if (!need_grad || cache_has_grad) return;
        }

        // log_x -> E
        Vec5 E = log_x.array().exp();

        // updata E to arr
        arr_data.block<5, 1>(2, 1) = E;

        // Forward Calculation
        SimResults results = Calculation(arr_data, need_grad);

        // extract u and J
        cached_u = results.result_displacement;

        if (need_grad) {
            const auto& J_phys = results.J_E;
            // dU/d(logE_j) = dU/dE_j * E_j
            cached_J_log = J_phys * E.asDiagonal();
            cache_has_grad = true;
            grad_count++;
        } else {
            cache_has_grad = false;
        }

        cached_x = log_x;
        cache_valid = true;
        call_count++;
    }


    // ============ cost function: r_i = (u_pred_i - u_target_i) / u_target_i ============
    int operator()(const Vec5& log_x, Vec10& residuals) const {
        compute_cached(log_x, false);
        residuals = (cached_u - u_target).array() / u_target.array();
        return 0;
    }
    // ============ Jac: dr_i / d(logE_j) = (1/u_target_i) * dU_i/d(logE_j) ============
    int df(const Vec5& log_x, Mat10x5& jac) const {
        compute_cached(log_x, true);
        jac = cached_J_log.array().colwise() / u_target.array();
        return 0;
    }
    // compute all in one call (for projection methods)
    int compute_all(const Vec5& log_x, Vec10& residuals, Mat10x5& jac) const {
        compute_cached(log_x, true);
        residuals = (cached_u - u_target).array() / u_target.array();
        jac = cached_J_log.array().colwise() / u_target.array();
        return 0;
    }
};