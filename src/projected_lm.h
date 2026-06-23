// projected_lm.h
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include "structures.h"

// ======================================================================
// Projection Levenberg-Marquardt Solver
// Reference: Core idea of ​​scipy.optimize.least_squares method='trf'
// Supports simple box constraints: lb <= x <= ub
// ========================================================================

struct ProjectedLMOptions {
    int max_iter = 100;         // max iter number
    int max_fev = 200;          // max feval number
    double ftol = 1e-6;         // function value convergence threshold
    double xtol = 1e-6;         // parameter convergence threshold
    double gtol = 1e-6;         // gradient convergence threshold
    double tau = 1e-3;          // initial damping factor
    int verbose = 2;            // 0=quiet, 1=summary, 2=per-iteration
    int min_iter = 10;              
    double cost_threshold = 1e-12;

};

enum class ProjectedLMStatus {
    Running = 0,
    ConvergedFtol = 1,
    ConvergedXtol = 2,
    ConvergedGtol = 3,
    MaxIterReached = 4,
    MaxFevReached = 5,
    SmallStep = 6,
    NumericalError = -1
};

struct ProjectedLMResult {
    Vec5 x;              
    double final_cost;               // 0.5 * ||r||^2
    int iterations;
    int func_evals;
    int grad_evals;
    ProjectedLMStatus status;
    std::string message;
};

// put x in the box defined by lb and ub
inline Vec5 project(const Vec5& x, const Vec5& lb, const Vec5& ub) {
    return x.cwiseMax(lb).cwiseMin(ub);
}

// compute the infinity norm of the projected gradient, defined as pg_i = x_i - project(x_i - g_i, lb_i, ub_i)
inline double projected_gradient_inf_norm(const Vec5& x, const Vec5& grad, const Vec5& lb, const Vec5& ub) {
    return (x - (x - grad).cwiseMax(lb).cwiseMin(ub)).cwiseAbs().maxCoeff();
}

// =====================================================================
// Diagonal scaling matrix (TRF style)
// Reduce the step size near the boundary to prevent over-boundary jumps.
// D_ii = 1 / sqrt(v_i), where v_i reflects the distance to the boundary.
// =====================================================================
inline Vec5 compute_scaling(const Vec5& x, const Vec5& grad, const Vec5& lb, const Vec5& ub){
    Vec5 dist_lb = (x - lb).array();
    Vec5 dist_ub = (ub - x).array();
    // grad < 0 
    auto mask_neg = (grad.array() < 0);
    // grad > 0 
    auto mask_pos = (grad.array() > 0);
    Vec5 dv = mask_neg.select(dist_ub, mask_pos.select(dist_lb, dist_lb.cwiseMin(dist_ub)));
    return dv.cwiseMax(1e-20);
}

template<typename Functor>
ProjectedLMResult projected_lm_solve(Functor& functor, const Vec5& x0, const Vec5& lb, const Vec5& ub, const ProjectedLMOptions& opts = ProjectedLMOptions()){
    const int n = 5;                  
    const int m = 10;  

    ProjectedLMResult result;
    result.iterations = 0;
    result.func_evals = 0;
    result.grad_evals = 0;
    result.status = ProjectedLMStatus::Running;    

    // --- initization ---
    Vec5 x = project(x0, lb, ub); 
    Vec10 r;                      
    Mat10x5 J;

    // compute initial residual and Jacobian
    functor.compute_all(x, r, J);
    result.func_evals++;
    result.grad_evals++;   

    double cost = 0.5 * r.squaredNorm();
    double cost_prev = cost;
    // grad: g = J^T * r
    Vec5 grad = J.transpose() * r;      
    // use tau * max(diag(J^T J)) as initial lambda, as recommended in TRF paper, to better adapt to the problem scale
    Vec5 JtJ_diag = (J.transpose() * J).diagonal();
    double lambda = opts.tau * JtJ_diag.maxCoeff();
    if (lambda < 1e-15) lambda = 1e-3; 
    double nu = 2.0;  // lambda adjustment factor for the "small step" case
    // Vec5 D_scale = (J.transpose() * J).diagonal().cwiseMax(1e-20);   
    if (opts.verbose >= 1) {
        std::cout << "\n===== Projected LM Optimization =====" << std::endl;
        std::cout << "  n_params=" << n << ", n_residuals=" << m << std::endl;
        std::cout << "  Initial cost = " << cost << std::endl;
    }    
    // main loop
    for (int iter = 0; iter < opts.max_iter; ++iter){
        result.iterations = iter + 1;
        if (cost < opts.cost_threshold) {
            result.status = ProjectedLMStatus::ConvergedFtol;
            result.message = "Converged: absolute cost below threshold";
            break;
        }

        bool cost_is_small = (cost < 1e-16);
        bool past_min_iter = (iter >= opts.min_iter);
        bool relaxed_stop  = cost_is_small || past_min_iter;
        
        // -------- Convergence Criterion: Projected Gradient --------
        double pg_norm = projected_gradient_inf_norm(x, grad, lb, ub);
        if (relaxed_stop && pg_norm < opts.gtol){
            result.status = ProjectedLMStatus::ConvergedGtol;
            result.message = "Converged: projected gradient norm < gtol";
            break;
        }
        // -------- 1. Solve the LM subproblem--------
        // (J^T J + lambda * I) * step = -J^T r

        // Determine liveness constraints
        const double tol = 1e-12;
        Vec5_bool active_lb = (x.array() <= lb.array() + tol) && (grad.array() > 0);
        Vec5_bool active_ub = (x.array() >= ub.array() - tol) && (grad.array() < 0);
        Vec5_bool free_var = !(active_lb || active_ub);
        int n_free = free_var.count();
        
        // build free variable mask and scaling
        Vec5 step = Vec5::Zero();
        if (n_free > 0){
            Mat5x5 A = J.transpose() * J;
            Vec5 rhs = -(J.transpose() * r); // g = J^T * r
            // A.diagonal().array() += lambda * D_scale.array();
            A.diagonal().array() += lambda;   

            for (int i = 0; i < 5; ++i) {
                if (!free_var(i)) {
                    A.row(i).setZero();
                    A.col(i).setZero(); 
                    A(i, i) = 1.0;
                    rhs(i) = 0.0;
                }
            }
            Eigen::LDLT<Mat5x5> solver(A);
        
            if (solver.info() != Eigen::Success) {
                // solver failed, treat as small step
                lambda *= 4.0;
                nu = 2.0;
                continue; 
            }
            step = solver.solve(rhs);
        }
        // -------- 2. Step size trimming (scale reduction to feasible region) --------
        // Calculate the maximum allowable step size ratio alpha_max ∈ (0, 1] such that x + alpha_max * step is within the bounds [lb, ub]
        double alpha = 1.0;

        for (int i = 0; i < 5; ++i) {
            if (step(i) > 1e-20) {
                double max_step = ub(i) - x(i);
                alpha = std::min(alpha, max_step / step(i));
            } 
            else if (step(i) < -1e-20) {
                double max_step = lb(i) - x(i);
                alpha = std::min(alpha, max_step / step(i));
            }
        }
        alpha = std::max(0.0, alpha);
        // actual step after trimming
        Vec5 dx = alpha * step;
        Vec5 x_new = (x + dx).cwiseMax(lb).cwiseMin(ub);
        dx = x_new - x;
        double step_norm = dx.norm();
        double x_norm = x.norm();

        // -------- Convergence criterion: Step size too small --------
        if (relaxed_stop && step_norm < opts.xtol * (opts.xtol + x_norm)) {
            result.status = ProjectedLMStatus::ConvergedXtol;
            result.message = "Converged: step size < xtol";
            break;
        }

        // -------- 3. new points of evaluation --------
        Vec10 r_new;
        Mat10x5 J_new;
        functor.compute_all(x_new, r_new, J_new);
        // functor(x_new, r_new);
        result.func_evals++;
        result.grad_evals++;
        double cost_new = 0.5 * r_new.squaredNorm();

        // -------- 4. Gain ratio and lambda adjustment --------
        // Actual reduction / Projected reduction
        double actual_reduction = cost - cost_new;
        // Projected reduction: L(0) - L(step) = -g^T step - 0.5 step^T (J^T J) step
        //   ≈ -grad^T dx - 0.5 * ||J * dx||^2  
        Vec10 Jdx = J * dx;
        double predicted_reduction = -grad.dot(dx) - 0.5 * Jdx.squaredNorm();
        // double predicted_reduction = -grad.dot(dx) - 0.5 * Jdx.squaredNorm() - 0.5 * lambda * (D_scale.array() * dx.array().square()).sum();
        double rho = 0.0;
        if (std::abs(predicted_reduction) > 1e-30) {
            rho = actual_reduction / predicted_reduction;
        } else if (actual_reduction > 0) {
            rho = 1.0;  
        }
        if (opts.verbose >= 2) {
            std::cout << "  iter=" << iter
                      << " cost=" << cost
                      << " cost_new=" << cost_new
                      << " rho=" << rho
                      << " lambda=" << lambda
                      << " |step|=" << step_norm
                      << " |pg|=" << pg_norm
                      << " n_free=" << n_free
                      << std::endl;
        }
        // -------- 5. Update strategy--------
        if (rho > 1e-4) {
            // accept the step
            double cost_change = std::abs(cost - cost_new);
            if (relaxed_stop && cost_change < opts.ftol * std::max(cost, 1e-10)) {
                x = x_new;
                // r = r_new; 
                cost_prev = cost;
                cost = cost_new;
                result.final_cost = cost; 
                result.status = ProjectedLMStatus::ConvergedFtol;
                result.message = "Converged: cost change < ftol";
                break;
            }

            x = x_new;
            r = r_new;
            cost_prev = cost;
            cost = cost_new;

            // renew Jacobian at new point
            J = J_new;
            JtJ_diag = (J.transpose() * J).diagonal();
            // D_scale = D_scale.cwiseMax(JtJ_diag).cwiseMax(1e-20);
            // D_scale = JtJ_diag.cwiseMax(1e-20);
            // functor.df(x, J);
            // result.grad_evals++;

            grad = J.transpose() * r;

            // decrease lambda (incease trust region) if the model is a good predictor
            // Nielsen strategy
            double temp = 2.0 * rho - 1.0;
            double factor = 1.0 - temp * temp * temp;
            lambda *= std::max(factor, 1.0 / 3.0);
            nu = 2.0;
        } else {
            // Reject the step size and increase lambda (shrink the trust region).
            lambda *= nu;
            nu *= 2.0;
        }
        // Maximum number of function evaluations
        if (result.func_evals >= opts.max_fev) {
            result.status = ProjectedLMStatus::MaxFevReached;
            result.message = "Maximum function evaluations reached";
            break;
        }

    }
    if (result.status == ProjectedLMStatus::Running) {
        result.status = ProjectedLMStatus::MaxIterReached;
        result.message = "Maximum iterations reached";
    }
    result.x = x;
    result.final_cost = 0.5 * r.squaredNorm();

    if (opts.verbose >= 1) {
        std::cout << "\n  Result: " << result.message << std::endl;
        std::cout << "  Final cost = " << result.final_cost << std::endl;
        std::cout << "  Iterations = " << result.iterations << std::endl;
        std::cout << "  Func evals = " << result.func_evals << std::endl;
        std::cout << "  Grad evals = " << result.grad_evals << std::endl;
        std::cout << "  x = " << x.transpose() << std::endl;
        std::cout << "=====================================" << std::endl;
    }

    return result;
}