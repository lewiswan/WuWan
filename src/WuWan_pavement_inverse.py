import time
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.special import jv
from scipy.integrate import quad  
from scipy.special import j0, j1
#import WuWan_pavement_forward
from scipy.optimize import least_squares
from tkinter import *  

class ForwardModelLogCached:
    def __init__(self, arr_template, u_target, x_prior_log, sigme_prior, forward_module=None):
        self.forward_module = forward_module
        self.arr = arr_template.copy()
        self.u_target = u_target
        self.sigma_prior = sigme_prior
        self.x_prior_log = x_prior_log
        
        # Cache variables
        self.last_log_x = None
        self.last_u = None
        self.last_J_log = None
        
        # Counters
        self.call_count = 0
        self.grad_call_count = 0

    def _compute(self, log_x, need_gradient=False):
        """
        Input: log_x (Natural logarithmic modulus)
               need_gradient (bool): whether need to calculate the gradient?
        Output: u_pred, J_log (if need_gradient=True) or None
        """
        # 1. check cache
        if self.last_log_x is not None and np.array_equal(log_x, self.last_log_x):
            # Case A: Gradients are needed, and they are in the cache -> Return directly.
            if need_gradient and self.last_J_log is not None:
                return self.last_u, self.last_J_log
            # Case B: No gradient needed -> Directly return the cached u
            elif not need_gradient:
                return self.last_u, None
            # Case C: Gradient is needed, but the cache only contains u and not J -> Continue execution to calculate the gradient.
            else:
                pass 

        # 2. Restoring physical modulus
        real_x = np.exp(log_x)
        E1, E2, E3, E4, E5 = real_x
        
        # Update parameters
        self.arr[1, 2] = E1
        self.arr[1, 3] = E2
        self.arr[1, 4] = E3
        self.arr[1, 5] = E4
        self.arr[1, 6] = E5
        
        arr_input = self.arr.T.copy()
        arr_input = np.ascontiguousarray(arr_input, dtype=np.float64)
        
        # Call C++ function
        result = self.forward_module.Calculation(arr_input, calc_grad=need_gradient)
        u1, u2, u3, u4, u5, u6, u7, u8, u9, u10 = result.result_displacement
        u_pred = np.array([u1, u2, u3, u4, u5, u6, u7, u8, u9, u10])
        
        # 3. Processing gradients
        J_log = None
        if need_gradient:
            J_phys = np.array(result.J_E)  # shape (10, 5)
            J_log = J_phys * real_x        # dU/d(logE) = dU/dE * E
            self.grad_call_count += 1
        
        # 4. Update cache
        self.last_log_x = log_x.copy()
        self.last_u = u_pred
        
        if need_gradient:
            self.last_J_log = J_log
        else:
            self.last_J_log = None
        
        self.call_count += 1
        
        return u_pred, J_log

    def fun(self, log_x):
        """Cost function"""
        u_pred, _ = self._compute(log_x, need_gradient=False)
        return (u_pred - self.u_target) / self.u_target

    def jac(self, log_x):
        """Jacobian function"""
        _, J_log = self._compute(log_x, need_gradient=True)
        return J_log / self.u_target[:, None]

def add_error(arr, arr2):
    u_input = arr[7, 1:] * 1e-3 # convert um to mm
    rng = np.random.default_rng() 

    # deflection error
    error_deflection_mm = arr2[6, 1:] * 1e-3
    mask = error_deflection_mm != 0
    noise_deflection = np.zeros_like(u_input)
    if mask.any():
        noise_deflection[mask] = rng.triangular(left=-error_deflection_mm[mask], mode=0.0, right=error_deflection_mm[mask], size=mask.sum())
    
    u_target = u_input + noise_deflection

    # thickness error
    error_thickness_mm = arr2[3, 2:6]
    mask_thickness = error_thickness_mm != 0
    noise_thickness = np.zeros_like(error_thickness_mm)
    thickness_original = arr[3, 2:6].copy()
    if mask_thickness.any():
        noise_thickness[mask_thickness] = rng.triangular(left=-error_thickness_mm[mask_thickness], mode=0.0, right=error_thickness_mm[mask_thickness], size=mask_thickness.sum())
    
    arr[3, 2:6] = noise_thickness + arr[3, 2:6]
    
    # r error
    error_r_mm = arr2[5, 2:]
    mask_r = error_r_mm != 0
    noise_r = np.zeros_like(error_r_mm)
    r_original = arr[-2, 2:].copy()
    if mask_r.any():
        noise_r[mask_r] = rng.triangular(left=-error_r_mm[mask_r], mode=0.0, right=error_r_mm[mask_r], size=mask_r.sum())
    
    arr[-2, 2:] = noise_r + arr[-2, 2:]

    # load error
    error_load = arr2[1, -1] * arr[1, -1]
    noise_load = 0.0
    load_original = arr[1, -1]
    if error_load != 0:
        noise_load = rng.triangular(left=-error_load, mode=0.0, right=error_load)
    arr[1, -1] = noise_load + arr[1, -1]
    
    # Print results
    print("="*80)
    print("ERROR ANALYSIS - ORIGINAL vs MODIFIED VALUES")
    print("="*80)
    
    print("\n--- Deflection ---")
    print(f"Original deflection [mm]: {u_input}")
    print(f"Added noise [mm]: {noise_deflection}")
    print(f"Modified deflection [mm]: {u_target}")
    
    print("\n--- Thickness ---")
    print(f"Original thickness [mm]: {thickness_original}")
    print(f"Added noise [mm]: {noise_thickness}")
    print(f"Modified thickness [mm]: {arr[3, 2:6]}")
    
    print("\n--- Radius ---")
    print(f"Original radius [mm]: {r_original}")
    print(f"Added noise [mm]: {noise_r}")
    print(f"Modified radius [mm]: {arr[-2, 2:]}")
    
    print("\n--- Load ---")
    print(f"Original load [MPa]: {load_original}")
    print(f"Added noise [MPa]: {noise_load}")
    print(f"Modified load [MPa]: {arr[1, -1]}")
    print("="*80)
    
    return u_target, arr

def Backcalculation(arr, arr2, forward_module=None):
    arr = arr.T
    arr2 = arr2.T
    u_target, arr = add_error(arr, arr2)
    lower_bounds_phys = arr2[1, 2:7]
    upper_bounds_phys = arr2[2, 2:7]
    x0_phys = arr[1, 2:7]
    
    x_prior_log = (np.log(lower_bounds_phys) + np.log(upper_bounds_phys)) / 2
    sigme_prior = (np.log(upper_bounds_phys) - np.log(lower_bounds_phys)) / 4
    if np.any(x0_phys == 0):
        x0_phys = np.exp(x_prior_log)
    print("\n  **Initial modulus values [MPa]:**", x0_phys)
    x0_log = np.log(x0_phys)

    model = ForwardModelLogCached(arr, u_target, x_prior_log, sigme_prior, forward_module=forward_module)

    bounds_log = (
        np.log(lower_bounds_phys), 
        np.log(upper_bounds_phys)
    )

    print("Start optimizing...")
    start_time = time.time()

    result = least_squares(
        model.fun,
        x0_log,                
        jac=model.jac,         
        bounds=bounds_log,     
        method='trf',
        verbose=2,
        max_nfev=100,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9
    )

    elapsed = time.time() - start_time

    final_log_x = result.x
    final_phys_x = np.exp(final_log_x)

    print(f"\n  **Optimization complete!**")
    print(f"**Run time:** {elapsed:.3f} seconds")
    print(f"**Actual number of C++ calls:** {model.call_count}")
    print(f"**Number of gradient calculations:** {model.grad_call_count}")
    print(f"**Number of iterations (nfev):** {result.nfev}")
    print(f"**Final residual:** {result.cost:.6e}")

    print(f"\n  **Inversion modulus results [MPa]:**")
    for i, (final, initial) in enumerate(zip(final_phys_x, arr[1, 2:7]), 1):
        print(f" Final **E{i}** = {final:.6f}")


    return final_phys_x