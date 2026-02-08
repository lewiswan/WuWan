# WuWan: High-Performance Layered Elastic Half-Space Solver

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-red)](https://isocpp.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()

**WuWan** is a high-performance computational library designed for the forward and inverse analysis of pavement mechanics. It solves deflections in a **5-layer elastic half-space** system with exceptional speed and accuracy.

Built with a modern **C++ backend** (relying on Boost and Eigen) and wrapped for **Python**, it leverages analytical gradients and custom linear algebra optimizations to achieve millisecond-level inversions.

---

## 🚀 Key Features

* **High-Performance Core**: Originally a Cython project, now fully rewritten in C++ for maximum efficiency.
* **Analytical Gradients**: Implements analytical derivatives for Jacobian calculations, significantly outperforming finite difference methods in stability and speed.
* **Advanced Back-Calculation**:
    * Utilizes `scipy.optimize` combined with high-speed C++ gradient providers.
    * Solves inverse problems in **tens of milliseconds**.
* **Robust Error Modeling**: Supports noise injection for thickness, deflection, load, and sensor positioning to simulate real-world measurement uncertainties.

---

## 🧠 Methodology

The core algorithm solves the Layered Elastic Theory (LET) equations using advanced numerical techniques:

1.  **Hankel Transform**: The integral transform is converted into algebraic equations using high-precision **Gauss-Legendre quadrature**.
2.  **System Solving**: Instead of using generic solvers, WuWan employs a **custom implementation of LU decomposition**. This is specifically optimized for the sparse, banded structure of the 5-layer system matrices, reducing memory overhead and computation time.
3.  **Gradient Computation**: The Jacobian matrix is computed via **analytical derivation** of the stiffness matrix. This allows for precise sensitivity analysis without the computational overhead or truncation errors associated with numerical differentiation.

---

## 📊 Performance Benchmarks

Benchmarks performed on a standard workstation (Single-threaded):

| Operation | Batch Size | Computation Time | Note |
| :--- | :--- | :--- | :--- |
| **Forward Calculation** | 10,000 calls (10 points/call) | **~1.5 seconds** | Pure deflection calculation |
| **Forward + Gradient** | 10,000 calls | **~3.5 seconds** | Deflection + Jacobian w.r.t moduli |
| **Inverse Analysis** | Single Basin | **~10 - 50 ms** | Dependent on convergence criteria |

> **Note:** The solver is optimized to handle large-scale batch processing for sensitivity analysis and probabilistic inversion.

---

## ⚡ Quick Start: Forward Calculation

This example demonstrates how to perform a high-speed forward calculation for a 5-layer pavement system and retrieve both deflections and analytical gradients (Jacobian) using the **WuWan** C++ backend.

### 1. Setup Data and Call Solver

The following logic follows the standard workflow: defining the structure, preparing contiguous memory for C++, and executing the solver.

```python
import time
import numpy as np
import pandas as pd
import WuWan_pavement_forward
data = {
        'Layer':            ['Layered System', 'layer #', '1', '2', '3', '4', '5', '', '', '','Stress [MPa]'],
        'Modulus [MPa]':    ['Layered System', 'Modulus [MPa]', '4000', '400', '300', '200', '100', '', '', '','0.95'],
        'Poisson [-]':      ['Layered System', 'Poisson [-]', '0.30', '0.35', '0.35', '0.40', '0.40', '', '', '', 'Radius [mm]'],
        'Thickness [mm]':   ['Layered System', 'Thickness [mm]', '150', '240', '300', '500', 'semi-inf', '', '', '','150'],
        '':  ['', '', '', '', '', '', '', '', '', '', ''],
        'Evaluation point': ['Evaluation point #', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10' ],
        'r [mm]':           ['r [mm]', '0', '300', '600', '900', '1200', '1500', '1800', '2000', '3000', '4000'],
        'Deflection [μm]':  ['Deflection [μm]', '', '', '', '', '', '', '', '', '', ''],
        }
df = pd.DataFrame(data)
df = pd.DataFrame.from_dict(data, orient='index')
df.to_csv('data.csv', index=False)

df_num = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
arr = df_num.to_numpy()
arr = np.ascontiguousarray(df_num.to_numpy(), dtype=np.float64)
input_data = np.ascontiguousarray(arr.T)
# === Call C++ ===
print("Python: Calling C++...")
ret = WuWan_pavement_forward.Calculation(input_data, calc_grad=True)
end = time.perf_counter()
print("deflections:", ret.result_displacement)
print("grad:", ret.J_E)
```

### 2. Output & Visualization

The solver generates the surface deflection basin profile. Below are the results from the demo:

```Plaintext
Python: Calling C++...
deflections: [0.44186744 0.30873646 0.21676407 0.16808461 0.13797657 0.11674337 
               0.10064475 0.09188363 0.0624335  0.04627289]
grad: [[-2.48956993e-05 -2.35325278e-04 -2.04079722e-04 -2.63150101e-04
  -1.34300592e-03]
 [-5.45282398e-06 -1.37737859e-04 -1.68268547e-04 -2.43524360e-04
  -1.32644585e-03]
 [-8.72171458e-07 -4.12614680e-05 -9.94771277e-05 -1.95007054e-04
  -1.27926249e-03]
 [-6.48066584e-07 -7.45964845e-06 -4.64284696e-05 -1.38462798e-04
  -1.20887387e-03]
 [-6.40643617e-07  6.83431698e-07 -1.78697327e-05 -8.93774172e-05
  -1.12450960e-03]
 [-5.31834037e-07  2.23559776e-06 -4.80736168e-06 -5.30852137e-05
  -1.03451018e-03]
 [-4.05946534e-07  2.41149686e-06  8.08046065e-07 -2.85735311e-05
  -9.45132738e-04]
 [-3.33276661e-07  2.32692214e-06  2.57471518e-06 -1.73096406e-05
  -8.87917836e-04]
 [-1.12120438e-07  1.37095054e-06  3.73979321e-06  5.75100302e-06
  -6.48055323e-04]
 [-3.64239311e-08  5.61953149e-07  2.21478585e-06  7.54918589e-06
  -4.85262516e-04]]
```
![Deflection Profile](demo_figure/wuwan_deflection_profile.png)

---

## 🔮 Roadmap

- [x] **C++ Core Rewrite**: Transformed from Cython to C++ with Eigen/Boost.
- [x] **Forward Calculation & Analytical Gradients**: Implementation of high-speed forward modeling and derivative calculation.
- [x] **Deterministic Back-calculation**: Fast inverse analysis for moduli estimation.
- [ ] **Bayesian Uncertainty Analysis**: Implementation of MCMC or variational inference for posterior distributions (In Progress).
- [ ] **Global Sensitivity Analysis**: Sobol indices or similar methods to quantify parameter influence (In Progress).
- [ ] **Batch Error Simulation**: Wrappers for large-scale Monte Carlo simulations with noise injection.

---

## 🛠 Installation & Dependencies

### Prerequisites
* **C++ Compiler** supporting C++14/17
* **Boost Math Library**
* **Eigen3 Linear Algebra Library**
* **Python 3.x**

### Building from Source

#### Clone the Repository
```bash
git clone https://github.com/lewiswan/WuWan.git
cd WuWan
```

#### Option A: Standard Installation (Recommended for Users)

This method automatically sets up a build environment, downloads necessary C++ libraries (Eigen & Boost), and compiles the project.
```bash
pip install .
```

**Note:** The first installation may take a few minutes as it downloads the Boost C++ headers.

#### Option B: Fast Re-installation (Recommended for Developers)

If you are modifying the C++ code or reinstalling frequently, use this method. It utilizes build isolation disabled to persist the CMake cache. This prevents re-downloading Boost/Eigen on every build, reducing compile time to seconds.

1. Install build tools (one-time setup):
```bash
pip install cmake ninja pybind11
```

2. Fast install command:
```bash
pip install . --no-build-isolation --no-deps --force-reinstall
```