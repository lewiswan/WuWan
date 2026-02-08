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

# 1. Define Layered System Parameters (Example from demo_forward.pdf)
data = {
    'Modulus [MPa]': [4000, 400, 300, 200, 100],
    'Poisson [-]': [0.30, 0.35, 0.35, 0.40, 0.45],
    'Thickness [mm]': [150, 240, 300, 500, 0], # 0 for semi-infinite layer
    'Load': [0.707, 150] # Pressure [MPa] and Radius [mm]
}

# 2. Prepare Evaluation Points (0 to 4000 mm)
r_coords = [0, 300, 600, 900, 1200, 1500, 1800, 2000, 3000, 4000]

# 3. Data Preparation for C++ Backend
df = pd.DataFrame(data)
df_num = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
# Ensure memory is contiguous for the C++ pointer
input_data = np.ascontiguousarray(df_num.to_numpy().T, dtype=np.float64)

# 4. Execution
print("Python: Calling C++...")
# calc_grad=True enables the computation of analytical Jacobians
ret = WuWan_pavement_forward.Calculation(input_data, calc_grad=True)

print("Deflections:", ret.result_displacement)
print("Gradients (J_E):", ret.J_E)
```

### 2. Output & Visualization

The solver generates the surface deflection basin profile. Below are the results from the demo:

```Plaintext
Python: Calling C++...
deflections: [0.44186744 0.30873646 0.21676407 0.16808461 0.13797657 0.11674337 
               0.10064475 0.09188363 0.0624335  0.04627289]
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