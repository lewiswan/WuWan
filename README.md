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

```bash
git clone [https://github.com/lewiswan/WuWan.git](https://github.com/lewiswan/WuWan.git)
cd WuWan
pip install .
