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

### System Requirements

- **Operating System**: Linux (Ubuntu/Debian), macOS, Windows (WSL2 recommended)
- **Python**: 3.8 or higher
- **C++ Compiler**: 
  - Linux: GCC 7+ or Clang 6+
  - macOS: Xcode Command Line Tools (Clang)
  - Windows: MSVC 2017+ or MinGW-w64
- **CMake**: 3.15 or higher

---

### Installation Steps

#### Linux (Ubuntu/Debian)
```bash
# Update package index
sudo apt-get update

# Install build tools
sudo apt-get install -y build-essential cmake git

# Install Boost (including Boost.Math)
sudo apt-get install -y libboost-all-dev

# Install Eigen3
sudo apt-get install -y libeigen3-dev

# Install Python development headers
sudo apt-get install -y python3-dev python3-pip

# Verify installations
cmake --version          # Should show >= 3.15
gcc --version            # Should show >= 7.0
python3 --version        # Should show >= 3.8

# Check library paths (optional)
dpkg -L libeigen3-dev | grep eigen3
dpkg -L libboost-dev | grep boost

# Clone the repository
git clone https://github.com/lewiswan/WuWan.git
cd WuWan

# Install Python package
pip install .

# Verify installation
python -c "import WuWan_pavement_forward; print('Installation successful!')"
```

#### macOS
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake boost eigen python@3.11

# Verify installations
cmake --version
clang --version
python3 --version

# Check library paths (optional)
brew list eigen
brew list boost

# Clone the repository
git clone https://github.com/lewiswan/WuWan.git
cd WuWan

# Install Python package
pip3 install .

# Verify installation
python3 -c "import WuWan_pavement_forward; print('Installation successful!')"
```

#### Windows (WSL2 Recommended)

**Option 1: Using WSL2 (Recommended)**
```bash
# Install WSL2 with Ubuntu from PowerShell (as Administrator)
wsl --install -d Ubuntu

# After WSL2 is installed, open Ubuntu terminal and follow the Linux installation steps above
```

**Option 2: Native Windows with Visual Studio**
```powershell
# Install Visual Studio 2019 or later with C++ development tools
# Download from: https://visualstudio.microsoft.com/

# Install vcpkg package manager
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install dependencies
.\vcpkg install boost-math:x64-windows eigen3:x64-windows

# Integrate vcpkg with Visual Studio
.\vcpkg integrate install

# Clone the repository
git clone https://github.com/lewiswan/WuWan.git
cd WuWan

# Install Python package (from Developer Command Prompt for VS)
pip install .

# Verify installation
python -c "import WuWan_pavement_forward; print('Installation successful!')"
```

---

### Alternative: Using Conda (Cross-platform)
```bash
# Create a new conda environment
conda create -n wuwan python=3.11 -y
conda activate wuwan

# Install build tools and dependencies
conda install -c conda-forge cmake cxx-compiler boost-cpp eigen -y

# Clone the repository
git clone https://github.com/lewiswan/WuWan.git
cd WuWan

# Install Python package
pip install .

# Verify installation
python -c "import WuWan_pavement_forward; print('Installation successful!')"
```

---

### Troubleshooting

**Issue: CMake cannot find Eigen3**
```bash
# Manually specify Eigen3 path
export EIGEN3_INCLUDE_DIR=/usr/include/eigen3
pip install .
```

**Issue: Boost version incompatible**
```bash
# Use conda to install latest Boost
conda install -c conda-forge boost-cpp
```

**Issue: Build fails on macOS M1/M2**
```bash
# Install ARM native versions
brew install boost eigen
# Or use Rosetta 2 for x86_64
arch -x86_64 brew install boost eigen
```

---

### Dependency Version Requirements

| Dependency | Minimum Version | Installed by |
|:-----------|:----------------|:-------------|
| CMake | 3.15 | System package manager |
| Boost | 1.70 | System package manager |
| Eigen | 3.3 | System package manager |
| scikit-build-core | Latest | pip (automatic) |
| pybind11 | Latest | pip (automatic) |
| numpy | 1.26.4 | pip (locked version) |
| pandas | 2.3.3 | pip (locked version) |

---

## 🔮 Roadmap

- [x] **C++ Core Rewrite**: Transformed from Cython to C++ with Eigen/Boost.
- [x] **Forward Calculation & Analytical Gradients**: Implementation of high-speed forward modeling and derivative calculation.
- [x] **Deterministic Back-calculation**: Fast inverse analysis for moduli estimation.
- [ ] **Bayesian Uncertainty Analysis**: Implementation of MCMC or variational inference for posterior distributions (In Progress).
- [ ] **Global Sensitivity Analysis**: Sobol indices or similar methods to quantify parameter influence (In Progress).
- [ ] **Batch Error Simulation**: Wrappers for large-scale Monte Carlo simulations with noise injection.

---