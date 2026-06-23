from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

BREW_PREFIX = "/opt/homebrew"
gsl_include = f'{BREW_PREFIX}/include'
gsl_lib = f'{BREW_PREFIX}/lib'

extensions = [
    Extension(
        "WuWanv021",
        ["WuWanv021.pyx", "mpfit.c"],
        #"WuWanv021_MCMC",
        #["WuWanv021_MCMC.pyx"],
        include_dirs=[
            numpy.get_include(),
            gsl_include,
        ],
        library_dirs=[
            gsl_lib,
        ],
        libraries=[
            "gsl", 
            "gslcblas",
        ],
        extra_compile_args=[
            "-O3", 
            "-march=native",           
            "-ffast-math",            
            "-funroll-loops",         
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
        ],
        language='c'
    ),
    Extension(
        "gsl_wrappers",
        ["gsl_wrappers.pyx"],
        #["WuWanv02.pyx",],
        include_dirs=[
            numpy.get_include(),
            gsl_include,
        ],
        library_dirs=[
            gsl_lib,
        ],
        libraries=[
            "gsl", 
            "gslcblas",
        ],
        extra_compile_args=[
            "-O3", 
            "-march=native",           
            "-ffast-math",            
            "-funroll-loops",         
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
        ],
        language='c'
    ),
    Extension(
        "math_fun",
        ["math_fun.pyx"],
        #["WuWanv02.pyx",],
        include_dirs=[
            numpy.get_include(),
            gsl_include,
        ],
        library_dirs=[
            gsl_lib,
        ],
        libraries=[
            "gsl", 
            "gslcblas",
        ],
        extra_compile_args=[
            "-O3", 
            "-march=native",           
            "-ffast-math",            
            "-funroll-loops",         
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
        ],
        language='c'
    ),
    Extension(
        "processing_function",
        ["processing_function.pyx"],
        #["WuWanv02.pyx",],
        include_dirs=[
            numpy.get_include(),
            gsl_include,
        ],
        library_dirs=[
            gsl_lib,
        ],
        libraries=[
            "gsl", 
            "gslcblas",
        ],
        extra_compile_args=[
            "-O3", 
            "-march=native",           
            "-ffast-math",            
            "-funroll-loops",         
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
        ],
        language='c'
    ),
    Extension(
        "integrand_solver_gradient",
        ["integrand_solver_gradient.pyx"],
        #["WuWanv02.pyx",],
        include_dirs=[
            numpy.get_include(),
            gsl_include,
        ],
        library_dirs=[
            gsl_lib,
        ],
        libraries=[
            "gsl", 
            "gslcblas",
        ],
        extra_compile_args=[
            "-O3", 
            "-march=native",           
            "-ffast-math",            
            "-funroll-loops",         
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
        ],
        language='c'
    )
]

setup(
    name="compute_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3", 
            'profile': False,
            'linetrace': False,
            'binding': False
        },
        annotate=True
    ),
)
