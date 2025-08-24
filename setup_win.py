from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os
import sys

CONDA_PREFIX = os.environ.get('CONDA_PREFIX')

# MinGW 特定的编译器选项
extra_compile_args = []
extra_link_args = []

if sys.platform == 'win32':
    # MinGW 特定选项
    extra_compile_args = [
        '-O2',
        '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '-msse2',  # 如果你想启用 SSE2 指令
    ]
    extra_link_args = [
        '-shared',
    ]

extensions = [
    Extension(
        "WuWanv02",
        ["WuWanv02.pyx"],
        include_dirs=[
            numpy.get_include(),
            os.path.join(CONDA_PREFIX, 'Library', 'include'),
        ],
        library_dirs=[
            os.path.join(CONDA_PREFIX, 'Library', 'lib'),
            os.path.join(CONDA_PREFIX, 'Library', 'bin'),
        ],
        libraries=[
         "gsl",
         "gslcblas",
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c',
    )
]

setup(
    name="WuWanv02",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
        }
    ),
)
