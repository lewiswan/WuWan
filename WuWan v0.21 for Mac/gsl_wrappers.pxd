# gsl_wrappers.pxd
from libc.stddef cimport size_t

# announcement the extern C
cdef extern from "gsl/gsl_sf_bessel.h" nogil:
    double gsl_sf_bessel_J0(double x)
    double gsl_sf_bessel_J1(double x)
    double gsl_sf_bessel_zero_J0(int n) 
    double gsl_sf_bessel_zero_J1(int n)

# announcement the extern C
cdef extern from "gsl/gsl_integration.h" nogil:
    # C Structure
    ctypedef struct gsl_integration_glfixed_table:
        size_t n
        double *x
        double *w
        
    gsl_integration_glfixed_table * gsl_integration_glfixed_table_alloc(size_t n)
    void gsl_integration_glfixed_table_free(gsl_integration_glfixed_table * t)
    int gsl_integration_glfixed_point(double a, double b, size_t i, double *xi, double *wi, const gsl_integration_glfixed_table *t)

cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng* gsl_rng_alloc(const gsl_rng_type * T)
    void gsl_rng_free(gsl_rng * r)
    void gsl_rng_set(gsl_rng * r, unsigned long int s)

cdef extern from "gsl/gsl_randist.h" nogil:
    double gsl_ran_gaussian(const gsl_rng * r, double sigma)