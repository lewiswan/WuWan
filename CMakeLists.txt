cmake_minimum_required(VERSION 3.15)
project(WuWan_pavement)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "IPO/LTO enabled")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include(FetchContent)

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(Eigen3)

set(BOOST_VERSION "1.82.0")
string(REPLACE "." "_" BOOST_VERSION_UNDERSCORE ${BOOST_VERSION})

FetchContent_Declare(
  Boost
  URL https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_UNDERSCORE}.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_GetProperties(Boost)
if(NOT boost_POPULATED)
  message(STATUS "Downloading Boost ${BOOST_VERSION} from archives.boost.io...")
  FetchContent_Populate(Boost)
endif()

add_library(boost_headers INTERFACE)
target_include_directories(boost_headers INTERFACE ${boost_SOURCE_DIR})

python_add_library(WuWan_pavement_forward MODULE 
    src/forward_main.cpp 
    src/interand_solver.cpp 
    WITH_SOABI
)

target_include_directories(WuWan_pavement_forward PRIVATE src)

target_link_libraries(WuWan_pavement_forward PRIVATE 
    pybind11::headers 
    Eigen3::Eigen 
    boost_headers
)

target_compile_definitions(WuWan_pavement_forward PRIVATE 
    NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
    EIGEN_VECTORIZE
    EIGEN_DONT_ALIGN_STATICALLY=0
)

target_compile_features(WuWan_pavement_forward PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(WuWan_pavement_forward PRIVATE /O2 /Ob2 /arch:AVX2 /fp:fast)
else()
    target_compile_options(WuWan_pavement_forward PRIVATE 
        -O3 
        -g 
        -march=native 
        -ffast-math 
        -funroll-loops
        -Wno-dev
    )
endif()

install(TARGETS WuWan_pavement_forward DESTINATION .)