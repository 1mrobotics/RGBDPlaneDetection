cmake_minimum_required(VERSION 3.4...3.18)

project(plane_detection)

add_subdirectory(pybind11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-fpermissive")
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_INSTALL_PREFIX usr/local)


set(MRF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/MRF2.2)

add_custom_target(mrf 
    BYPRODUCTS ${MRF_DIR}/libMRF.a
    COMMAND make -C ${MRF_DIR}
)

# put FindGLFW.cmake in the source directory of CMakeLists.txt
find_package(OpenCV REQUIRED  core imgcodecs imgproc)
find_package(Eigen3)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include/peac)
include_directories(include/MRF2.2)
#
link_directories(include/MRF2.2)

python_add_library(plane_detection MODULE plane_detection.cpp mrf_optimization.cpp WITH_SOABI)

target_compile_options(plane_detection PUBLIC "-Wno-register")

target_link_libraries(plane_detection PRIVATE libMRF.a)
target_link_libraries(plane_detection PRIVATE ${OpenCV_LIBS} Eigen3::Eigen pybind11::headers)
add_dependencies(plane_detection mrf)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(plane_detection
        PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

install(TARGETS plane_detection DESTINATION .)
