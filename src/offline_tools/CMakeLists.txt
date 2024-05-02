cmake_minimum_required(VERSION 3.8)
project(offline_tools)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_C_FLAGS_DEBUG "-O0 -Wall")
set(CMAKE_CXX_FLAGS "-std=c++17")
# set(CMAKE_C_FLAGS_RELEASE "-O0 -Wall")
# set(CMAKE_CXX_FLAGS_RELEASE "-O0 -Wall -g -std=c++17")

# YAML
find_package(yaml-cpp REQUIRED)

# CUDA
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_package(CUDA)
if(NOT CUDA_FOUND)
   message(FATAL_ERROR "CUDA 10.2 or 11.1 not found.")
else()
   message("CUDA ${CUDA_VERSION} found.")
endif()
find_package(CUDAToolkit REQUIRED)

# OpenCV
find_package(OpenCV 4.7.0 REQUIRED
   COMPONENTS imgcodecs core calib3d ximgproc cudastereo
   NO_MODULE #Should be optional, tells CMake to use config mode
   PATHS /usr/local # Tells CMake to look here
   NO_DEFAULT_PATH #and don't look anywhere else
)
if(NOT OpenCV_FOUND)
   message(FATAL_ERROR "OpenCV >= 4.7.0 not found.")
endif()
MESSAGE("OPENCV VERSION in offline_tools: ${OpenCV_VERSION}")

# Eigen3
find_package(Eigen3 3.4.0 REQUIRED)
MESSAGE("Eigen3 VERSION: ${Eigen3_VERSION}")

set(CMAKE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../cmake)
set(FETCHCONTENT_PREFER_GIT ON)
include(${CMAKE_DIR}/fmtlib.cmake)
include(${CMAKE_DIR}/spdlog.cmake)
include(${CMAKE_DIR}/json.cmake)
include(${CMAKE_DIR}/libSGM.cmake)
include(${CMAKE_DIR}/cli11.cmake)

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_subdirectory(src/cv_bridge_interface)
add_subdirectory(src/depth_generator)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # comment the line when a copyright and license is added to all source files
    set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # comment the line when this package is in a git repo and when
    # a copyright and license is added to all source files
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
