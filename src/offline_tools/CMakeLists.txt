cmake_minimum_required(VERSION 3.8)
project(offline_tools)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../cmake)
# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_C_FLAGS_DEBUG "-O0 -Wall")
# set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -std=c++17")
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

set(FETCHCONTENT_PREFER_GIT ON)
include(${CMAKE_DIR}/eigen.cmake)
include(${CMAKE_DIR}/fmtlib.cmake)
include(${CMAKE_DIR}/spdlog.cmake)
include(${CMAKE_DIR}/json.cmake)
include(${CMAKE_DIR}/opencv.cmake)
include(${CMAKE_DIR}/libSGM.cmake)
include(${CMAKE_DIR}/cli11.cmake)

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

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
