include_guard()
include(${CMAKE_DIR}/fetchcontent_declare.cmake)

set(EIGEN3_WORLD_VERSION 3)
set(EIGEN3_MAJOR_VERSION 4)
set(EIGEN3_MINOR_VERSION 0)
set(VERSION ${EIGEN3_WORLD_VERSION}.${EIGEN3_MAJOR_VERSION}.${EIGEN3_MINOR_VERSION})

FetchContent_Declare_GL(eigen libeigen/eigen ${VERSION})

FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()

set(CMAKE_DISABLE_FIND_PACKAGE_Eigen3 TRUE)
set(Eigen3_FOUND TRUE)
set(EIGEN3_FOUND TRUE)
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${eigen_SOURCE_DIR}
)
