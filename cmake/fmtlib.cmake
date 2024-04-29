include_guard()
include(${CMAKE_DIR}/fetchcontent_declare.cmake)

FetchContent_Declare_GH(fmtlib fmtlib/fmt 7.0.2)

FetchContent_GetProperties(fmtlib)
if(NOT fmtlib_POPULATED)
    FetchContent_Populate(fmtlib)

    add_subdirectory(${fmtlib_SOURCE_DIR} ${fmtlib_BINARY_DIR} EXCLUDE_FROM_ALL)

    set(CMAKE_DISABLE_FIND_PACKAGE_fmt TRUE)
    set(fmt_FOUND TRUE)
endif()
