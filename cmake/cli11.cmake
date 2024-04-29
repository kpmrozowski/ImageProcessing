include_guard()
include(${CMAKE_DIR}/fetchcontent_declare.cmake)

set(VERSION 1.9.1)

if(USE_SYSTEM_LIBS)
    find_package(CLI11 ${VERSION})

    if(CLI11_FOUND)
        return()
    endif()
endif()

FetchContent_Declare_GH(cli11 CLIUtils/CLI11 "v${VERSION}")

FetchContent_GetProperties(cli11)
if(NOT cli11_POPULATED)
    FetchContent_Populate(cli11)

    add_subdirectory(${cli11_SOURCE_DIR} ${cli11_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
