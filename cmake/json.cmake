include_guard()
include(${CMAKE_DIR}/fetchcontent_declare.cmake)

FetchContent_Declare_GH(json nlohmann/json v3.9.1)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
    FetchContent_Populate(json)

    set(JSON_BuildTests OFF CACHE BOOL "Build the unit tests when BUILD_TESTING is enabled.")
    set(JSON_Install OFF CACHE BOOL Install "CMake targets during install step.")

    add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
