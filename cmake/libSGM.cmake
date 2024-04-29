include_guard()
include(${CMAKE_DIR}/fetchcontent_declare.cmake)

FetchContent_Declare_GH(libsgm kpmrozowski/libSGM b802ab0825190c3ff71db186a035d92697be03e9)

FetchContent_GetProperties(libsgm)
if(NOT libsgm_POPULATED)
    FetchContent_Populate(libsgm)

    add_subdirectory(${libsgm_SOURCE_DIR} ${libsgm_BINARY_DIR} EXCLUDE_FROM_ALL)

    set(libsgm_FOUND TRUE)
endif()
