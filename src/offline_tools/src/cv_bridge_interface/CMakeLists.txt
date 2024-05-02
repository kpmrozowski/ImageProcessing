cmake_minimum_required(VERSION 3.8)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

add_library(cv_bridge_interface
    Converter.cpp
)
target_include_directories(cv_bridge_interface PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
ament_target_dependencies(cv_bridge_interface
    std_msgs
    cv_bridge
)

# target_compile_features(cv_bridge_interface PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_link_libraries(cv_bridge_interface
    Eigen3::Eigen
    fmt::fmt-header-only
    spdlog::spdlog_header_only
    nlohmann_json::nlohmann_json
    # efence
)

install(TARGETS cv_bridge_interface
    DESTINATION lib/${PROJECT_NAME}
)
