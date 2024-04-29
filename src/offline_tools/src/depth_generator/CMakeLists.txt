cmake_minimum_required(VERSION 3.8)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)

add_executable(stereo2depth
    DepthGeneration.cpp
    Stereo2DepthNode.cpp
    utils.cpp
    stereo2depth.cpp
)
target_include_directories(stereo2depth PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
ament_target_dependencies(stereo2depth
    rclcpp
    sensor_msgs
    message_filters
    cv_bridge
)
# target_compile_features(stereo2depth PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_link_libraries(stereo2depth
    Eigen3::Eigen
    fmt::fmt-header-only
    spdlog::spdlog_header_only
    nlohmann_json::nlohmann_json
    OpenCV::imgcodecs
    OpenCV::core
    OpenCV::calib3d
    OpenCV::ximgproc
    OpenCV::cudastereo
    sgm
    yaml-cpp
    CLI11::CLI11
    efence
)

install(TARGETS stereo2depth
    DESTINATION lib/${PROJECT_NAME}
)
