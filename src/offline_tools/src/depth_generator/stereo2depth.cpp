#include "depth_generator/Stereo2DepthNode.hpp"

#include <fmt/format.h>
#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    spdlog::info("Hello from stereo2depth!");

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<tools::Stereo2DepthNode>(argc, argv));
    rclcpp::shutdown();

    spdlog::info("Stereo2DepthNode quits successfully!");

    return 0;
}