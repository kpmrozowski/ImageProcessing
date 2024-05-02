#include "depth_generator/Publishers.hpp"

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include <rclcpp/node.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include "cv_bridge_interface/Converter.hpp"

namespace tools {

Publishers::Publishers(rclcpp::Node* node, const YAML::Node& cfg) : cfg_(cfg) {
    const rclcpp::QoS qos(10);

    // publishers
    pub_color_ = node->create_publisher<Image>("gen/color", qos);
    pub_disparityL_ = node->create_publisher<Image>("gen/disparityL", qos);
    pub_disparityR_ = node->create_publisher<Image>("gen/disparityR", qos);
    pub_depthL_ = node->create_publisher<Image>("gen/depthL", qos);
    pub_depthR_ = node->create_publisher<Image>("gen/depthR", qos);
    pub_local_point_cloud_ = node->create_publisher<PointCloud>("gen/local_cloud", qos);
}

void Publishers::PublishColor(const cv::Mat& img, const std_msgs::msg::Header& header) {
    if (not cfg_["DepthGeneration"]["computeColor"].as<bool>()) {
        return;
    }
    if (nullptr == pub_color_) {
        spdlog::warn("Waiting for pub_color_ to be advertised");
        return;
    }
    if (not cfg_["verbose"].as<bool>()) {
        spdlog::info("PublishColor");
    }
    const Image img_msg = Converter::Convert(img, sensor_msgs::image_encodings::BGR8, header);
    try {
        pub_color_->publish(img_msg);
    } catch (const std::exception& e) {
        spdlog::warn("publisher pub_color_: {}", e.what());
    }
}

template <typename pub_t>
static void PublishDisparity(const cv::Mat& img, const std_msgs::msg::Header& header, const bool compute,
                             const bool verbose, pub_t& publisher) {
    if (not compute) {
        return;
    }
    if (nullptr == publisher) {
        spdlog::warn("Waiting for pub_disparity_ to be advertised");
        return;
    }
    if (not verbose) {
        spdlog::info("PublishDisparity");
    }
    cv::Mat cv_img;
    img.convertTo(cv_img, CV_16UC1);
    const sensor_msgs::msg::Image img_msg = Converter::Convert(cv_img, sensor_msgs::image_encodings::MONO16, header);
    try {
        publisher->publish(img_msg);
    } catch (const std::exception& e) {
        spdlog::warn("publisher pub_disparity_: {}", e.what());
    }
}

void Publishers::PublishDisparityL(const cv::Mat& img, const std_msgs::msg::Header& header) {
    PublishDisparity(img, header, cfg_["DepthGeneration"]["computeDisparity"].as<bool>(), cfg_["verbose"].as<bool>(),
                     pub_disparityL_);
}

void Publishers::PublishDisparityR(const cv::Mat& img, const std_msgs::msg::Header& header) {
    PublishDisparity(img, header, cfg_["DepthGeneration"]["computeDisparity"].as<bool>(), cfg_["verbose"].as<bool>(),
                     pub_disparityR_);
}

template <typename pub_t>
static void PublisDepth(const cv::Mat& img, const std_msgs::msg::Header& header, const bool compute,
                             const bool verbose, pub_t& publisher) {
    if (not compute) {
        return;
    }
    if (nullptr == publisher) {
        spdlog::warn("Waiting for pub_depth_ to be advertised");
        return;
    }
    if (not verbose) {
        spdlog::info("PublishDepth");
    }
    const sensor_msgs::msg::Image img_msg = Converter::Convert(img, sensor_msgs::image_encodings::TYPE_32FC1, header);
    try {
        publisher->publish(img_msg);
    } catch (const std::exception& e) {
        spdlog::warn("publisher pub_depth_: {}", e.what());
    }
}

void Publishers::PublishDepthL(const cv::Mat& img, const std_msgs::msg::Header& header) {
    PublisDepth(img, header, cfg_["DepthGeneration"]["computeDepth"].as<bool>(), cfg_["verbose"].as<bool>(), pub_depthL_);
}

void Publishers::PublishDepthR(const cv::Mat& img, const std_msgs::msg::Header& header) {
    PublisDepth(img, header, cfg_["DepthGeneration"]["computeDepth"].as<bool>(), cfg_["verbose"].as<bool>(), pub_depthR_);
}

void Publishers::PublishLocalPointCloud(const Eigen::Matrix3Xf& cloud, const std_msgs::msg::Header& header) {
    if (not cfg_["DepthGeneration"]["computePointCloud"].as<bool>()) {
        return;
    }
    if (nullptr == pub_local_point_cloud_) {
        spdlog::warn("Waiting for pub_local_point_cloud_ to be advertised");
        return;
    }
    if (not cfg_["verbose"].as<bool>()) {
        spdlog::info("publishLocalPointCloud");
    }
    const size_t pcSize = cloud.cols();

    PointCloud msg;
    msg.points.resize(pcSize);
    for (int pointId = 0; pointId < cloud.cols(); ++pointId) {
        Point32 point;
        point.x = cloud(0, pointId);
        point.y = cloud(1, pointId);
        point.z = cloud(2, pointId);
        msg.points[pointId] = std::move(point);
    }
    msg.header = header;
    msg.header.frame_id = "map";

    ChannelFloat32 rgbChannel;
    rgbChannel.name = "rgb";
    uint8_t rColor = 1u, gColor = 1u, bColor = 255u;
    uint32_t rgb = 0;
    rgb += rColor;
    rgb <<= 8;
    rgb += gColor;
    rgb <<= 8;
    rgb += bColor;
    float floatRgb = *reinterpret_cast<float*>(&rgb);
    rgbChannel.values = std::vector<float>(pcSize, floatRgb);
    msg.channels.emplace_back(std::move(rgbChannel));

    try {
        pub_local_point_cloud_->publish(msg);
    } catch (const std::exception& e) {
        spdlog::warn("publisher pub_local_point_cloud_: {}", e.what());
    }
}

}  // namespace tools
