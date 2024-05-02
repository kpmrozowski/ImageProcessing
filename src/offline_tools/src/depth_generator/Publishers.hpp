#ifndef Publishers_HPP
#define Publishers_HPP

#include <yaml-cpp/node/node.h>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/detail/point32__struct.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <opencv2/core/mat.hpp>
#include <rclcpp/publisher.hpp>
#include <sensor_msgs/msg/channel_float32.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>

namespace rclcpp {
class Node;
}

namespace tools {

class Publishers {
    using Point32 = geometry_msgs::msg::Point32;
    using Image = sensor_msgs::msg::Image;
    using PointCloud = sensor_msgs::msg::PointCloud;
    using ChannelFloat32 = sensor_msgs::msg::ChannelFloat32;

    const YAML::Node& cfg_;
    std::shared_ptr<rclcpp::Publisher<Image>> pub_color_, pub_disparityL_, pub_disparityR_, pub_depthL_, pub_depthR_;
    std::shared_ptr<rclcpp::Publisher<PointCloud>> pub_local_point_cloud_;

public:
    Publishers(rclcpp::Node* node, const YAML::Node& cfg);

    void PublishColor(const cv::Mat& img, const std_msgs::msg::Header& header);
    void PublishDisparityL(const cv::Mat& img, const std_msgs::msg::Header& header);
    void PublishDisparityR(const cv::Mat& img, const std_msgs::msg::Header& header);
    void PublishDepthL(const cv::Mat& img, const std_msgs::msg::Header& header);
    void PublishDepthR(const cv::Mat& img, const std_msgs::msg::Header& header);
    void PublishLocalPointCloud(const Eigen::Matrix3Xf& cloud, const std_msgs::msg::Header& header);
};

}  // namespace tools

#endif  // Publishers_HPP
