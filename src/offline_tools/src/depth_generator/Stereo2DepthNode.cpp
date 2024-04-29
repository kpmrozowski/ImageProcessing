#include "depth_generator/Stereo2DepthNode.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>

#include "depth_generator/DepthGeneration.hpp"

namespace tools {

using namespace std::chrono_literals;

Stereo2DepthNode::Stereo2DepthNode(int argc, char** argv) : rclcpp::Node("Stereo2DepthNode") {
    depth_generator_ = std::make_unique<DepthGeneration>(argc, argv);

    const rclcpp::QoS qos(10);
    const auto rmw_qos_profile = qos.get_rmw_qos_profile();
    sub_rgb_.subscribe(this, "/oakd/rgb/image_raw", rmw_qos_profile);
    sub_ir1_.subscribe(this, "/oakd/left/image_raw", rmw_qos_profile);
    sub_ir2_.subscribe(this, "/oakd/right/image_raw", rmw_qos_profile);

    // Uncomment this to verify that the messages indeed reach the
    sub_rgb_.registerCallback(std::bind(&Stereo2DepthNode::CallbackRgb, this, std::placeholders::_1));
    sub_ir1_.registerCallback(std::bind(&Stereo2DepthNode::CallbackIr1, this, std::placeholders::_1));
    sub_ir2_.registerCallback(std::bind(&Stereo2DepthNode::CallbackIr2, this, std::placeholders::_1));

    Policy policy(500);
    policy.setMaxIntervalDuration(600us);
    images_sync_ =
        std::make_unique<message_filters::Synchronizer<Policy>>(std::move(policy), sub_rgb_, sub_ir1_, sub_ir2_);
    images_sync_->registerCallback(std::bind(&Stereo2DepthNode::CallbackSyncedImages, this, std::placeholders::_1,
                                             std::placeholders::_2, std::placeholders::_3));
}

void Stereo2DepthNode::CallbackRgb(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    RCLCPP_INFO(this->get_logger(), "Rgb frame '%s', size (%d, %d), stamp %lf", msg->header.frame_id.c_str(),
                msg->height, msg->width, rclcpp::Time(msg->header.stamp).seconds());
}
void Stereo2DepthNode::CallbackIr1(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    RCLCPP_INFO(this->get_logger(), "Ir1: frame '%s', size (%d, %d), stamp %lf", msg->header.frame_id.c_str(),
                msg->height, msg->width, rclcpp::Time(msg->header.stamp).seconds());
}
void Stereo2DepthNode::CallbackIr2(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    RCLCPP_INFO(this->get_logger(), "Ir2 frame '%s', size (%d, %d), stamp %lf", msg->header.frame_id.c_str(),
                msg->height, msg->width, rclcpp::Time(msg->header.stamp).seconds());
}
void Stereo2DepthNode::CallbackSyncedImages(const sensor_msgs::msg::Image::ConstSharedPtr& msg_rgb,
                                            const sensor_msgs::msg::Image::ConstSharedPtr& msg_ir1,
                                            const sensor_msgs::msg::Image::ConstSharedPtr& msg_ir2) {
    RCLCPP_INFO(this->get_logger(),
                "Sync frames:"
                "\n\tRgb frame '%s', size (%d, %d), stamp %lf;"
                "\n\tIr1: frame '%s', size (%d, %d), stamp %lf;"
                "\n\tIr2: frame '%s', size (%d, %d), stamp %lf;",
                msg_rgb->header.frame_id.c_str(), msg_rgb->height, msg_rgb->width,
                rclcpp::Time(msg_rgb->header.stamp).seconds(), msg_ir1->header.frame_id.c_str(), msg_ir1->height,
                msg_ir1->width, rclcpp::Time(msg_ir1->header.stamp).seconds(), msg_ir2->header.frame_id.c_str(),
                msg_ir2->height, msg_ir2->width, rclcpp::Time(msg_ir2->header.stamp).seconds());
}

}  // namespace tools