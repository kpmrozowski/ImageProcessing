#ifndef Stereo2DepthNode_HPP
#define Stereo2DepthNode_HPP

#include <memory>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "depth_generator/DepthGeneration.hpp"

namespace tools {

class DepthGeneration;

class Stereo2DepthNode : public rclcpp::Node {
public:
    Stereo2DepthNode(int argc, char** argv);

private:
    typedef sensor_msgs::msg::Image Image;
    typedef message_filters::sync_policies::ApproximateTime<Image, Image, Image> Policy;

    void CallbackRgb(const Image::ConstSharedPtr& msg);
    void CallbackIr1(const Image::ConstSharedPtr& msg);
    void CallbackIr2(const Image::ConstSharedPtr& msg);
    void CallbackSyncedImages(const Image::ConstSharedPtr& msg_rgb, const Image::ConstSharedPtr& msg_ir1,
                              const Image::ConstSharedPtr& msg_ir2);

    std::unique_ptr<DepthGeneration> depth_generator_;

    message_filters::Subscriber<Image> sub_rgb_;
    message_filters::Subscriber<Image> sub_ir1_;
    message_filters::Subscriber<Image> sub_ir2_;
    std::unique_ptr<message_filters::Synchronizer<Policy>> images_sync_;
};

}  // namespace tools

#endif
