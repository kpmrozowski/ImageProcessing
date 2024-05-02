#ifndef Stereo2DepthNode_HPP
#define Stereo2DepthNode_HPP

#include <memory>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <yaml-cpp/node/node.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "depth_generator/DepthGeneration.hpp"
#include "depth_generator/Publishers.hpp"

namespace tools {

class DepthGeneration;

class Stereo2DepthNode : public rclcpp::Node {
public:
    Stereo2DepthNode(int argc, char** argv);
    ~Stereo2DepthNode();

    std::string Name() const {
        return "DepthGeneration";
    }

private:
    using Image = sensor_msgs::msg::Image;
    using CompressedImage = sensor_msgs::msg::CompressedImage;
    typedef message_filters::sync_policies::ApproximateTime<Image, Image, Image> Policy;
    typedef message_filters::sync_policies::ApproximateTime<CompressedImage, CompressedImage, CompressedImage>
        PolicyCompressed;

    void ParseOptions(int argc, char** argv);
    void PreparePaths();
    void LoadConfig();

    void CallbackRgb(const Image::ConstSharedPtr& msg);
    void CallbackIr1(const Image::ConstSharedPtr& msg);
    void CallbackIr2(const Image::ConstSharedPtr& msg);
    void CallbackSyncedImages(const Image::ConstSharedPtr& msg_rgb, const Image::ConstSharedPtr& msg_ir1,
                              const Image::ConstSharedPtr& msg_ir2);

    void CallbackCompressedRgb(const CompressedImage::ConstSharedPtr& msg);
    void CallbackCompressedIr1(const CompressedImage::ConstSharedPtr& msg);
    void CallbackCompressedIr2(const CompressedImage::ConstSharedPtr& msg);
    void CallbackSyncedCompressedImages(const CompressedImage::ConstSharedPtr& msg_rgb,
                                        const CompressedImage::ConstSharedPtr& msg_ir1,
                                        const CompressedImage::ConstSharedPtr& msg_ir2);

    // CLI member variables
    std::string bag_dir_path_;
    std::string config_path_;
    std::string output_directory_;

    std::string results_path_;
    YAML::Node cfg_;

    std::unique_ptr<DepthGeneration> depth_generator_;

    message_filters::Subscriber<Image> sub_rgb_;
    message_filters::Subscriber<Image> sub_ir1_;
    message_filters::Subscriber<Image> sub_ir2_;
    message_filters::Subscriber<CompressedImage> sub_compressed_rgb_;
    message_filters::Subscriber<CompressedImage> sub_compressed_ir1_;
    message_filters::Subscriber<CompressedImage> sub_compressed_ir2_;
    std::unique_ptr<message_filters::Synchronizer<Policy>> images_sync_;
    std::unique_ptr<message_filters::Synchronizer<PolicyCompressed>> compressed_images_sync_;

    std::unique_ptr<Publishers> publishers_;
};

}  // namespace tools

#endif
