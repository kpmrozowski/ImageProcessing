#include "depth_generator/Stereo2DepthNode.hpp"

#include <memory>

#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <opencv2/core/check.hpp>
#include <opencv2/highgui.hpp>
#include <rclcpp/node.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/detail/compressed_image__struct.hpp>

#include "cv_bridge_interface/Converter.hpp"
#include "depth_generator/DepthGeneration.hpp"
#include "depth_generator/Publishers.hpp"
#include "depth_generator/filesystem.hpp"
#include "depth_generator/utils.hpp"

namespace tools {

static const std::string kDefaultResultsPath = "/tmp/DepthGenerator/";

using namespace std::chrono_literals;

Stereo2DepthNode::Stereo2DepthNode(int argc, char** argv) : rclcpp::Node(Name()) {
    this->ParseOptions(argc, argv);
    this->PreparePaths();
    this->LoadConfig();

    depth_generator_ = std::make_unique<DepthGeneration>(cfg_);

    const rclcpp::QoS qos(10);
    const auto rmw_qos_profile = qos.get_rmw_qos_profile();
    if (cfg_["compressed"].as<bool>()) {
        spdlog::info("Subscribing to compressed images.");
        sub_compressed_rgb_.subscribe(this, cfg_["topicRgb"].as<std::string>(), rmw_qos_profile);
        sub_compressed_ir1_.subscribe(this, cfg_["topicStereoL"].as<std::string>(), rmw_qos_profile);
        sub_compressed_ir2_.subscribe(this, cfg_["topicStereoR"].as<std::string>(), rmw_qos_profile);
        // Uncomment this to verify that the messages indeed reach the
        // sub_compressed_rgb_.registerCallback(
        //     std::bind(&Stereo2DepthNode::CallbackCompressedRgb, this, std::placeholders::_1));
        // sub_compressed_ir1_.registerCallback(
        //     std::bind(&Stereo2DepthNode::CallbackCompressedIr1, this, std::placeholders::_1));
        // sub_compressed_ir2_.registerCallback(
        //     std::bind(&Stereo2DepthNode::CallbackCompressedIr2, this, std::placeholders::_1));
        PolicyCompressed policy(500);
        policy.setMaxIntervalDuration(34ms);
        compressed_images_sync_ = std::make_unique<message_filters::Synchronizer<PolicyCompressed>>(
            std::move(policy), sub_compressed_rgb_, sub_compressed_ir1_, sub_compressed_ir2_);
        compressed_images_sync_->registerCallback(std::bind(&Stereo2DepthNode::CallbackSyncedCompressedImages, this,
                                                            std::placeholders::_1, std::placeholders::_2,
                                                            std::placeholders::_3));
    } else {
        spdlog::info("Subscribing to not-compressed images.");
        sub_rgb_.subscribe(this, cfg_["topicRgb"].as<std::string>(), rmw_qos_profile);
        sub_ir1_.subscribe(this, cfg_["topicStereoL"].as<std::string>(), rmw_qos_profile);
        sub_ir2_.subscribe(this, cfg_["topicStereoR"].as<std::string>(), rmw_qos_profile);
        // Uncomment this to verify that the messages indeed reach the
        // sub_rgb_.registerCallback(std::bind(&Stereo2DepthNode::CallbackRgb, this, std::placeholders::_1));
        // sub_ir1_.registerCallback(std::bind(&Stereo2DepthNode::CallbackIr1, this, std::placeholders::_1));
        // sub_ir2_.registerCallback(std::bind(&Stereo2DepthNode::CallbackIr2, this, std::placeholders::_1));
        Policy policy(500);
        policy.setMaxIntervalDuration(600us);
        images_sync_ =
            std::make_unique<message_filters::Synchronizer<Policy>>(std::move(policy), sub_rgb_, sub_ir1_, sub_ir2_);
        images_sync_->registerCallback(std::bind(&Stereo2DepthNode::CallbackSyncedImages, this, std::placeholders::_1,
                                                 std::placeholders::_2, std::placeholders::_3));
    }

    // publishers
    publishers_ = std::make_unique<Publishers>(static_cast<rclcpp::Node*>(this), cfg_);

    spdlog::info("Creating output directory: {}", results_path_);
    utils::createDirectory(results_path_);
}

Stereo2DepthNode::~Stereo2DepthNode() {
    if (!fs::is_directory({results_path_})) {
        spdlog::warn("Output directory does not exist: {}", results_path_);
        return;
    }
    if (fs::is_empty({results_path_})) {
        spdlog::info("Output directory is empty. Removing: {}", results_path_);
        utils::removeDirectory(results_path_);
    } else {
        spdlog::info("Output directory is not empty. See it's contents: {}", results_path_);
    }
}

void Stereo2DepthNode::ParseOptions(int argc, char** argv) {
    CLI::App app{Name()};

    app.add_option("--bag-dir", bag_dir_path_, "bag_dir_path_")->check(CLI::ExistingDirectory)->required();
    app.add_option("--params-file", config_path_, "config_path_")->check(CLI::ExistingFile)->required();
    app.add_option("--results-dir", output_directory_, "output_directory_")->check(CLI::ExistingDirectory);

    try {
        app.parse(argc, argv);
    } catch (CLI::ParseError& e) {
        std::exit(app.exit(e));
    }
}

void Stereo2DepthNode::PreparePaths() {
    // determine dataset_path
    fs::path dataset_path{bag_dir_path_ + "/"};
    while (dataset_path.filename().empty()) {
        dataset_path = dataset_path.parent_path();
    }
    dataset_path += "/";

    // determine results_path
    const std::string dataset_name = dataset_path.parent_path().filename();
    if (output_directory_.empty()) {
        results_path_ = fs::path{kDefaultResultsPath} / dataset_name;
    } else {
        results_path_ = fs::path{output_directory_} / dataset_name;
    }
    utils::DetermineResultsPath(results_path_);

    // path configuration
    spdlog::info("dataset_name: {}", dataset_name);
    spdlog::info("dataset_path: {}", dataset_path.string());
    spdlog::info("results_path: {}", results_path_);
    spdlog::info("config_path: {}", config_path_);
}

#define ASSERT_YAML1(x)                                                \
    if (not cfg_[#x]) {                                                \
        spdlog::warn("not cfg[{}]", #x);                               \
        success = false;                                               \
    } else {                                                           \
        fmt::print("\tcfg[{}]: {}\n", #x, cfg_[#x].as<std::string>()); \
    }
#define ASSERT_YAML2(x, y)                                                         \
    if (not cfg_[#x][#y]) {                                                        \
        spdlog::warn("not cfg[{}][{}]", #x, #y);                                   \
        success = false;                                                           \
    } else {                                                                       \
        fmt::print("\tcfg[{}][{}]: {}\n", #x, #y, cfg_[#x][#y].as<std::string>()); \
    }

void Stereo2DepthNode::LoadConfig() {
    spdlog::info("Loading Config");

    cfg_ = YAML::LoadFile(config_path_);
    spdlog::info("Loading {}", config_path_);
    fmt::print("Configuration:\n[\n");
    bool success = true;
    ASSERT_YAML1(calibrationPath)
    ASSERT_YAML1(topicRgb)
    ASSERT_YAML1(topicStereoL)
    ASSERT_YAML1(topicStereoR)
    ASSERT_YAML1(compressed)
    ASSERT_YAML1(rgbWidth)
    ASSERT_YAML1(rgbHeight)
    ASSERT_YAML1(stereoWidth)
    ASSERT_YAML1(stereoHeight)
    ASSERT_YAML1(verbose)
    ASSERT_YAML2(DepthGeneration, computeColor)
    ASSERT_YAML2(DepthGeneration, computeDepth)
    ASSERT_YAML2(DepthGeneration, computeDisparity)
    ASSERT_YAML2(DepthGeneration, computePointCloud)
    ASSERT_YAML2(DepthGeneration, computeSLAM)
    ASSERT_YAML2(DepthGeneration, useBilateralFilter)
    ASSERT_YAML2(DepthGeneration, bilateralRadius)
    ASSERT_YAML2(DepthGeneration, bilateralIters)
    ASSERT_YAML2(DepthGeneration, useWlsFilter)
    ASSERT_YAML2(DepthGeneration, stereoSGMMode)
    ASSERT_YAML2(DepthGeneration, depthPadding)
    ASSERT_YAML2(SGM, imgWidth)
    ASSERT_YAML2(SGM, imgHeight)
    ASSERT_YAML2(SGM, disparitySize)
    ASSERT_YAML2(SGM, p1)
    ASSERT_YAML2(SGM, p2)
    ASSERT_YAML2(SGM, uniqueness)
    ASSERT_YAML2(SGM, subpixel)
    ASSERT_YAML2(SGM, numPaths)
    ASSERT_YAML2(SGM, minimalDisparity)
    ASSERT_YAML2(SGM, lrMaxDiff)
    ASSERT_YAML2(SGM, censusTypeId)
    fmt::print("]\n");
    if (not success) {
        spdlog::error("Error reading the config");
        exit(1);
    }
}
#undef ASSERT_YAML2
#undef ASSERT_YAML1

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

void Stereo2DepthNode::CallbackCompressedRgb(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg) {
    RCLCPP_INFO(this->get_logger(), "Rgb frame '%s', stamp %lf", msg->header.frame_id.c_str(),
                rclcpp::Time(msg->header.stamp).seconds());
}

void Stereo2DepthNode::CallbackCompressedIr1(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg) {
    RCLCPP_INFO(this->get_logger(), "Ir1: frame '%s', stamp %lf", msg->header.frame_id.c_str(),
                rclcpp::Time(msg->header.stamp).seconds());
}

void Stereo2DepthNode::CallbackCompressedIr2(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg) {
    RCLCPP_INFO(this->get_logger(), "Ir2 frame '%s', stamp %lf", msg->header.frame_id.c_str(),
                rclcpp::Time(msg->header.stamp).seconds());
}

void Stereo2DepthNode::CallbackSyncedImages(const sensor_msgs::msg::Image::ConstSharedPtr& msg_rgb,
                                            const sensor_msgs::msg::Image::ConstSharedPtr& msg_ir1,
                                            const sensor_msgs::msg::Image::ConstSharedPtr& msg_ir2) {
    const std::string rgb_enc = msg_rgb->encoding;
    const std::string ir1_enc = msg_ir1->encoding;
    const std::string ir2_enc = msg_ir2->encoding;
    // const double rgb_ts_sec = rclcpp::Time(msg_rgb->header.stamp).seconds();
    // const double ir1_ts_sec = rclcpp::Time(msg_ir1->header.stamp).seconds();
    // const double ir2_ts_sec = rclcpp::Time(msg_ir2->header.stamp).seconds();
    // RCLCPP_INFO(this->get_logger(),
    //             "Sync frames:"
    //             "\n\tRgb frame '%s', size (%d, %d), encoding %s, stamp %lf;"
    //             "\n\tIr1: frame '%s', size (%d, %d), encoding %s, stamp %lf;"
    //             "\n\tIr2: frame '%s', size (%d, %d), encoding %s, stamp %lf;",
    //             msg_rgb->header.frame_id.c_str(), msg_rgb->height, msg_rgb->width, rgb_enc.c_str(), rgb_ts_sec,
    //             msg_ir1->header.frame_id.c_str(), msg_ir1->height, msg_ir1->width, ir1_enc.c_str(), ir1_ts_sec,
    //             msg_ir2->header.frame_id.c_str(), msg_ir2->height, msg_ir2->width, ir2_enc.c_str(), ir2_ts_sec);

    const cv::Mat rgb_raw = Converter::Convert(*msg_rgb, rgb_enc);
    const cv::Mat ir1_raw = Converter::Convert(*msg_ir1, ir1_enc);
    const cv::Mat ir2_raw = Converter::Convert(*msg_ir2, ir2_enc);
    cv::Mat out_disparityL, out_disparityR, out_depthL, out_depthR;
    Eigen::Matrix3Xf out_cloud;
    RCLCPP_INFO(this->get_logger(), "rgb_raw.type: %s, ir1_raw.type: %s, ir2_raw.type: %s",
                cv::typeToString(rgb_raw.type()).c_str(), cv::typeToString(ir1_raw.type()).c_str(),
                cv::typeToString(ir2_raw.type()).c_str());
    depth_generator_->Compute(rgb_raw, ir1_raw, ir2_raw, out_disparityL, out_disparityR, out_depthL, out_depthR,
                              out_cloud);

    if (not out_disparityL.empty()) {
        publishers_->PublishDisparityL(out_disparityL, msg_ir1->header);
    } else {
        spdlog::warn("out_disparityL.empty");
    }
    if (not out_disparityL.empty()) {
        publishers_->PublishDisparityR(out_disparityR, msg_ir2->header);
    } else {
        spdlog::warn("out_disparityR.empty");
    }

    if (not out_depthL.empty()) {
        publishers_->PublishDepthL(out_depthL, msg_ir1->header);
    } else {
        spdlog::warn("out_depthL.empty");
    }
    if (not out_depthR.empty()) {
        publishers_->PublishDepthR(out_depthR, msg_ir2->header);
    } else {
        spdlog::warn("out_depthR.empty");
    }
    if (not(0 == out_cloud.cols())) {
        publishers_->PublishLocalPointCloud(out_cloud, msg_ir2->header);
    } else {
        spdlog::warn("out_cloud.empty");
    }
}

void Stereo2DepthNode::CallbackSyncedCompressedImages(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg_rgb,
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg_ir1,
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg_ir2) {
    const cv::Mat rgb_raw = Converter::Convert(*msg_rgb, sensor_msgs::image_encodings::BGR8);
    const cv::Mat ir1_raw = Converter::Convert(*msg_ir1, sensor_msgs::image_encodings::MONO8);
    const cv::Mat ir2_raw = Converter::Convert(*msg_ir2, sensor_msgs::image_encodings::MONO8);
    cv::Mat out_disparityL, out_disparityR, out_depthL, out_depthR;
    Eigen::Matrix3Xf out_cloud;
    RCLCPP_INFO(this->get_logger(), "rgb_raw.type: %s, ir1_raw.type: %s, ir2_raw.type: %s",
                cv::typeToString(rgb_raw.type()).c_str(), cv::typeToString(ir1_raw.type()).c_str(),
                cv::typeToString(ir2_raw.type()).c_str());
    depth_generator_->Compute(rgb_raw, ir1_raw, ir2_raw, out_disparityL, out_disparityR, out_depthL, out_depthR,
                              out_cloud);

    if (not out_disparityL.empty()) {
        publishers_->PublishDisparityL(out_disparityL, msg_ir1->header);
    } else {
        spdlog::warn("out_disparityL.empty");
    }
    if (not out_disparityL.empty()) {
        publishers_->PublishDisparityR(out_disparityR, msg_ir2->header);
    } else {
        spdlog::warn("out_disparityR.empty");
    }

    if (not out_depthL.empty()) {
        publishers_->PublishDepthL(out_depthL, msg_ir1->header);
    } else {
        spdlog::warn("out_depthL.empty");
    }
    if (not out_depthR.empty()) {
        publishers_->PublishDepthR(out_depthR, msg_ir2->header);
    } else {
        spdlog::warn("out_depthR.empty");
    }
    if (not(0 == out_cloud.cols())) {
        publishers_->PublishLocalPointCloud(out_cloud, msg_ir2->header);
    } else {
        spdlog::warn("out_cloud.empty");
    }
}

}  // namespace tools
