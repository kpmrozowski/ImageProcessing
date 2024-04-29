#pragma once

#include <yaml-cpp/yaml.h>
#include <opencv2/core/mat.hpp>

#include "depth_generator/types.hpp"

namespace tools {

/**
 * @brief A program to read ros2bag and generate depth
 */
class DepthGeneration {
public:
    DepthGeneration(int argc, char** argv);
    ~DepthGeneration();

    void ParseOptions(int argc, char** argv);
    void Init();
    bool Compute(const cv::Mat& rgb_raw, const cv::Mat& ir1_raw, const cv::Mat& ir2_raw, cv::Mat& out_disparity,
                 cv::Mat& out_depth, cv::Mat& out_cloud);

private:
    YAML::Node cfg_;
    std::string results_path_;
    CamSysCalib calib_;
    Rectification rectification_;
    cv::Mat mapping_;

    // CLI member variables
    std::string bag_dir_path_;
    std::string calibration_path_;
    std::vector<std::string> cameras_names_;
    std::vector<std::string> cameras_topics_;
    std::string config_path_;
    std::string output_directory_;
    bool verbose_;

    std::string Name() const {
        return "DepthGeneration";
    }
    void PreparePaths();
    void LoadConfig();
};

}  // namespace tools
