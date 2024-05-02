#pragma once

#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include <functional>
#include <memory>
#include <opencv2/core/mat.hpp>

#include "depth_generator/types.hpp"

namespace sgm {
class StereoSGM;
}

namespace cv {
namespace ximgproc {
class DisparityWLSFilter;
}

namespace cuda {
class DisparityBilateralFilter;
}  // namespace cuda
}  // namespace cv

namespace tools {

/**
 * @brief A program to read ros2bag and generate depth
 */
class DepthGeneration {
public:
    DepthGeneration(const YAML::Node& cfg);

    void Init();
    bool Compute(const cv::Mat& rgb_raw, const cv::Mat& stereoL_raw, const cv::Mat& stereoR_raw,
                 cv::OutputArray& out_disparityL, cv::OutputArray& out_disparityR, cv::OutputArray& out_depthL,
                 cv::OutputArray& out_depthR, Eigen::Matrix3Xf& out_cloud);

private:
    const YAML::Node& cfg_;
    CamSysCalib calib_;
    Rectification rectification_;
    cv::Mat mapping_;
    std::unique_ptr<std::reference_wrapper<sgm::StereoSGM>> ssgm_;
    cv::Ptr<cv::cuda::DisparityBilateralFilter> bilateral_filter_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter_;
    cv::Size stereo_size_;
    cv::Size sgm_size_;
};

}  // namespace tools
