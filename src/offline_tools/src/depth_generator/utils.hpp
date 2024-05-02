#ifndef utils_HPP
#define utils_HPP

#include <vector>

#include <yaml-cpp/node/node.h>
#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>

#include "depth_generator/types.hpp"

namespace sgm {
class StereoSGM;
}

namespace tools {
namespace utils {

void removeDirectory(const std::string& path);

void createDirectory(const std::string& path);

void createDirectories(std::vector<std::string>&& paths);

nlohmann::json readJson(const std::string& path);

sgm::StereoSGM& CreateStereoSGM(const YAML::Node& cfg);

void PerformRectification(Rectification& rectification, const CamSysCalib& calib, const YAML::Node& cfg);

void LoadCalibration(CamSysCalib& calib, const std::string& path);

cv::Mat ComputeProjectionMatrix(const CamSysCalib& calib);

std::pair<cv::Mat, cv::Mat> ComputeDisparity(sgm::StereoSGM& ssgm, const cv::Mat& stereoL_raw,
                                             const cv::Mat& stereoR_raw);

void saveImage(cv::InputArray image, const std::string& stereo_imag_path, const std::string output_folder_path,
               std::string file_format);

void saveDepth(cv::InputArray pc_to_save, const std::string& stereo_imag_path, const std::string output_folder_path,
               std::string file_format);

void saveColorPointCloud(cv::InputArray pc_to_save, cv::InputArray image, const std::string& stereo_imag_path,
                         const std::string output_folder_path, std::string file_format);

void savePointCloud(cv::InputArray pc_to_save, const std::string& stereo_imag_path,
                    const std::string output_folder_path, std::string file_format);

void multiplyChannelWise(cv::Mat& img, std::vector<int> channel_ids, double value);
void removeBoundaryPoints(cv::Mat& img, const int padding);
Eigen::Matrix3Xf convertPcMatToEigen(const cv::Mat& pcMat);

void saveLaserScan(cv::InputArray pc_to_save, const std::string& stereo_imag_path, const std::string output_folder_path,
                   std::string file_format);

void DetermineResultsPath(std::string& results_path);

}  // namespace utils
}  // namespace tools

#endif  // utils_HPP
