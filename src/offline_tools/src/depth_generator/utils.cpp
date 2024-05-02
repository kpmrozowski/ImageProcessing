#include "depth_generator/utils.hpp"

#include <exception>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core/check.hpp>
#include <sstream>

#include <fmt/format.h>
#include <libsgm.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "depth_generator/constants.hpp"
#include "depth_generator/filesystem.hpp"

namespace {

std::string parentFolderName(std::string path) {
    const size_t last_pos = path.find_last_of("/");
    const std::string parent_path = path.substr(0, last_pos);
    const size_t prelast_pos = parent_path.find_last_of("/");
    const std::string parent_folder = parent_path.substr(prelast_pos + 1, last_pos - prelast_pos);

    return parent_folder;
}

std::string prepareFilePath(const std::string& stereo_imag_path, const std::string output_folder_path,
                            std::string file_format) {
    static constexpr size_t stereo_str_size = 7;
    size_t folder_path_len = stereo_imag_path.find_last_of("/");
    if (std::string::npos == folder_path_len) {
        throw std::runtime_error(fmt::format("no '/' in {}", stereo_imag_path));
    }
    std::string file_name =
        stereo_imag_path.substr(folder_path_len, stereo_imag_path.size() - folder_path_len -
                                                     tools::kStereoFileFormat.size() - stereo_str_size) +
        "-" + parentFolderName(output_folder_path) + file_format;

    return output_folder_path + file_name;
}

}  // namespace

namespace tools {

void utils::removeDirectory(const std::string& path) {
    if (fs::exists(path)) {
        fs::remove_all(path);
    } else {
        fmt::print("directory '{}' does not exist\n", path);
    }
}

void utils::createDirectory(const std::string& path) {
    if (not fs::exists(path)) {
        fs::create_directories(path);
    } else {
        fmt::print("directory '{}' exists\n", path);
    }
}
void utils::createDirectories(std::vector<std::string>&& paths) {
    for (const std::string& path : paths) {
        createDirectory(path);
    }
}

nlohmann::json utils::readJson(const std::string& path) {
    int num_tries = 0;
    if (not fs::exists(path)) {
        spdlog::error("path does not exist: '{}'", path);
        exit(1);
    }
    if (fs::is_directory(path)) {
        spdlog::error("path is not a file: '{}'", path);
        exit(1);
    }
    while (num_tries < 20) {
        try {
            spdlog::info("Reading json: '{}'", path);
            std::ifstream file;
            file.open(path);
            if (file.fail()) {
                spdlog::error("Failed to open (read) '{}'", path);
                std::exit(EXIT_FAILURE);
            }
            nlohmann::json data = nlohmann::json::parse(file);
            spdlog::info("Successfully parsed json.");
            return data;
        } catch (...) {
            ++num_tries;
        }
    }
    spdlog::error("error reading json: '{}'", path);
    exit(1);
}

// =============================================================
// ========================= FUNCTIONS =========================

sgm::StereoSGM& utils::CreateStereoSGM(const YAML::Node& cfg) {
    sgm::PathType pathType = sgm::PathType::SCAN_4PATH;
    switch (cfg["SGM"]["numPaths"].as<int>()) {
        case 4:
            pathType = sgm::PathType::SCAN_4PATH;
            break;
        case 8:
            pathType = sgm::PathType::SCAN_8PATH;
            break;
        default:
            spdlog::error("numPaths can be equal 4 or 8");
            exit(1);
    }

    sgm::CensusType censusType = sgm::CensusType::CENSUS_9x7;
    switch (cfg["SGM"]["censusTypeId"].as<int>()) {
        case 0:
            censusType = sgm::CensusType::CENSUS_9x7;
            break;
        case 1:
            censusType = sgm::CensusType::SYMMETRIC_CENSUS_9x7;
            break;
        default: {
            spdlog::error("censusType can be equal 0 or 1");
            exit(1);
        }
    }

    static constexpr int kStereoBits = 8;
    static constexpr int kDisparityBits = 16;

    // clang-format off
    spdlog::info(
      "sgm::StereoSGM:"
      "\n\twidth: {}"
      "\n\theight: {}"
      "\n\tdisparity_size: {}"
      "\n\tinput_depth_bits: {}"
      "\n\toutput_depth_bits: {}"
      "\n\tinout_type: {}"
      "\n\t\tparameters.P1: {}"
      "\n\t\tparameters.P2: {}"
      "\n\t\tparameters.uniqueness: {}"
      "\n\t\tparameters.subpixel: {}"
      "\n\t\tparameters.path_type: {}"
      "\n\t\tparameters.min_disp: {}"
      "\n\t\tparameters.LR_max_diff: {}"
      "\n\t\tparameters.census_type: {}",
      cfg["SGM"]["imgWidth"].as<int>(),
      cfg["SGM"]["imgHeight"].as<int>(),
      cfg["SGM"]["disparitySize"].as<int>(),
      kStereoBits,
      kDisparityBits,
      sgm::EXECUTE_INOUT_HOST2HOST,
      // parameters
      cfg["SGM"]["p1"].as<int>(),
      cfg["SGM"]["p2"].as<int>(),
      cfg["SGM"]["uniqueness"].as<float>(),
      cfg["SGM"]["subpixel"].as<bool>(),
      pathType,
      cfg["SGM"]["minimalDisparity"].as<int>(),
      cfg["SGM"]["lrMaxDiff"].as<int>(),
      censusType
      );

    static sgm::StereoSGM ssgm{
        cfg["SGM"]["imgWidth"].as<int>(),
        cfg["SGM"]["imgHeight"].as<int>(),
        cfg["SGM"]["disparitySize"].as<int>(),
        kStereoBits,
        kDisparityBits,
        sgm::EXECUTE_INOUT_HOST2HOST,
        {
            cfg["SGM"]["p1"].as<int>(),
            cfg["SGM"]["p2"].as<int>(),
            cfg["SGM"]["uniqueness"].as<float>(),
            cfg["SGM"]["subpixel"].as<bool>(),
            pathType,
            cfg["SGM"]["minimalDisparity"].as<int>(),
            cfg["SGM"]["lrMaxDiff"].as<int>(),
            censusType
        }
    };
    return ssgm;
    // clang-format on
}

std::pair<cv::Mat, cv::Mat> utils::ComputeDisparity(sgm::StereoSGM& ssgm, const cv::Mat& stereoL_raw,
                                                    const cv::Mat& stereoR_raw) {
    static constexpr int kInvalidDisparityValue = 65520;

    cv::Mat stereoL_rect{stereoL_raw.size(), CV_8U}, stereoR_rect{stereoR_raw.size(), CV_8U};
    stereoL_raw.convertTo(stereoL_rect, CV_8U);
    stereoR_raw.convertTo(stereoR_rect, CV_8U);

    cv::Mat disparityL(stereoL_rect.size(), CV_16U), disparityR(stereoR_rect.size(), CV_16U);
    try {
        ssgm.execute(stereoL_rect.data, stereoR_rect.data, disparityL.data, disparityR.data);
    } catch (const cv::Exception& e) {
        if (e.code == cv::Error::GpuNotSupported) {
            spdlog::error("generate_depth, libSGM: GpuNotSupported");
            exit(1);
        } else {
            spdlog::error("generate_depth: {}", e.what());
            exit(2);
        }
    }
    // create mask for invalid disp
    const cv::Mat mask1L = disparityL == ssgm.get_invalid_disparity();
    const cv::Mat mask1R = disparityR == ssgm.get_invalid_disparity();
    const cv::Mat mask2L = disparityL == kInvalidDisparityValue;
    const cv::Mat mask2R = disparityR == kInvalidDisparityValue;
    disparityL.setTo(0, mask1L);
    disparityR.setTo(0, mask1R);
    disparityL.setTo(0, mask2L);
    disparityR.setTo(0, mask2R);
    return {disparityL, disparityR};
}

void utils::saveImage(cv::InputArray image, const std::string& stereo_imag_path, const std::string output_folder_path,
                      std::string file_format) {
    std::string file_path = prepareFilePath(stereo_imag_path, output_folder_path, file_format);
    cv::imwrite(file_path, image);
    // fmt::print("'{}' saved\n", file_path);
}

void utils::saveDepth(cv::InputArray pc_to_save, const std::string& stereo_imag_path,
                      const std::string output_folder_path, std::string file_format) {
    cv::Mat pc_mat = pc_to_save.getMat();
    cv::Mat depth(pc_mat.rows, pc_mat.cols, CV_64F);
    std::vector<cv::Mat> channels_xyz(3);

    cv::split(pc_mat, channels_xyz);

    channels_xyz[2].setTo(0, channels_xyz[2] == kInvalidDepth);

    std::string file_path = prepareFilePath(stereo_imag_path, output_folder_path, file_format);
    cv::imwrite(file_path, channels_xyz[2]);
    // fmt::print("'{}' saved\n", file_path);
}

void utils::saveColorPointCloud(cv::InputArray pc_to_save, cv::InputArray image, const std::string& stereo_imag_path,
                                const std::string output_folder_path, std::string file_format) {
    cv::Mat pc_mat = pc_to_save.getMat();
    cv::Mat image_mat = image.getMat();

    size_t pc_size = 0;
    for (int row = 0; row < pc_mat.rows; ++row) {
        for (int col = 0; col < pc_mat.cols; ++col) {
            if (std::isinf(pc_mat.at<cv::Vec3f>(row, col)[0]) or std::isinf(pc_mat.at<cv::Vec3f>(row, col)[1]) or
                std::isinf(pc_mat.at<cv::Vec3f>(row, col)[2]) or pc_mat.at<cv::Vec3f>(row, col)[2] == kInvalidDepth) {
                continue;
            }
            ++pc_size;
        }
    }

    std::string file_path = prepareFilePath(stereo_imag_path, output_folder_path, file_format);
    std::ofstream outfile(file_path);
    outfile << "ply\n"
            << "format ascii 1.0\n"
            << "comment VTK generated PLY File\n";
    outfile << "element vertex " << pc_size << "\n";
    outfile << "property float x\n"
            << "property float y\n"
            << "property float z\n";
    outfile << "property uchar red\n"
            << "property uchar green\n"
            << "property uchar blue\n"
            << "property uchar alpha\n";
    outfile << "element face 0\n";
    outfile << "property list uchar int vertex_indices\n"
            << "end_header";

    for (int row = 0; row < pc_mat.rows; ++row) {
        for (int col = 0; col < pc_mat.cols; ++col) {
            if (std::isinf(pc_mat.at<cv::Vec3f>(row, col)[0]) or std::isinf(pc_mat.at<cv::Vec3f>(row, col)[1]) or
                std::isinf(pc_mat.at<cv::Vec3f>(row, col)[2]) or pc_mat.at<cv::Vec3f>(row, col)[2] == kInvalidDepth) {
                continue;
            }
            outfile << "\n";
            outfile << -pc_mat.at<cv::Vec3f>(row, col)[0] << " ";
            outfile << -pc_mat.at<cv::Vec3f>(row, col)[1] << " ";
            outfile << -pc_mat.at<cv::Vec3f>(row, col)[2] << " ";
            outfile << static_cast<int>(image_mat.at<cv::Vec3b>(row, col)[0]) << " ";
            outfile << static_cast<int>(image_mat.at<cv::Vec3b>(row, col)[1]) << " ";
            outfile << static_cast<int>(image_mat.at<cv::Vec3b>(row, col)[2]) << " 255";
        }
    }

    outfile.close();
    // fmt::print("'{}' saved\n", file_path);
}

void utils::savePointCloud(cv::InputArray pc_to_save, const std::string& stereo_imag_path,
                           const std::string output_folder_path, std::string file_format) {
    cv::Mat pc_mat = pc_to_save.getMat();

    size_t pc_size = 0;
    for (int row = 0; row < pc_mat.rows; ++row) {
        for (int col = 0; col < pc_mat.cols; ++col) {
            if (std::isinf(pc_mat.at<cv::Vec3f>(row, col)[0]) or std::isinf(pc_mat.at<cv::Vec3f>(row, col)[1]) or
                std::isinf(pc_mat.at<cv::Vec3f>(row, col)[2]) or pc_mat.at<cv::Vec3f>(row, col)[2] == kInvalidDepth) {
                continue;
            }
            ++pc_size;
        }
    }

    std::string file_path = prepareFilePath(stereo_imag_path, output_folder_path, file_format);
    std::ofstream outfile(file_path);
    outfile << "ply\n"
            << "format ascii 1.0\n"
            << "comment VTK generated PLY File\n";
    outfile << "element vertex " << pc_size << "\n";
    outfile << "property float x\n"
            << "property float y\n"
            << "property float z\n";
    outfile << "element face 0\n";
    outfile << "property list uchar int vertex_indices\n"
            << "end_header";

    for (int row = 0; row < pc_mat.rows; ++row) {
        for (int col = 0; col < pc_mat.cols; ++col) {
            if (std::isinf(pc_mat.at<cv::Vec3f>(row, col)[0]) or std::isinf(pc_mat.at<cv::Vec3f>(row, col)[1]) or
                std::isinf(pc_mat.at<cv::Vec3f>(row, col)[2]) or pc_mat.at<cv::Vec3f>(row, col)[2] == kInvalidDepth) {
                continue;
            }
            outfile << "\n";
            outfile << -pc_mat.at<cv::Vec3f>(row, col)[0] << " ";
            outfile << -pc_mat.at<cv::Vec3f>(row, col)[1] << " ";
            outfile << -pc_mat.at<cv::Vec3f>(row, col)[2];
        }
    }

    outfile.close();
    // fmt::print("'{}' saved\n", file_path);
}

void utils::multiplyChannelWise(cv::Mat& img, std::vector<int> channel_ids, double value) {
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    for (int id : channel_ids) {
        channels[id] *= value;
    }
    cv::merge(channels, img);
}

void utils::removeBoundaryPoints(cv::Mat& img, const int padding) {
    if (padding == 0) {
        return;
    }
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    const int cols = channels.front().cols;
    const int rows = channels.front().rows;
    const cv::Rect roiN(0, 0, cols, padding);
    const cv::Rect roiE(cols - padding, padding, padding, rows - 2 * padding);
    const cv::Rect roiS(0, rows - padding, cols, padding);
    const cv::Rect roiW(0, padding, padding, rows - 2 * padding);
    channels[2](roiN) = 0.;
    channels[2](roiE) = 0.;
    channels[2](roiS) = 0.;
    channels[2](roiW) = 0.;
    cv::merge(channels, img);
}

Eigen::Matrix3Xf utils::convertPcMatToEigen(const cv::Mat& pcMat) {
    const size_t pcSize = std::count_if(
        std::execution::par, pcMat.begin<cv::Vec3f>(), pcMat.end<cv::Vec3f>(), [](const cv::Vec3f& point) -> bool {
            if (std::isinf(point[0]) or std::isinf(point[1]) or std::isinf(point[2]) or point[2] <= 0.) {
                return false;
            }
            return true;
        });

    Eigen::Matrix3Xf cloud = Eigen::Matrix3Xf::Zero(3, pcSize);

    size_t pointId = 0;
    for (int row = 0; row < pcMat.rows; ++row) {
        for (int col = 0; col < pcMat.cols; ++col) {
            if (std::isinf(pcMat.at<cv::Vec3f>(row, col)[0]) or std::isinf(pcMat.at<cv::Vec3f>(row, col)[1]) or
                std::isinf(pcMat.at<cv::Vec3f>(row, col)[2]) or pcMat.at<cv::Vec3f>(row, col)[2] <= 0.) {
                continue;
            }
            const cv::Vec3f& cvPoint = pcMat.at<cv::Vec3f>(row, col);
            cloud.col(pointId) = Eigen::Vector3f{cvPoint[2], cvPoint[0], cvPoint[1]};
            ++pointId;
        }
    }
    return cloud;
}

void utils::saveLaserScan(cv::InputArray pc_to_save, const std::string& stereo_imag_path,
                          const std::string output_folder_path, std::string file_format) {
    cv::Mat pc_mat = pc_to_save.getMat();

    cv::Mat laser_scan_img = cv::Mat::zeros(pc_mat.rows, pc_mat.cols, CV_8UC3);
    cv::line(laser_scan_img, cv::Point2i(0, pc_mat.rows / 2), cv::Point2i(pc_mat.cols, pc_mat.rows / 2),
             cv::Scalar(255, 255, 255), 1);
    cv::line(laser_scan_img, cv::Point2i(pc_mat.cols / 2, 0), cv::Point2i(pc_mat.cols / 2, pc_mat.rows),
             cv::Scalar(255, 255, 255), 1);

    float x_max = 20000;
    float x_min = -20000;
    float y_max = 500;
    float y_min = -500;
    float z_max = x_max * static_cast<float>(pc_mat.rows) / static_cast<float>(pc_mat.cols);
    float z_min = x_min * static_cast<float>(pc_mat.rows) / static_cast<float>(pc_mat.cols);

    for (int row = 0; row < pc_mat.rows; ++row) {
        for (int col = 0; col < pc_mat.cols; ++col) {
            if (std::isinf(pc_mat.at<cv::Vec3f>(row, col)[0]) or std::isinf(pc_mat.at<cv::Vec3f>(row, col)[1]) or
                std::isinf(pc_mat.at<cv::Vec3f>(row, col)[2]) or pc_mat.at<cv::Vec3f>(row, col)[2] == kInvalidDepth) {
                continue;
            }
            float x = -pc_mat.at<cv::Vec3f>(row, col)[0];
            float y = -pc_mat.at<cv::Vec3f>(row, col)[1];
            float z = -pc_mat.at<cv::Vec3f>(row, col)[2];

            if (x > x_max || x < x_min || y > y_max || y < y_min || z > z_max || z < z_min) {
                continue;
            }
            int u = static_cast<int>((x - x_min) / (x_max - x_min) * static_cast<float>(pc_mat.cols));
            int v = static_cast<int>((z - z_min) / (z_max - z_min) * static_cast<float>(pc_mat.rows));
            if (v < 0 || v >= pc_mat.rows || u < 0 || u >= pc_mat.cols) {
                continue;
            }
            cv::circle(laser_scan_img, cv::Point2i(u, v), 2, cv::Scalar(0, 255, 0));

            // laser_scan_img.at<unsigned char>(u+1, v+1) = 255;
            // laser_scan_img.at<unsigned char>(u+1, v) = 255;
            // laser_scan_img.at<unsigned char>(u+1, v-1) = 255;
            // laser_scan_img.at<unsigned char>(u, v+1) = 255;
            // laser_scan_img.at<unsigned char>(u, v) = 255;
            // laser_scan_img.at<unsigned char>(u, v-1) = 255;
            // laser_scan_img.at<unsigned char>(u-1, v+1) = 255;
            // laser_scan_img.at<unsigned char>(u-1, v) = 255;
            // laser_scan_img.at<unsigned char>(u-1, v-1) = 255;
        };
    }
    std::string file_path = prepareFilePath(stereo_imag_path, output_folder_path, file_format);
    cv::imwrite(file_path, laser_scan_img);

    // fmt::print("'{}' saved\n", file_path);
}

void utils::DetermineResultsPath(std::string& results_path) {
    int results_path_id = 0;
    while (fs::is_directory(results_path + fmt::format("-v{:02}/", results_path_id))) {
        ++results_path_id;
    }
    results_path += fmt::format("-v{:02}/", results_path_id);
}

#if WITH_ORB_SLAM_3 == 1
static void utils::saveTrackedPoints(const std::vector<ORB_SLAM3::MapPoint*>& pc_to_save,
                                     const std::string output_folder_path, const Settings& s) {
    std::map<unsigned long, std::vector<Eigen::Vector3f>> maps;

    for (ORB_SLAM3::MapPoint* point : pc_to_save) {
        if (point == nullptr) {
            continue;
        }
        if (point->isBad()) {
            continue;
        }
        if (point->GetMap()->IsBad()) {
            continue;
        }
        maps[point->GetMap()->GetId()].push_back(point->GetWorldPos());
    }

    for (const auto& [mapId, map_points] : maps) {
        std::string file_path = output_folder_path + "/" + s.output_tracked_points_file_name + "-mapId_" +
                                std::to_string(mapId) + s.pc_file_format;
        std::ofstream outfile(file_path);
        outfile << "ply\n"
                << "format ascii 1.0\n"
                << "comment VTK generated PLY File\n";
        outfile << "element vertex " << map_points.size() << "\n";
        outfile << "property float x\n"
                << "property float y\n"
                << "property float z\n";
        outfile << "element face 0\n";
        outfile << "property list uchar int vertex_indices\n"
                << "end_header";
        for (const Eigen::Vector3f& point : map_points) {
            outfile << "\n";
            outfile << point(0) << " ";
            outfile << point(1) << " ";
            outfile << point(2);
        }
        fmt::print("'{}' saved\n", file_path);
    }
}

static void utils::saveMapPointsAndTrajectories(const OrbSlam3Wrapper& slam, const std::string output_folder_path,
                                                const Settings& s) {
    std::vector<ORB_SLAM3::Map*> maps_vec = slam.GetAllMaps();

    for (ORB_SLAM3::Map* p_map : maps_vec) {
        std::string pc_ply_path = output_folder_path + "/" + s.output_map_points_file_name + "-mapId_" +
                                  std::to_string(p_map->GetId()) + s.pc_file_format;
        std::string pc_csv_path = output_folder_path + "/" + s.output_map_points_file_name + "-mapId_" +
                                  std::to_string(p_map->GetId()) + ".csv";
        std::string trajectory_path = output_folder_path + "/" + s.output_trajectory_file_name + "-mapId_" +
                                      std::to_string(p_map->GetId()) + ".csv";
        std::string kp_trajectory_path = output_folder_path + "/" + s.output_kp_trajectory_file_name + "-mapId_" +
                                         std::to_string(p_map->GetId()) + ".csv";

        /// TODO: decide if remove bads or not!
        if (p_map->IsBad()) {
            utils::removeDirectory(pc_ply_path);
            utils::removeDirectory(pc_csv_path);
            utils::removeDirectory(trajectory_path);
            utils::removeDirectory(kp_trajectory_path);
            continue;
        }

        // ######################################################
        // ################ save ply point cloud ################

        std::vector<ORB_SLAM3::MapPoint*> map_points = p_map->GetAllMapPoints();
        size_t num_points = 0;
        for (ORB_SLAM3::MapPoint* map_point : map_points) {
            if (map_point->isBad()) {
                continue;
            }
            ++num_points;
        }

        std::ofstream pc_ply_file(pc_ply_path);
        pc_ply_file << "ply\n"
                    << "format ascii 1.0\n"
                    << "comment VTK generated PLY File\n";
        pc_ply_file << "element vertex " << num_points << "\n";
        pc_ply_file << "property float x\n"
                    << "property float y\n"
                    << "property float z\n";
        pc_ply_file << "element face 0\n";
        pc_ply_file << "property list uchar int vertex_indices\n"
                    << "end_header";
        for (ORB_SLAM3::MapPoint* map_point : map_points) {
            if (map_point->isBad()) {
                continue;
            }
            const Eigen::Vector3f& point = map_point->GetWorldPos();
            pc_ply_file << "\n";
            pc_ply_file << point(0) << " ";
            pc_ply_file << point(1) << " ";
            pc_ply_file << point(2);
        }
        pc_ply_file.close();
        fmt::print("'{}' saved\n", pc_ply_path);

        // ######################################################
        // ################ save csv point cloud ################

        std::ofstream pc_csv_file(pc_csv_path);
        pc_csv_file << "x,y,z,observations\n";
        for (ORB_SLAM3::MapPoint* map_point : map_points) {
            if (map_point->isBad()) {
                continue;
            }
            const Eigen::Vector3f& point = map_point->GetWorldPos();
            pc_csv_file << point(0) << "," << point(1) << "," << point(2) << "," << map_point->GetObservations().size()
                        << "\n";
        }
        pc_csv_file.close();
        fmt::print("'{}' saved\n", pc_csv_path);

        // #################################################
        // ################ save trajectory ################

        slam.SaveTrajectoryEuRoC(trajectory_path, p_map);

        // ###########################################################
        // ################ save key-point trajectory ################

        slam.SaveKeyFrameTrajectoryEuRoC(kp_trajectory_path, p_map);
    }
}
#endif

static void getMonoCalibration(MonoCalibration& out, const CamSysCalib& calib) {
    if constexpr (kFitLeftStereoToRgb) {
        out.intrinsics = calib.stereoL_rect.intrinsics;
        out.distortion = calib.stereoL_rect.distortion;
        out.rotation = calib.stereoL_rect.rotation;
        out.translation = calib.stereoL_rect.translation;
        // gray_calib.translation = cv::Mat_<double>{7.5456632900812878e-02, 0, 0};
    } else {
        out.intrinsics = calib.stereoR_rect.intrinsics;
        out.distortion = calib.stereoR_rect.distortion;
        out.rotation = calib.stereoR_rect.rotation;
        out.translation = calib.stereoR_rect.translation;
        // gray_calib.translation = cv::Mat_<double>{-2.4666640199322742e-02, 0, 0};
    }
    out.translation = -out.translation;
    out.translation.at<double>(1, 0) = 0;
    out.translation.at<double>(2, 0) = 0;
}

static void getOptimizedCameraMatrix(cv::InputOutputArray out, const MonoCalibration& gray_calib,
                                     const cv::Size& image_size, const cv::Size& stereo_resol) {
    cv::Mat mat;
    if constexpr (kFitImagesToStereo) {
        cv::Rect gray_valid_pix_roi_2;
        if (gray_calib.intrinsics.empty() or gray_calib.distortion.empty()) {
            spdlog::info("empty");
        }
        spdlog::info("getOptimalNewCameraMatrix");
        std::cout << "gray_calib.intrinsics:" << gray_calib.intrinsics << "\n"
                  << "gray_calib.distortion: " << gray_calib.distortion << "\n"
                  << "stereo_resol: " << stereo_resol << "\n"
                  << "image_size: " << image_size << "\n";
        mat = cv::getOptimalNewCameraMatrix(gray_calib.intrinsics, gray_calib.distortion, stereo_resol, 1.0, image_size,
                                            &gray_valid_pix_roi_2);
    } else {
        cv::Rect rgb_valid_pix_roi_2;
        if (gray_calib.intrinsics.empty() or gray_calib.distortion.empty()) {
            spdlog::info("empty");
        }
        mat = cv::getOptimalNewCameraMatrix(gray_calib.intrinsics, gray_calib.distortion, stereo_resol, 1.0, image_size,
                                            &rgb_valid_pix_roi_2);
    }
    spdlog::info("after getOptimalNewCameraMatrix");
    out.assign(mat);
    spdlog::info("after assign");
}

struct StereoRectifyOut {
    cv::Mat rgb_rectification;   //(3, 3, CV_64F);
    cv::Mat gray_rectification;  //(3, 3, CV_64F);
    cv::Mat rgb_projectionMat;   //(3, 4, CV_64F);
    cv::Mat gray_projectionMat;  //(3, 4, CV_64F);
    cv::Mat mapping;             //(4, 4, CV_64F);
    cv::Rect rgb_valid_pix_roi_1;
    cv::Rect gray_valid_pix_roi_1;
    StereoRectifyOut()
            : rgb_rectification(3, 3, CV_64F)
            , gray_rectification(3, 3, CV_64F)
            , gray_projectionMat(3, 4, CV_64F)
            , mapping(4, 5, CV_64F)
            , rgb_valid_pix_roi_1()
            , gray_valid_pix_roi_1() {
    }
};

static void performStereoRectify(StereoRectifyOut& sr, const CamSysCalib& calib, const MonoCalibration& gray_calib,
                                 const cv::Size& image_size, const cv::Size& stereo_resol) {
    std::cout << "calib.rgb.intrinsics:\n"
              << calib.rgb.intrinsics << "\n"
              << "calib.rgb.distortion:\n"
              << calib.rgb.distortion << "\n"
              << "gray_calib.intrinsics:\n"
              << gray_calib.intrinsics << "\n"
              << "gray_calib.distortion:\n"
              << gray_calib.distortion << "\n"
              << "image_size:\n"
              << image_size << "\n"
              << "gray_calib.rotation:\n"
              << gray_calib.rotation << "\n"
              << "gray_calib.translation:\n"
              << gray_calib.translation << "\n"
              << "s.stereo_resol:\n"
              << stereo_resol << "\n";
    spdlog::info("stereoRectify start");

    cv::stereoRectify(gray_calib.intrinsics,     // Input
                      gray_calib.distortion,     // Input
                      calib.rgb.intrinsics,      // Input
                      calib.rgb.distortion,      // Input
                      image_size,                // Input
                      gray_calib.rotation,       // Input
                      gray_calib.translation,    // Input
                      sr.gray_rectification,     // Output
                      sr.rgb_rectification,      // Output
                      sr.gray_projectionMat,     // Output
                      sr.rgb_projectionMat,      // Output
                      sr.mapping,                // Output
                      cv::CALIB_ZERO_DISPARITY,  // Input
                      1.0,                       // Input
                      stereo_resol,              // Input
                      &sr.gray_valid_pix_roi_1,  // Output
                      &sr.rgb_valid_pix_roi_1    // Output
    );
    sr.mapping.at<double>(4, 3) *= -1;
    sr.mapping.at<double>(4, 4) *= -1;
    sr.mapping.at<double>(3, 4) *= -1;
    std::cout << "gray_calib.intrinsics:\n"
              << gray_calib.intrinsics << "\n"
              << "rgb_rectification:\n"
              << sr.rgb_rectification << "\n"
              << "gray_rectification:\n"
              << sr.gray_rectification << "\n"
              << "rgb_projectionMat:\n"
              << sr.rgb_projectionMat << "\n"
              << "gray_projectionMat:\n"
              << sr.gray_projectionMat << "\n"
              << "mapping:\n"
              << sr.mapping << "\n";
}

static Rectification performInitUndistortRectifyMap(Rectification& out, const CamSysCalib& calib,
                                                    const MonoCalibration& gray_calib, const cv::Mat& new_intrinsics,
                                                    const cv::Size& image_size, const cv::Size& stereo_resol) {
    spdlog::info("performStereoRectify");
    StereoRectifyOut sr;
    performStereoRectify(sr, calib, gray_calib, image_size, stereo_resol);

    std::cout << "calib.rgb.intrinsics:\n"
              << calib.rgb.intrinsics << "\n"
              << "calib.rgb.distortion:\n"
              << calib.rgb.distortion << "\n";
    spdlog::info("================================");
    cv::initUndistortRectifyMap(calib.rgb.intrinsics, calib.rgb.distortion, sr.rgb_rectification, new_intrinsics,
                                image_size, CV_32FC2, out.rgb_map_1, out.rgb_map_2);

    std::cout << "gray_calib.intrinsics:\n"
              << gray_calib.intrinsics << "\n"
              << "gray_calib.distortion:\n"
              << gray_calib.distortion << "\n";
    cv::initUndistortRectifyMap(gray_calib.intrinsics, gray_calib.distortion, sr.gray_rectification, new_intrinsics,
                                image_size, CV_32FC2, out.gray_map_1, out.gray_map_2);

    out.new_intrinsics = new_intrinsics.clone();

    std::cout << "rgb_map_1(" << out.rgb_map_1.size.dims() << ").size=(" << out.rgb_map_1.size[0] << ", "
              << out.rgb_map_1.size[1] << ")\n"
              << "rgb_map_2(" << out.rgb_map_2.size.dims() << ").size=(" << out.rgb_map_2.size[0] << ", "
              << out.rgb_map_2.size[1] << ")\n"
              << "gray_map_1(" << out.gray_map_1.size.dims() << ").size=(" << out.gray_map_1.size[0] << ", "
              << out.gray_map_1.size[1] << ")\n"
              << "gray_map_2(" << out.gray_map_2.size.dims() << ").size=(" << out.gray_map_2.size[0] << ", "
              << out.gray_map_2.size[1] << ")\n";
    return out;
}

void utils::PerformRectification(Rectification& rectification, const CamSysCalib& calib, const YAML::Node& cfg) {
    const cv::Size stereo_resol{cfg["stereoWidth"].as<int>(), cfg["stereoHeight"].as<int>()};
    const cv::Size rgb_resol{cfg["rgbWidth"].as<int>(), cfg["rgbHeight"].as<int>()};

    cv::Size image_size{};
    if constexpr (kFitImagesToStereo) {
        image_size = stereo_resol;
    } else {
        image_size = rgb_resol;
    }
    MonoCalibration gray_calib;
    getMonoCalibration(gray_calib, calib);

    cv::Mat new_intrinsics(3, 3, CV_64F);
    getOptimizedCameraMatrix(new_intrinsics, gray_calib, image_size, stereo_resol);

    performInitUndistortRectifyMap(rectification, calib, gray_calib, new_intrinsics, image_size, stereo_resol);
}

void utils::LoadCalibration(CamSysCalib& calib, const std::string& path) {
    nlohmann::json data = utils::readJson(path);
    calib.rgb.intrinsics = cv::Mat(data["intrinsics_calib_rgb"]["data"].get<std::vector<double>>(), true).reshape(0, 3);
    calib.rgb.distortion = cv::Mat(data["distortion_calib_rgb"]["data"].get<std::vector<double>>(), true);
    calib.rgb.rotation = cv::Mat::eye(3, 3, CV_64F);
    calib.rgb.translation = cv::Mat::zeros(3, 1, CV_64F);

    calib.stereoL_rect.intrinsics =
        cv::Mat(data["intrinsics_calib_ir1"]["data"].get<std::vector<double>>(), true).reshape(0, 3);
    calib.stereoL_rect.distortion = cv::Mat(data["distortion_calib_ir1"]["data"].get<std::vector<double>>(), true);
    calib.stereoL_rect.rotation =
        cv::Mat(data["rotation_calib_rgb_calib_ir1"]["data"].get<std::vector<double>>(), true).reshape(0, 3);
    calib.stereoL_rect.translation =
        cv::Mat(data["translation_calib_rgb_calib_ir1"]["data"].get<std::vector<double>>(), true);

    calib.stereoR_rect.intrinsics =
        cv::Mat(data["intrinsics_calib_ir2"]["data"].get<std::vector<double>>(), true).reshape(0, 3);
    calib.stereoR_rect.distortion = cv::Mat(data["distortion_calib_ir2"]["data"].get<std::vector<double>>(), true);
    calib.stereoR_rect.rotation =
        cv::Mat(data["rotation_calib_rgb_calib_ir2"]["data"].get<std::vector<double>>(), true).reshape(0, 3);
    calib.stereoR_rect.translation =
        cv::Mat(data["translation_calib_rgb_calib_ir2"]["data"].get<std::vector<double>>(), true);

    calib.stereoL_stereoR_transform =
        cv::Mat(data["transformation_calib_ir1_calib_ir2"]["data"].get<std::vector<double>>(), true);
}

cv::Mat utils::ComputeProjectionMatrix(const CamSysCalib& calib) {
    cv::Mat mapping(4, 4, CV_64F);

    const double Tx = calib.stereoL_stereoR_transform.at<double>(0, 3);
    mapping.at<double>(0, 0) = 1;
    mapping.at<double>(1, 1) = 1;
    mapping.at<double>(3, 2) = -1 / Tx;
    mapping.at<double>(0, 3) = -calib.stereoL_rect.intrinsics.at<double>(0, 2);
    mapping.at<double>(1, 3) = -calib.stereoL_rect.intrinsics.at<double>(1, 2);
    mapping.at<double>(2, 3) = -calib.stereoL_rect.intrinsics.at<double>(0, 0);
    mapping.at<double>(3, 3) =
        (calib.stereoL_rect.intrinsics.at<double>(0, 2) - calib.stereoR_rect.intrinsics.at<double>(0, 2)) / Tx;
    std::cout << "my_mapping:\n" << mapping << "\n";
    return mapping;
}

}  // namespace tools
