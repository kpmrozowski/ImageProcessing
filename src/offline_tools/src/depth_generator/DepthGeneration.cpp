#include "depth_generator/DepthGeneration.hpp"

#include <filesystem>

#include <libsgm.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include <CLI/CLI.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

#include "depth_generator/utils.hpp"
#include "opencv2/imgproc.hpp"

namespace tools {

static const std::string kDefaultResultsPath = "/tmp/DepthGenerator/";

DepthGeneration::DepthGeneration(int argc, char** argv) {
    this->ParseOptions(argc, argv);
    this->Init();
}

DepthGeneration::~DepthGeneration() {
    if (!std::filesystem::is_directory({results_path_})) {
        spdlog::warn("Output directory does not exist: {}", results_path_);
        return;
    }
    if (std::filesystem::is_empty({results_path_})) {
        spdlog::info("Output directory is empty. Removing: {}", results_path_);
        utils::removeDirectory(results_path_);
    } else {
        spdlog::info("Output directory is not empty. See it's contents: {}", results_path_);
    }
}

void DepthGeneration::ParseOptions(int argc, char** argv) {
    CLI::App app{Name()};

    app.add_option("--bag-dir", bag_dir_path_, "bag_dir_path_")->check(CLI::ExistingDirectory)->required();
    app.add_option("--calib-file", calibration_path_, "calibration_path_")->check(CLI::ExistingFile)->required();
    app.add_option("--cams", cameras_names_, "cameras name (used in calibration)")->required();
    app.add_option("--topics", cameras_topics_, "cameras topics")->required();
    app.add_option("--params-file", config_path_, "config_path_")->check(CLI::ExistingFile)->required();
    app.add_option("--results-dir", output_directory_, "output_directory_")->check(CLI::ExistingDirectory);
    app.add_flag("--verbose", verbose_, "verbose_");

    try {
        app.parse(argc, argv);
    } catch (CLI::ParseError& e) {
        std::exit(app.exit(e));
    }
}

void DepthGeneration::Init() {
    bool success = false;
    size_t unsuccessful_runs = 0;
    while (not success) {
        spdlog::error("unsuccessful_runs: {}", unsuccessful_runs);
        try {
            PreparePaths();
            LoadConfig();

            spdlog::info("Creating sgm::StereoSGM");
            sgm::StereoSGM ssgm = utils::CreateStereoSGM(cfg_);

            spdlog::info("Creating BilateralFilter");
            auto bilateral_filter =
                cv::cuda::createDisparityBilateralFilter(cfg_["SGM"]["disparitySize"].as<int>(), 5, 2);

            spdlog::info("Creating cv::cuda::StereoSGM");
            cv::Ptr<cv::cuda::StereoSGM> cv_sgm_left = cv::cuda::createStereoSGM(
                0, cfg_["SGM"]["disparitySize"].as<int>(), cfg_["SGM"]["p1"].as<int>(), cfg_["SGM"]["p2"].as<int>(),
                100 * static_cast<int>(1.f - cfg_["SGM"]["uniqueness"].as<float>()), cv::cuda::StereoSGM::MODE_HH);
            // cv_sgm_left->setPreFilterCap(cv::cuda::StereoBM::PREFILTER_XSOBEL);
            auto wls_filter = cv::ximgproc::createDisparityWLSFilter(cv_sgm_left);
            // wls_filter->setLambda(8000);
            // wls_filter->setSigmaColor(2.0);
            // wls_filter->setLRCthresh(consts->lrMaxDiff());
            spdlog::info("StereoSGM created");

            // ====================================================================
            // ==================== Create PointCloudCollector ====================

            calib_ = utils::LoadCalibration(calibration_path_);
            rectification_ = utils::PerformRectification(calib_);
            mapping_ = utils::PrepareImagesMapping(calib_);

            spdlog::info("Creating output directory: {}", results_path_);
            utils::createDirectory(results_path_);
        } catch (const std::exception& e) {
            ++unsuccessful_runs;
            spdlog::info("unsuccessful_runs: {}", unsuccessful_runs);
            continue;
        }
        success = true;
    }
    spdlog::info("unsuccessful_runs: {}", unsuccessful_runs);
}

void DepthGeneration::PreparePaths() {
    // determine dataset_path
    std::filesystem::path dataset_path{bag_dir_path_ + "/"};
    while (dataset_path.filename().empty()) {
        dataset_path = dataset_path.parent_path();
    }
    dataset_path += "/";

    // determine results_path
    const std::string dataset_name = dataset_path.parent_path().filename();
    if (output_directory_.empty()) {
        results_path_ = std::filesystem::path{kDefaultResultsPath} / dataset_name;
    } else {
        results_path_ = std::filesystem::path{output_directory_} / dataset_name;
    }
    utils::DetermineResultsPath(results_path_);

    // path configuration
    spdlog::info("dataset_name: {}", dataset_name);
    spdlog::info("dataset_path: {}", dataset_path.string());
    spdlog::info("results_path: {}", results_path_);
    spdlog::info("calibration_path: {}", calibration_path_);
    spdlog::info("config_path: {}", config_path_);
    spdlog::info("verbose: {}", verbose_);
}

void DepthGeneration::LoadConfig() {
    spdlog::info("Loading Config");

    cfg_ = YAML::LoadFile(config_path_);
    spdlog::info("Loading {}", config_path_);
    if (not cfg_["SGM"]["disparitySize"]) {
        spdlog::warn("not cfg[SGM][disparitySize]");
    }
    if (not cfg_["SGM"]["imgWidth"]) {
        spdlog::warn("not cfg[SGM][imgWidth]");
    }
    if (not cfg_["SGM"]["imgHeight"]) {
        spdlog::warn("not cfg[SGM][imgHeight]");
    }
    if (not cfg_["SGM"]["numPaths"]) {
        spdlog::warn("not cfg[SGM][numPaths]");
    }
    if (not cfg_["SGM"]["censusTypeId"]) {
        spdlog::warn("not cfg[SGM][censusTypeId]");
    }
    if (not cfg_["SGM"]["p1"]) {
        spdlog::warn("not cfg[SGM][p1]");
    }
    if (not cfg_["SGM"]["p2"]) {
        spdlog::warn("not cfg[SGM][p2]");
    }
    if (not cfg_["SGM"]["uniqueness"]) {
        spdlog::warn("not cfg[SGM][uniqueness]");
    }
    if (not cfg_["SGM"]["subpixel"]) {
        spdlog::warn("not cfg[SGM][subpixel]");
    }
    if (not cfg_["SGM"]["lrMaxDiff"]) {
        spdlog::warn("not cfg[SGM][lrMaxDiff]");
    }
    if (not cfg_["DepthGeneration"]["computeColor"]) {
        spdlog::warn("not cfg[DepthGeneration][computeColor]");
    }
    if (not cfg_["DepthGeneration"]["computeDepth"]) {
        spdlog::warn("not cfg[DepthGeneration][computeDepth]");
    }
    if (not cfg_["DepthGeneration"]["computeDisparity"]) {
        spdlog::warn("not cfg[DepthGeneration][computeDisparity]");
    }
    if (not cfg_["DepthGeneration"]["computePointCloud"]) {
        spdlog::warn("not cfg[DepthGeneration][computePointCloud]");
    }
    if (not cfg_["DepthGeneration"]["computeSLAM"]) {
        spdlog::warn("not cfg[DepthGeneration][computeSLAM]");
    }
    if (not cfg_["DepthGeneration"]["useBilateralFilter"]) {
        spdlog::warn("not cfg[DepthGeneration][useBilateralFilter]");
    }
    if (not cfg_["DepthGeneration"]["bilateralRadius"]) {
        spdlog::warn("not cfg[DepthGeneration][bilateralRadius]");
    }
    if (not cfg_["DepthGeneration"]["bilateralIters"]) {
        spdlog::warn("not cfg[DepthGeneration][bilateralIters]");
    }
    if (not cfg_["DepthGeneration"]["useWlsFilter"]) {
        spdlog::warn("not cfg[DepthGeneration][useWlsFilter]");
    }
    if (not cfg_["DepthGeneration"]["depthPadding"]) {
        spdlog::warn("not cfg[DepthGeneration][depthPadding]");
    }
}

bool DepthGeneration::Compute(const cv::Mat& rgb_raw, const cv::Mat& ir1_raw, const cv::Mat& ir2_raw,
                              cv::Mat& out_disparity, cv::Mat& out_depth, cv::Mat& out_cloud) {
    cv::Mat color_img, rgb_rect, stereoL_rect, stereoR_rect, fpv_undis, stereoL_undis, stereoR_undis;
    
    if (cfg_["DepthGeneration"]["computeColor"].as<bool>()) {
        cv::undistort(rgb_raw, fpv_undis, calib_.rgb.intrinsics, calib_.rgb.distortion);
        cv::remap(fpv_undis, rgb_rect, rectification_.rgb_map_1, rectification_.rgb_map_2, cv::INTER_LINEAR);
    }
}

}  // namespace tools
