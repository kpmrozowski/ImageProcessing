#include "depth_generator/DepthGeneration.hpp"

#include <functional>
#include <iostream>

#include <libsgm.h>
#include <opencv2/core/hal/interface.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <string>

#include "depth_generator/utils.hpp"

namespace tools {

DepthGeneration::DepthGeneration(const YAML::Node& cfg) : cfg_(cfg) {
    this->Init();
}

void DepthGeneration::Init() {
    bool success = false;
    size_t unsuccessful_runs = 0;
    while (not success and unsuccessful_runs < 20) {
        spdlog::info("unsuccessful_runs: {}", unsuccessful_runs);
        try {
            // ====================================================================
            // ==================== Create PointCloudCollector ====================

            spdlog::info("Loading CamSysCalib");
            utils::LoadCalibration(calib_, cfg_["calibrationPath"].as<std::string>());

            spdlog::info("Performing rectification");
            utils::PerformRectification(rectification_, calib_, cfg_);

            spdlog::info("Compute Projection Matrix");
            mapping_ = utils::ComputeProjectionMatrix(calib_);

            if (cfg_["DepthGeneration"]["computeDisparity"].as<bool>()) {
                spdlog::info("Creating sgm::StereoSGM");
                ssgm_ = std::make_unique<std::reference_wrapper<sgm::StereoSGM>>(utils::CreateStereoSGM(cfg_));
            } else {
                spdlog::info("please enable [DepthGeneration][computeDisparity]");
                exit(0);
            }

            if (cfg_["DepthGeneration"]["computeDisparity"].as<bool>() &&
                cfg_["DepthGeneration"]["useBilateralFilter"].as<bool>()) {
                spdlog::info("Creating BilateralFilter...");
                bilateral_filter_ = cv::cuda::createDisparityBilateralFilter(cfg_["SGM"]["disparitySize"].as<int>(),
                                                                             cfg_["DepthGeneration"]["bilateralRadius"].as<int>(),
                                                                             cfg_["DepthGeneration"]["bilateralIters"].as<int>());
                spdlog::info("Created BilateralFilter");
            }

            if (cfg_["DepthGeneration"]["computeDisparity"].as<bool>() &&
                cfg_["DepthGeneration"]["useWlsFilter"].as<bool>()) {
                spdlog::info("Creating cv::cuda::StereoSGM...");
                int stereo_sgm_mode = 0;

                // clang-format off
                switch (cfg_["DepthGeneration"]["stereoSGMMode"].as<int>()) {
                    case 0: stereo_sgm_mode = cv::cuda::StereoSGM::MODE_SGBM; break;
                    case 1: stereo_sgm_mode = cv::cuda::StereoSGM::MODE_HH; break;
                    case 2: stereo_sgm_mode = cv::cuda::StereoSGM::MODE_SGBM_3WAY; break;
                    case 3: stereo_sgm_mode = cv::cuda::StereoSGM::MODE_HH4; break;
                    default: spdlog::error("invalid cfg[DepthGeneration][stereoSGMMode]"); exit(1);
                }
                cv::Ptr<cv::cuda::StereoSGM> cv_sgm_left = cv::cuda::createStereoSGM(
                    0,
                    cfg_["SGM"]["disparitySize"].as<int>(),
                    cfg_["SGM"]["p1"].as<int>(), cfg_["SGM"]["p2"].as<int>(),
                    100 * static_cast<int>(1.f - cfg_["SGM"]["uniqueness"].as<float>()),
                    stereo_sgm_mode);
                // clang-format on
                spdlog::info("Created cv::cuda::StereoSGM");

                spdlog::info("Creating cv::ximgproc::DisparityWLSFilter...");
                // cv_sgm_left->setPreFilterCap(cv::cuda::StereoBM::PREFILTER_XSOBEL);
                wls_filter_ = cv::ximgproc::createDisparityWLSFilter(cv_sgm_left);
                spdlog::info("Created cv::ximgproc::DisparityWLSFilter");
                // wls_filter_->setLambda(8000);
                // wls_filter_->setSigmaColor(2.0);
                // wls_filter_->setLRCthresh(consts->lrMaxDiff());
            }
            success = true;
        } catch (const std::exception& e) {
            ++unsuccessful_runs;
            spdlog::error("unsuccessful_runs: {}, what: {}", unsuccessful_runs, e.what());
            continue;
        }
        success = true;
    }
    spdlog::info("unsuccessful_runs: {}", unsuccessful_runs);

    stereo_size_ = cv::Size{cfg_["stereoWidth"].as<int>(), cfg_["stereoHeight"].as<int>()};
    sgm_size_ = cv::Size{cfg_["SGM"]["imgWidth"].as<int>(), cfg_["SGM"]["imgHeight"].as<int>()};
}

bool DepthGeneration::Compute(const cv::Mat& rgb_raw, const cv::Mat& stereoL_raw, const cv::Mat& stereoR_raw,
                              cv::OutputArray& out_disparityL, cv::OutputArray& out_disparityR,
                              cv::OutputArray& out_depthL, cv::OutputArray& out_depthR, Eigen::Matrix3Xf& out_cloud) {
    if (rgb_raw.empty()) {
        spdlog::warn("rgb_raw is empty");
        return false;
    }
    if (stereoL_raw.empty()) {
        spdlog::warn("stereoL_raw is empty");
        return false;
    }
    if (stereoR_raw.empty()) {
        spdlog::warn("stereoR_raw is empty");
        return false;
    }
    cv::Mat color_img, rgb_rect, stereoL_rect, stereoR_rect, fpv_undis, stereoL_undis, stereoR_undis;

    auto start_rectify = std::chrono::high_resolution_clock::now();
    auto start_disparity = std::chrono::high_resolution_clock::now();
    auto start_point_cloud = std::chrono::high_resolution_clock::now();

    auto duration_rectify = std::chrono::duration<double, std::milli>();
    auto duration_disparity = std::chrono::duration<double, std::milli>();
    auto duration_point_cloud = std::chrono::duration<double, std::milli>();

    if (cfg_["DepthGeneration"]["computeColor"].as<bool>()) {
        cv::undistort(rgb_raw, fpv_undis, calib_.rgb.intrinsics, calib_.rgb.distortion);
        cv::remap(fpv_undis, rgb_rect, rectification_.rgb_map_1, rectification_.rgb_map_2, cv::INTER_LINEAR);
    }
    start_rectify = std::chrono::high_resolution_clock::now();
    cv::undistort(stereoL_raw, stereoL_undis, calib_.stereoL_rect.intrinsics, calib_.stereoL_rect.distortion);
    cv::remap(stereoL_undis, stereoL_rect, rectification_.gray_map_1, rectification_.gray_map_2, cv::INTER_LINEAR);

    cv::undistort(stereoR_raw, stereoR_undis, calib_.stereoR_rect.intrinsics, calib_.stereoR_rect.distortion);

    cv::remap(stereoR_undis, stereoR_rect, rectification_.gray_map_1, rectification_.gray_map_2, cv::INTER_LINEAR);
    duration_rectify = std::chrono::high_resolution_clock::now() - start_rectify;

    // compute disparity
    start_disparity = std::chrono::high_resolution_clock::now();
    cv::Mat disparityL, disparityR, filtered_dispL, filtered_dispR;
    cv::cuda::GpuMat d_filtered_dispL, d_filtered_dispR;
    // cv::cuda::GpuMat d_stereoL_rect(stereoL_rect), d_stereoR_rect(stereoR_rect), d_disparityL, d_disparityR;
    if (cfg_["DepthGeneration"]["computeDisparity"].as<bool>()) {
        try {
            // fmt::print("ComputeDisparity start\n");
            cv::Mat stereoL_resized, stereoR_resized, disparityL_resized, disparityR_resized;
            if (sgm_size_ != stereo_size_) {
                cv::resize(stereoL_rect, stereoL_resized, sgm_size_, 0., 0., cv::INTER_LINEAR);
                cv::resize(stereoR_rect, stereoR_resized, sgm_size_, 0., 0., cv::INTER_LINEAR);
            } else {
                stereoL_rect.copyTo(stereoL_resized);
                stereoR_rect.copyTo(stereoR_resized);
            }

            if (nullptr == ssgm_) {
                spdlog::error("nullptr == ssgm_");
                exit(1);
            }
            std::tie(disparityL_resized, disparityR_resized) =
                utils::ComputeDisparity(ssgm_->get(), stereoL_resized, stereoR_resized);

            if (sgm_size_ != stereo_size_) {
                cv::resize(disparityL_resized, disparityL, stereo_size_, 0., 0., cv::INTER_CUBIC);
                cv::resize(disparityR_resized, disparityR, stereo_size_, 0., 0., cv::INTER_CUBIC);
                disparityL.convertTo(disparityL, CV_16U);
                disparityR.convertTo(disparityR, CV_16U);
            } else {
                disparityL_resized.convertTo(disparityL, CV_16U);
                disparityR_resized.convertTo(disparityR, CV_16U);
            }
            // fmt::print("ComputeDisparity end\n");

            // cv_sgm_left->compute(d_stereoL_rect, d_stereoR_rect, d_disparityL);
            // d_disparityL.download(disparityL);
            // cv_sgm_left->compute(d_stereoR_rect, d_stereoL_rect, d_disparityR);
            // d_disparityR.download(disparityR);
            // cv_sgm_right->compute(stereoR_rect, stereoL_rect, disparityR);
            // disparityL.convertTo(disparityL, CV_16S);
            // disparityR.convertTo(disparityR, CV_16S);
            // d_disparityL.upload(disparityL);
            // d_disparityR.upload(disparityR);
            // bilateral_filter_->apply(d_disparityL, d_stereoL_rect, d_disparityL);
            // bilateral_filter_->apply(d_disparityR, d_stereoR_rect, d_disparityR);
            // d_disparityL.download(disparityL);
            // d_disparityR.download(disparityR);
            // disparityR *= -16;
        } catch (const cv::Exception& e) {
            std::cout << "\nError in generate_depth.cpp in cv_sgm_left->compute: " << e.what()
                      << "\ndisparity.type: " << cv::typeToString(disparityL.type())
                      << "\ndisparity.size: " << disparityL.size() << ", types: " << cv::typeToString(CV_8UC1) << ", "
                      << cv::typeToString(CV_16SC1) << ", " << cv::typeToString(CV_32SC1) << ", "
                      << cv::typeToString(CV_32FC1) << "\n";
            return false;
        }
        disparityL.convertTo(disparityL, CV_16S);
        disparityR.convertTo(disparityR, CV_16S);
        disparityL.assignTo(filtered_dispL);
        disparityR.assignTo(filtered_dispR);
        if (cfg_["SGM"]["subpixel"].as<bool>()) {
            filtered_dispR *= -16;
        }
        try {
            if (cfg_["DepthGeneration"]["useBilateralFilter"].as<bool>()) {
                // apply bilateral_filter
                cv::cuda::GpuMat d_stereoL_rect(stereoL_rect), d_stereoR_rect(stereoR_rect), d_disparity, d_disparityL,
                    d_disparityR;
                d_disparityL.upload(filtered_dispL);
                d_disparityR.upload(filtered_dispR);
                bilateral_filter_->apply(d_disparityL, d_stereoL_rect, d_disparityL);
                bilateral_filter_->apply(d_disparityR, d_stereoR_rect, d_disparityR);
                d_disparityL.download(filtered_dispL);
                d_disparityR.download(filtered_dispR);
            }
        } catch (const cv::Exception& e) {
            std::cout << "\nError in generate_depth.cpp in bilateral_filter->apply: " << e.what()
                      << "\ndisparityL.type: " << cv::typeToString(disparityL.type())
                      << "\nmapping.type: " << cv::typeToString(mapping_.type())
                      << ", types: " << cv::typeToString(CV_8UC1) << ", " << cv::typeToString(CV_16SC1) << ", "
                      << cv::typeToString(CV_32SC1) << ", " << cv::typeToString(CV_32FC1) << "\n";
            return false;
        }
        try {
            if (cfg_["DepthGeneration"]["useWlsFilter"].as<bool>()) {
                wls_filter_->filter(filtered_dispL, stereoL_rect, filtered_dispL, filtered_dispR);
            }
        } catch (const cv::Exception& e) {
            std::cout << "\nError in generate_depth.cpp in wls_filter_->filter: " << e.what()
                      << "\ndisparityL.type: " << cv::typeToString(disparityL.type())
                      << "\nmapping.type: " << cv::typeToString(mapping_.type())
                      << ", types: " << cv::typeToString(CV_8UC1) << ", " << cv::typeToString(CV_16SC1) << ", "
                      << cv::typeToString(CV_32SC1) << ", " << cv::typeToString(CV_32FC1) << "\n";
            return false;
        }

        // convert to floats
        // filtered_dispL.convertTo(filtered_dispL, CV_16SC1);
        // filtered_dispR.convertTo(filtered_dispR, CV_16SC1);
        filtered_dispL.assignTo(filtered_dispL, CV_32FC1);
        filtered_dispR.assignTo(filtered_dispR, CV_32FC1);
        if (cfg_["SGM"]["subpixel"].as<bool>()) {
            filtered_dispL /= 16;
            filtered_dispR /= -16;
        }
        disparityL.convertTo(disparityL, CV_32FC1);
        disparityR.convertTo(disparityR, CV_32FC1);
    }
    duration_disparity = std::chrono::high_resolution_clock::now() - start_disparity;

    // spdlog::info("disparityL({}, {}).type: {}, disparityR({}, {}).type: {}", disparityL.rows, disparityL.cols,
    //              cv::typeToString(disparityL.type()), disparityR.rows, disparityR.cols,
    //              cv::typeToString(disparityR.type()));
    // cv::Mat tempL, tempR;
    // disparityL.convertTo(tempL, CV_32FC1, 1./5000.);
    // disparityR.convertTo(tempR, CV_32FC1, 1./5000.);
    // cv::imshow("disparityL", tempL);
    // cv::imshow("disparityR", tempR);
    // cv::waitKey(0);

    /// TODO: publish
    filtered_dispL.copyTo(out_disparityL);
    filtered_dispR.copyTo(out_disparityR);
    // if (cfg_["DepthGeneration"]["computeDisparity"].as<bool>() and consts->saveDisparity()) {
    //     // save disparity
    //     std::thread saving_thread{saveImageAsDisparityPath, filtered_disp,   fpv_stereoL_stereoR_paths.stereo_l,
    //                                 output_disparity_path,    "filtered.tiff", s};
    //     std::thread saving_threadL{saveImageAsDisparityPath, disparityL, fpv_stereoL_stereoR_paths.stereo_l,
    //                                 output_disparity_path,    "L.tiff",   s};
    //     std::thread saving_threadR{saveImageAsDisparityPath, disparityR, fpv_stereoL_stereoR_paths.stereo_l,
    //                                 output_disparity_path,    "R.tiff",   s};
    //     saving_thread.detach();
    //     saving_threadL.detach();
    //     saving_threadR.detach();
    //     // saveImageAsDisparityPath(disparity, fpv_stereoL_stereoR_paths.stereo_l, output_disparity_path,
    //     // ".tiff",
    //     //                          s);
    // }

    /// TODO: merge two alligned images: stereoL and fpv
    // stereoL_rect.convertTo(stereoL_rect, fpv_rect.type());
    // cv::addWeighted(fpv_rect, 0.5, stereoL_rect, 0.5, 0, color_img);
    // fmt::print("c color_img.size=({}, {}),\n", color_img.size[0], color_img.size[1]);

    /// TODO: publish
    // if (consts->computeColor()) {
    //     // std::thread saving_thread1{saveImageAsDisparityPath, fpv_undis, fpv_stereoL_stereoR_paths[1],
    //     //                            output_fpv_undis_path, ".png", s};
    //     saveImageAsDisparityPath(fpv_undis, fpv_stereoL_stereoR_paths.stereo_l, output_fpv_undis_path, ".png",
    //                                 s);
    //     saveImageAsDisparityPath(stereoL_undis, fpv_stereoL_stereoR_paths.stereo_l, output_stereoL_undis_path,
    //                                 ".png", s);
    //     saveImageAsDisparityPath(stereoR_undis, fpv_stereoL_stereoR_paths.stereo_r, output_stereoR_undis_path,
    //                                 ".png", s);

    //     saveImageAsDisparityPath(fpv_rect, fpv_stereoL_stereoR_paths.stereo_l, output_fpv_rect_path, ".png", s);
    //     saveImageAsDisparityPath(stereoL_rect, fpv_stereoL_stereoR_paths.stereo_l, output_stereoL_rect_path,
    //                                 ".png", s);
    //     saveImageAsDisparityPath(stereoR_rect, fpv_stereoL_stereoR_paths.stereo_r, output_stereoR_rect_path,
    //                                 ".png", s);
    // }

    // compute depth
    start_point_cloud = std::chrono::high_resolution_clock::now();
    cv::Mat img2d_to_coords3d_mapL, img2d_to_coords3d_mapR{};
    cv::cuda::GpuMat d_img2d_to_coords3d_mapL, d_img2d_to_coords3d_mapR;
    // bool handle_missing_values = true;
    if (cfg_["DepthGeneration"]["computeDepth"].as<bool>() or cfg_["DepthGeneration"]["computePointCloud"].as<bool>()) {
        // wls_filter_->filter(disparity,stereoL_rect,disparity,disparity);

        // cv::Mat confidence = 0 * cv::Mat::ones(stereoL_rect.size(), CV_32FC1);
        // auto filter = cv::ximgproc::createFastBilateralSolverFilter(stereoL_rect, 8.0f, 8.0f, 8.0f);
        // filter->filter(disparity, confidence, disparity);
        try {
            // cv::reprojectImageTo3D(disparity, img2d_to_coords3d_map, mapping_, handle_missing_values);
            cv::Mat mapping_float;
            mapping_.convertTo(mapping_float, CV_32F);
            d_filtered_dispL.upload(filtered_dispL);
            d_filtered_dispR.upload(filtered_dispR);
            cv::cuda::reprojectImageTo3D(d_filtered_dispL, d_img2d_to_coords3d_mapL, mapping_float, 3);
            cv::cuda::reprojectImageTo3D(d_filtered_dispR, d_img2d_to_coords3d_mapR, mapping_float, 3);
            d_img2d_to_coords3d_mapL.download(img2d_to_coords3d_mapL);
            d_img2d_to_coords3d_mapR.download(img2d_to_coords3d_mapR);
        } catch (const cv::Exception& e) {
            std::cout << "\nError in generate_depth.cpp in reprojectImageTo3D: " << e.what()
                      << "\ndisparityL.type: " << cv::typeToString(filtered_dispL.type())
                      << "\ndisparityR.type: " << cv::typeToString(filtered_dispR.type())
                      << "\nmapping.type: " << cv::typeToString(mapping_.type())
                      << ", types: " << cv::typeToString(CV_8UC1) << ", " << cv::typeToString(CV_16SC1) << ", "
                      << cv::typeToString(CV_32SC1) << ", " << cv::typeToString(CV_32FC1) << "\n";
            return false;
        }
    }
    duration_point_cloud = std::chrono::high_resolution_clock::now() - start_point_cloud;
    if (cfg_["DepthGeneration"]["computeDepth"].as<bool>()) {
        /// TODO: publish depth
        std::vector<cv::Mat> channelsL, channelsR;
        cv::split(img2d_to_coords3d_mapL, channelsL);
        cv::split(img2d_to_coords3d_mapR, channelsR);
        cv::Mat depthL = channelsL[2];
        cv::Mat depthR = channelsR[2];
        // std::cout << depthR;
        // exit(0);
        depthL.setTo(0., depthL < 0.);
        depthR.setTo(0., depthR < 0.);

        double min, max;
        // cv::minMaxLoc(-channels[2], &min, &max);
        min = 0.;
        max = 10000.;
        depthL = (depthL - min) / (max - min);
        depthR = (depthR - min) / (max - min);
        depthL.setTo(0., depthL > 1.);
        depthL.setTo(0., depthL < 0.);
        depthR.setTo(0., depthR > 1.);
        depthR.setTo(0., depthR < 0.);
        depthL.copyTo(out_depthL);
        depthR.copyTo(out_depthR);
        // channels[2].copyTo(out_depth);

        // cv::imshow("after out_depth", out_depth);
        // cv::waitKey(0);
        // saveDepthAsDisparityPath(img2d_to_coords3d_map, fpv_stereoL_stereoR_paths.stereo_l, output_depth_path,
        // ".tiff",
        //                          s);
    }
    if (cfg_["DepthGeneration"]["computePointCloud"].as<bool>() and
        cfg_["DepthGeneration"]["computeColor"].as<bool>()) {
        /// TODO: publish point-cloud
        // saveColorPointCloudAsDisparityPath(img2d_to_coords3d_map, fpv_rect, fpv_stereoL_stereoR_paths.stereo_l,
        //                                     output_pc_path, "-color.ply", s);
    } else if (cfg_["DepthGeneration"]["computePointCloud"].as<bool>()) {
        // utils::multiplyChannelWise(img2d_to_coords3d_mapL, {2}, -1.);
        // std::cout << img2d_to_coords3d_mapL;
        // exit(0);
        utils::removeBoundaryPoints(img2d_to_coords3d_mapL, cfg_["DepthGeneration"]["depthPadding"].as<int>());
        Eigen::MatrixXf cloud = utils::convertPcMatToEigen(img2d_to_coords3d_mapL);
        out_cloud = cloud;
        // std::thread saving_thread1{savePointCloudAsDisparityPath,
        //                             img2d_to_coords3d_map,
        //                             fpv_stereoL_stereoR_paths.stereo_l,
        //                             output_pc_path,
        //                             ".ply",
        //                             s};
        // saving_thread1.detach();
    }

    // fmt::print("duration_rectify={}\nduration_disparity={}\nduration_point_cloud={}\n", duration_rectify.count(),
    //            duration_disparity.count(), duration_point_cloud.count());

    return true;
}

}  // namespace tools
