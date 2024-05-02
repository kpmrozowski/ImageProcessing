#ifndef types_HPP
#define types_HPP

#include <cstdint>
#include <string>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace tools {

struct ImuMeas {
    cv::Point3f acc;
    cv::Point3f gyr;
    double timestamp;
};

struct MonoCalibration {
    cv::Mat intrinsics;
    cv::Mat distortion;
    cv::Mat rotation;
    cv::Mat translation;

    MonoCalibration(MonoCalibration& other) {
        intrinsics = other.intrinsics.clone();
        distortion = other.distortion.clone();
        rotation = other.rotation.clone();
        translation = other.translation.clone();
    }
    MonoCalibration(MonoCalibration&& other) {
        intrinsics = other.intrinsics.clone();
        distortion = other.distortion.clone();
        rotation = other.rotation.clone();
        translation = other.translation.clone();
    }
    MonoCalibration& operator=(MonoCalibration& other) {
        intrinsics = other.intrinsics.clone();
        distortion = other.distortion.clone();
        rotation = other.rotation.clone();
        translation = other.translation.clone();
        return *this;
    }
    MonoCalibration& operator=(MonoCalibration&& other) {
        intrinsics = other.intrinsics.clone();
        distortion = other.distortion.clone();
        rotation = other.rotation.clone();
        translation = other.translation.clone();
        return *this;
    }

    void freeIntrinsics() {
        intrinsics.release();
    }
    void freeDistortion() {
        distortion.release();
    }
    void freeRotation() {
        rotation.release();
    }
    void freeTranslation() {
        translation.release();
    }

    MonoCalibration()
            : intrinsics(3, 3, CV_64F), distortion(5, 1, CV_64F), rotation(3, 3, CV_64F), translation(3, 1, CV_64F) {
    }

    ~MonoCalibration() {
        freeIntrinsics();
        freeDistortion();
        freeRotation();
        freeTranslation();
    }
};

struct CamSysCalib {
    MonoCalibration rgb;
    MonoCalibration stereoL_rect;
    MonoCalibration stereoR_rect;
    cv::Mat stereoL_stereoR_transform;

    void freeRgb() {
        rgb.freeIntrinsics();
        rgb.freeDistortion();
        rgb.freeRotation();
        rgb.freeTranslation();
    }
    void freeStereoL_rect() {
        stereoL_rect.freeIntrinsics();
        stereoL_rect.freeDistortion();
        stereoL_rect.freeRotation();
        stereoL_rect.freeTranslation();
    }
    void freeStereoR_rect() {
        stereoR_rect.freeIntrinsics();
        stereoR_rect.freeDistortion();
        stereoR_rect.freeRotation();
        stereoR_rect.freeTranslation();
    }
    void freeStereoL_stereoR_transform() {
        stereoL_stereoR_transform.release();
    }

    CamSysCalib(CamSysCalib& other) {
        rgb = other.rgb;
        stereoL_rect = other.stereoL_rect;
        stereoR_rect = other.stereoR_rect;
        stereoL_stereoR_transform = other.stereoL_stereoR_transform.clone();
    }
    CamSysCalib(CamSysCalib&& other) {
        rgb = other.rgb;
        stereoL_rect = other.stereoL_rect;
        stereoR_rect = other.stereoR_rect;
        stereoL_stereoR_transform = other.stereoL_stereoR_transform.clone();
    }

    CamSysCalib& operator=(CamSysCalib& other) {
        rgb = other.rgb;
        stereoL_rect = other.stereoL_rect;
        stereoR_rect = other.stereoR_rect;
        stereoL_stereoR_transform = other.stereoL_stereoR_transform.clone();
        return *this;
    }
    CamSysCalib& operator=(CamSysCalib&& other) {
        rgb = other.rgb;
        stereoL_rect = other.stereoL_rect;
        stereoR_rect = other.stereoR_rect;
        stereoL_stereoR_transform = other.stereoL_stereoR_transform.clone();
        return *this;
    }

    CamSysCalib() : rgb(), stereoL_rect(), stereoR_rect(), stereoL_stereoR_transform(4, 4, CV_64F) {
    }

    ~CamSysCalib() {
        freeRgb();
        freeStereoL_rect();
        freeStereoR_rect();
        freeStereoL_stereoR_transform();
    }
};

struct Rectification {
    cv::Mat rgb_map_1;
    cv::Mat rgb_map_2;
    cv::Mat gray_map_1;
    cv::Mat gray_map_2;
    cv::Mat new_intrinsics;

    Rectification() = default;

    Rectification(const cv::Size& resolution)
            : rgb_map_1(resolution, CV_32FC2)
            , rgb_map_2(resolution, CV_32FC2)
            , gray_map_1(resolution, CV_32FC2)
            , gray_map_2(resolution, CV_32FC2)
            , new_intrinsics(3, 3, CV_64F) {
    }
};

struct PointCloudSubstrates {
    uint32_t currentImgIdx;
    uint32_t lastProcessedImgIdx;
    uint64_t timestamp;
    cv::Mat stereoL;
    cv::Mat stereoR;
    std::string pathL;
    std::string pathR;
};

enum class CameraDirection : uint8_t { FRONT, REAR, RIGHT, LEFT, UP, DOWN };

// Define a 2D bounding box
// ! Remember that vision_msgs/BoundingBox2D uses center + width + height
struct BoundingBox2D {
    cv::Point2f min;  // Minimum 2D point (top-left corner)
    cv::Point2f max;  // Maximum 2D point (bottom-right corner)
};

}  // namespace tools

#endif  // types_HPP
