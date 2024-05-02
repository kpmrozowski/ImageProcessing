#include "cv_bridge_interface/Converter.hpp"

#include <sensor_msgs/msg/detail/compressed_image__struct.hpp>
#include <string>

#include <cv_bridge/cv_bridge.h>

namespace tools {

cv::Mat Converter::Convert(const sensor_msgs::msg::Image& msg, const std::string encoding) {
    auto bridge = cv_bridge::toCvCopy(msg, encoding);
    return bridge->image;
}

cv::Mat Converter::Convert(const sensor_msgs::msg::CompressedImage& msg, const std::string encoding) {
    auto bridge = cv_bridge::toCvCopy(msg, encoding);
    return bridge->image;
}

sensor_msgs::msg::Image Converter::Convert(const cv::Mat& img, const std::string encoding, const std_msgs::msg::Header& header) {
    cv_bridge::CvImage bridge{header, encoding, img};
    return *bridge.toImageMsg();
}

}
