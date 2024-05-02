#include <opencv2/core/mat.hpp>
#include <sensor_msgs/msg/detail/compressed_image__struct.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace tools {

class Converter {
public:
    static cv::Mat Convert(const sensor_msgs::msg::Image& msg, const std::string encoding);
    static cv::Mat Convert(const sensor_msgs::msg::CompressedImage& msg, const std::string encoding);
    static sensor_msgs::msg::Image Convert(const cv::Mat& img, const std::string encoding, const std_msgs::msg::Header& header);
};

}  // namespace tools::conv
