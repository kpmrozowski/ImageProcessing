#ifndef constants_HPP
#define constants_HPP

#include <string>

#include <opencv2/core/types.hpp>

namespace tools {

static constexpr float kInvalidDepth = 10000.f;
static constexpr bool kFitImagesToStereo = true;
static constexpr bool kFitLeftStereoToRgb = true;
static const std::string kStereoFileFormat = ".png";

}  // namespace tools

#endif  // constants_HPP
