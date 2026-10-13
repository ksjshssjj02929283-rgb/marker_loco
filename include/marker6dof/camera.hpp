#pragma once

#include "marker6dof/core.hpp"
#include <cstddef>
#include <map>
#include <string>

namespace marker6dof {
Camera cameraFromMetadata(const Json& metadata);
void checkImageSize(const Camera& camera, const cv::Mat& image);

// Recorded clocks are expressed as elapsed seconds from the first frame.
// Keeping their real gaps is essential to the temporal rejection gates.
struct FrameTimes {
    std::map<std::size_t, double> elapsed;
    std::string source = "frame_index/video_fps";
    double fps = 0.0;
    double at(std::size_t frame) const;
};
FrameTimes loadFrameTimes(const std::string& session, double fps,
                         const std::string& explicitCsv = "");
} // namespace marker6dof
