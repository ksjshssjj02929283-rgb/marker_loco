#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace demo {

struct CameraModel {
    int width = 0;
    int height = 0;
    cv::Matx33d matrix = cv::Matx33d::eye();
    std::vector<double> distortion;
    std::string distortionModel;
    double markerSizeM = 0.048;
};

struct SessionInput {
    std::string configPath;
    std::string videoPath;
    std::string rigPath;
    int referenceId = 2;
    std::string timestampSource;
    CameraModel camera;
    std::vector<uint64_t> timestampsNs;
    bool hasRecordedTimes = false;

    // Keeps package-mode generated configuration alive, then removes it.
    std::shared_ptr<void> temporaryConfigOwner;
};

SessionInput loadVideoInput(const std::string& configPath,
                           const std::string& videoPath,
                           const std::string& timestampCsv = "");
SessionInput loadPackageInput(const std::string& packagePath,
                             const std::string& sessionName = "",
                             const std::string& rigOverride = "");

} // namespace demo
