#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <array>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace marker6dof {
using Json = nlohmann::json;
using Transform = cv::Matx44d;
using Corners = std::array<cv::Point2d, 4>;
using Detections = std::map<int, Corners>;

struct Camera {
    int width = 0;
    int height = 0;
    cv::Matx33d matrix = cv::Matx33d::eye();
    cv::Mat distortion;
    std::string distortionModel = "opencv";
};
struct Quality {
    int minMarkers = 2;
    int minInlierCorners = 6;
    double ransacThresholdPx = 3.0;
    double ransacConfidence = 0.999;
    int ransacIterations = 100;
    double maxRmsePx = 2.5;
    double maxErrorPx = 6.0;
};
struct Body {
    std::string name;
    std::map<int, std::array<cv::Point3d, 4>> corners;
    Transform bodyTcp = Transform::eye();
};
struct Calibration {
    Camera camera;
    std::string dictionary = "DICT_4X4_50";
    std::map<std::string, Body> bodies;
    Quality quality;
};
struct Rig {
    int referenceId = 2;
    double markerSizeM = 0.048;
    std::map<int, Transform> referenceTag;
};
struct PoseEstimate {
    std::string body;
    std::string status = "LOST";
    std::string reason;
    std::vector<int> visibleIds;
    std::vector<int> usedIds;
    int cornerCount = 0;
    int inlierCount = 0;
    std::optional<double> rmsePx;
    std::optional<double> maxErrorPx;
    std::optional<Transform> cameraBody;
    std::optional<Transform> cameraTcp;
};
struct TrackingResult {
    std::string status = "LOST";
    std::string reason;
    std::vector<int> visibleIds;
    std::vector<int> usedIds;
    std::vector<int> rejectedIds;
    std::optional<double> reprojectionErrorPx;
    std::map<int, std::optional<double>> tagErrors;
    std::optional<Transform> pose;
};

Json readJson(const std::string& path);
Camera cameraFromJson(const Json& value);
Json cameraToJson(const Camera& camera);
Calibration loadCalibration(const std::string& path);
Rig loadRig(const std::string& path, double markerSizeM);
Transform transformFromJson(const Json& value, const std::string& name = "transform");
Json transformToJson(const Transform& transform);
Transform inverseRigid(const Transform& transform);
cv::Point3d transformPoint(const Transform& transform, const cv::Point3d& point);
Transform makeTransform(const cv::Vec3d& rvec, const cv::Vec3d& tvec);
cv::Vec3d rotationVector(const Transform& transform);
cv::Vec3d translation(const Transform& transform);
cv::Vec4d quaternionXyzw(const Transform& transform);
std::array<cv::Point3d, 4> markerObjectPoints(double sizeM);
int dictionaryId(const std::string& name);
Detections detectMarkers(const cv::Mat& image, const std::string& dictionary = "DICT_4X4_50");
std::vector<cv::Point2d> normalizePixels(const Camera& camera, const Corners& corners);
std::vector<cv::Point2d> normalizePixels(const Camera& camera, const std::vector<cv::Point2d>& pixels);
std::vector<cv::Point2d> projectCamera(const Camera& camera, const std::vector<cv::Point3d>& points);
Json poseToJson(const PoseEstimate& pose);
Json trackingToJson(const TrackingResult& result);

class MarkerLocalizer {
public:
    explicit MarkerLocalizer(Calibration calibration);
    const Calibration& calibration() const;
    Detections detect(const cv::Mat& image) const;
    std::vector<PoseEstimate> estimate(const Detections& detections) const;
private:
    Calibration calibration_;
};

class RigTracker {
public:
    explicit RigTracker(Rig rig, double maxErrorPx = 2.5, double minEdgePx = 24.0);
    ~RigTracker();
    RigTracker(const RigTracker&) = delete;
    RigTracker& operator=(const RigTracker&) = delete;
    void reset();
    TrackingResult solve(const Detections& detections, const Camera& camera, double timestamp);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class RelativeTrace {
public:
    explicit RelativeTrace(std::size_t limit = 3000);
    void reset();
    Json update(const TrackingResult& result, double timestamp);
    Json trace() const;
private:
    std::size_t limit_;
    std::optional<Transform> initialInverse_;
    std::optional<cv::Vec3d> filtered_, lastRaw_;
    std::optional<cv::Vec4d> filteredQuaternion_;
    std::optional<double> lastTime_;
    std::deque<Json> trace_;
    int segment_ = 0;
    bool broken_ = true;
};
} // namespace marker6dof
