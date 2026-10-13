#include "marker6dof/core.hpp"
#include "marker6dof/camera.hpp"

#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace marker6dof;

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Action>
void requireInvalid(Action action, const std::string& message) {
    try {
        action();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

Transform knownTransform(const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
    cv::Matx33d rotation;
    cv::Rodrigues(rvec, rotation);
    Transform result = Transform::eye();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) result(row, col) = rotation(row, col);
        result(row, 3) = tvec[row];
    }
    return result;
}

void requirePose(const Transform& actual, const Transform& expected, const std::string& message) {
    double squaredPosition = 0;
    cv::Matx33d actualRotation, expectedRotation;
    for (int row = 0; row < 3; ++row) {
        const double difference = actual(row, 3) - expected(row, 3);
        squaredPosition += difference * difference;
        for (int col = 0; col < 3; ++col) {
            actualRotation(row, col) = actual(row, col);
            expectedRotation(row, col) = expected(row, col);
        }
    }
    cv::Vec3d angle;
    cv::Rodrigues(actualRotation.t() * expectedRotation, angle);
    require(std::sqrt(squaredPosition) < 1e-5 && cv::norm(angle) < 1e-3, message);
}

Camera syntheticCamera() {
    Camera camera;
    camera.width = 1280;
    camera.height = 720;
    camera.matrix = {650, 0, 640, 0, 645, 360, 0, 0, 1};
    camera.distortion = cv::Mat::zeros(5, 1, CV_64F);
    camera.distortionModel = "none";
    return camera;
}

Rig syntheticRig() {
    Rig rig;
    rig.markerSizeM = .048;
    rig.referenceId = 2;
    rig.referenceTag[2] = Transform::eye();
    rig.referenceTag[3] = knownTransform({0, 0, 0}, {.075, 0, 0});
    rig.referenceTag[4] = knownTransform({0, .7, 0}, {.035, .075, .020});
    return rig;
}

std::array<cv::Point3d, 4> independentSquare(double size) {
    const double half = size / 2;
    return {{{-half, half, 0}, {half, half, 0}, {half, -half, 0}, {-half, -half, 0}}};
}

std::array<cv::Point3d, 4> bodyCorners(const Rig& rig, int tag) {
    auto points = independentSquare(rig.markerSizeM);
    const auto& transform = rig.referenceTag.at(tag);
    for (auto& point : points) {
        const cv::Vec4d output = transform * cv::Vec4d(point.x, point.y, point.z, 1);
        point = {output[0], output[1], output[2]};
    }
    return points;
}

Corners projectCorners(const std::array<cv::Point3d, 4>& points,
                       const Transform& pose, const Camera& camera) {
    cv::Matx33d rotation;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) rotation(row, col) = pose(row, col);
    }
    cv::Vec3d rvec;
    cv::Rodrigues(rotation, rvec);
    std::vector<cv::Point2d> pixels;
    cv::projectPoints(std::vector<cv::Point3d>(points.begin(), points.end()), rvec,
                      cv::Vec3d(pose(0, 3), pose(1, 3), pose(2, 3)), camera.matrix,
                      camera.distortion, pixels);
    Corners corners;
    std::copy(pixels.begin(), pixels.end(), corners.begin());
    return corners;
}

Detections projectRig(const Rig& rig, const Transform& pose, const Camera& camera) {
    Detections detections;
    for (const auto& tag : rig.referenceTag) detections[tag.first] = projectCorners(bodyCorners(rig, tag.first), pose, camera);
    return detections;
}

Calibration syntheticCalibration() {
    Calibration calibration;
    calibration.camera = syntheticCamera();
    Body body;
    body.name = "tool";
    const Rig rig = syntheticRig();
    for (const auto& tag : rig.referenceTag) body.corners[tag.first] = bodyCorners(rig, tag.first);
    body.bodyTcp = knownTransform({.08, -.03, .12}, {.01, .09, .04});
    calibration.bodies[body.name] = body;
    return calibration;
}

Json calibrationJson() {
    const auto calibration = syntheticCalibration();
    Json raw = cameraToJson(calibration.camera);
    raw["schema_version"] = 1;
    raw["units"] = "m";
    raw["aruco"] = {{"dictionary", calibration.dictionary}};
    raw["bodies"] = Json::object();
    for (const auto& item : calibration.bodies) {
        Json tags = Json::object();
        for (const auto& tag : item.second.corners) {
            Json points = Json::array();
            for (const auto& point : tag.second) points.push_back({point.x, point.y, point.z});
            tags[std::to_string(tag.first)] = points;
        }
        raw["bodies"][item.first] = {{"tag_corners_gripper_m", tags}, {"T_gripper_tcp", transformToJson(item.second.bodyTcp)}};
    }
    return raw;
}

class TempJson {
public:
    explicit TempJson(const Json& json) {
        static unsigned sequence = 0;
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() / ("marker6dof_test_" + std::to_string(tick) + "_" + std::to_string(sequence++) + ".json");
        std::ofstream stream(path_);
        if (!stream) throw std::runtime_error("Cannot create temporary config");
        stream << json.dump();
    }
    ~TempJson() { std::error_code ignored; std::filesystem::remove(path_, ignored); }
    std::string path() const { return path_.string(); }
private:
    std::filesystem::path path_;
};

void testLocalizerPoseAndTcp() {
    auto calibration = syntheticCalibration();
    calibration.camera.distortionModel = "opencv";
    calibration.camera.distortion = cv::Mat(std::vector<double>{-.06, .03, .001, -.0005, -.01}, true);
    const auto expected = knownTransform({.22, -.12, .07}, {.03, -.01, .55});
    Detections detections;
    for (const auto& tag : calibration.bodies.at("tool").corners) detections[tag.first] = projectCorners(tag.second, expected, calibration.camera);
    const auto estimates = MarkerLocalizer(calibration).estimate(detections);
    require(estimates.size() == 1 && estimates.front().status == "MEASURED", "Joint calibrated PnP should measure the body");
    const auto& estimate = estimates.front();
    require(estimate.cameraBody && estimate.cameraTcp, "Accepted estimate must include body and TCP transforms");
    requirePose(*estimate.cameraBody, expected, "Joint body pose must recover the independent projection");
    requirePose(*estimate.cameraTcp, expected * calibration.bodies.at("tool").bodyTcp, "TCP extrinsic must compose in body coordinates");
    require(estimate.usedIds.size() == 3 && estimate.cornerCount == 12 && estimate.inlierCount == 12, "All three clean faces must contribute to joint PnP");
}

void testUnknownDuplicateAndImageSize() {
    const auto calibration = syntheticCalibration();
    const auto expected = knownTransform({.22, -.12, .07}, {.03, -.01, .55});
    auto detections = projectRig(syntheticRig(), expected, calibration.camera);
    MarkerLocalizer localizer(calibration);
    detections[99] = detections.at(2);
    auto estimate = localizer.estimate(detections).front();
    require(estimate.status == "MEASURED" && std::find(estimate.usedIds.begin(), estimate.usedIds.end(), 99) == estimate.usedIds.end(), "Unknown ID must not alter a valid body estimate");
    Detections insufficient{{2, detections.at(2)}, {99, detections.at(99)}};
    require(!localizer.estimate(insufficient).front().cameraTcp, "Unknown ID cannot satisfy minimum registered faces");
    insufficient.erase(99);
    Corners invalid = detections.at(3);
    invalid[0].x = std::numeric_limits<double>::quiet_NaN();
    insufficient[3] = invalid;
    estimate = localizer.estimate(insufficient).front();
    require(!estimate.cameraBody && !estimate.cameraTcp && estimate.reason == "duplicate_marker_id", "Ambiguous duplicate sentinel must not fabricate an accepted pose");
    requireInvalid([&] { localizer.detect(cv::Mat::zeros(32, 32, CV_8UC1)); }, "Mismatched image dimensions must be rejected");
}

void testRigSingleAndMultipleFaces() {
    const auto rig = syntheticRig();
    const auto camera = syntheticCamera();
    const auto expected = knownTransform({.22, -.12, .07}, {.03, -.01, .55});
    const auto detections = projectRig(rig, expected, camera);
    RigTracker tracker(rig);
    auto result = tracker.solve({{2, detections.at(2)}}, camera, 0);
    require(result.status == "SINGLE" && result.pose && result.usedIds == std::vector<int>{2}, "One clean face must be labeled SINGLE");
    requirePose(*result.pose, expected, "Single-face IPPE must recover a clean nonfrontal marker");
    tracker.reset();
    result = tracker.solve(detections, camera, 1);
    require(result.status == "MULTI" && result.pose && result.usedIds.size() == 3, "Multiple compatible faces must be jointly labeled MULTI");
    requirePose(*result.pose, expected, "Multi-face rig must recover the reference pose");
}

void testWholeFaceConsensusAndGeometryGates() {
    const auto rig = syntheticRig();
    const auto camera = syntheticCamera();
    const auto expected = knownTransform({.22, -.12, .07}, {.03, -.01, .55});
    auto detections = projectRig(rig, expected, camera);
    for (auto& corner : detections.at(4)) corner += cv::Point2d(90, -65);
    RigTracker tracker(rig);
    const auto result = tracker.solve(detections, camera, 0);
    require(result.status == "MULTI" && result.pose && result.usedIds == std::vector<int>({2, 3}), "A displaced bad face must not drag two agreeing faces away");
    require(std::find(result.rejectedIds.begin(), result.rejectedIds.end(), 4) != result.rejectedIds.end(), "Rejected whole face must be reported");
    requirePose(*result.pose, expected, "Whole-face consensus must preserve the supported pose");
    const Corners tiny{{{635, 365}, {645, 365}, {645, 355}, {635, 355}}};
    tracker.reset();
    const auto small = tracker.solve({{2, tiny}}, camera, 1);
    require(small.status == "WARN" && !small.pose, "Subthreshold marker must be refused");
    tracker.reset();
    const auto unknown = tracker.solve({{99, tiny}}, camera, 2);
    require(unknown.status == "LOST" && !unknown.pose, "Unregistered faces cannot produce rig pose");
}

void testContinuityRejectsJumpAndRecovers() {
    const auto rig = syntheticRig();
    const auto camera = syntheticCamera();
    const auto start = knownTransform({.22, -.12, .07}, {.03, -.01, .55});
    const auto jump = knownTransform({.22, -.12, .07}, {.18, -.01, .55});
    RigTracker tracker(rig);
    require(tracker.solve(projectRig(rig, start, camera), camera, 0).pose.has_value(), "Initial pose must be accepted");
    const auto rejected = tracker.solve(projectRig(rig, jump, camera), camera, .01);
    require(rejected.status == "WARN" && !rejected.pose, "Short-time 150 mm jump must be rejected");
    const auto recovered = tracker.solve(projectRig(rig, jump, camera), camera, .04);
    require(recovered.pose.has_value(), "Tracker must recover after rejecting and resetting a jump");
    requirePose(*recovered.pose, jump, "Recovered pose must follow the new observation");
}

TrackingResult measurement(const Transform& pose) {
    TrackingResult result;
    result.status = "MULTI";
    result.usedIds = {2, 3};
    result.pose = pose;
    return result;
}

void testRelativeOriginRawAndSmoothedData() {
    RelativeTrace trace(3);
    auto first = trace.update(measurement(knownTransform({0, 0, 0}, {.15, .1, .6})), 0);
    require(first.at("segment") == 1 && std::abs(first.at("position_m").at(0).get<double>()) < 1e-12, "First accepted pose must establish the relative origin");
    auto next = trace.update(measurement(knownTransform({0, 0, .04}, {.16, .1, .6})), 1.0 / 30);
    const double rawX = next.at("position_m").at(0).get<double>();
    const double smoothX = next.at("smooth_position_m").at(0).get<double>();
    require(std::abs(rawX - .01) < 1e-12 && smoothX > 0 && smoothX < rawX, "Smoothing must be separate from the measured 10 mm displacement");
    require(std::abs(next.at("quaternion_xyzw").at(2).get<double>() - std::sin(.02)) < 1e-12, "Raw quaternion must preserve measured relative rotation");
    require(std::abs(next.at("smooth_quaternion_xyzw").at(2).get<double>()) < std::abs(next.at("quaternion_xyzw").at(2).get<double>()), "Display quaternion must smooth independently");
    TrackingResult lost;
    auto missing = trace.update(lost, .07);
    require(missing.at("position_m").is_null() && missing.at("quaternion_xyzw").is_null(), "Loss must not fabricate a pose");
    auto resumed = trace.update(measurement(knownTransform({0, 0, 0}, {.17, .1, .6})), .1);
    require(resumed.at("segment") == 2 && std::abs(resumed.at("position_m").at(0).get<double>() - .02) < 1e-12, "Loss must split the trace while preserving the original origin");
    require(resumed.at("position_m") == resumed.at("smooth_position_m"), "New segment must restart display smoothing from measured pose");
    auto backwards = trace.update(measurement(knownTransform({0, 0, 0}, {.175, .1, .6})), .09);
    require(backwards.at("segment") == 3, "Nonmonotonic timestamp must split the trace");
    auto gap = trace.update(measurement(knownTransform({0, 0, 0}, {.18, .1, .6})), .5);
    require(gap.at("segment") == 4 && trace.trace().size() == 3, "Long gap must split the trace and retention must remain bounded");
    trace.reset();
    auto reset = trace.update(measurement(knownTransform({0, 0, 0}, {.18, .1, .6})), 1);
    require(reset.at("segment") == 1 && std::abs(reset.at("position_m").at(0).get<double>()) < 1e-12, "Explicit reset must establish a new origin");
}

void testInvalidCalibrationAndRig() {
    Json raw = calibrationJson();
    TempJson valid(raw);
    require(loadCalibration(valid.path()).bodies.size() == 1, "Valid metric calibration must load");
    raw["units"] = "mm";
    TempJson nonmetric(raw);
    requireInvalid([&] { loadCalibration(nonmetric.path()); }, "Nonmetric calibration must be rejected");
    raw = calibrationJson();
    raw["bodies"]["another"] = raw["bodies"]["tool"];
    TempJson duplicate(raw);
    requireInvalid([&] { loadCalibration(duplicate.path()); }, "A marker assigned to two bodies must be rejected");
    raw = calibrationJson();
    raw["bodies"]["tool"]["tag_corners_gripper_m"]["2"] = Json::array({{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
    TempJson degenerate(raw);
    requireInvalid([&] { loadCalibration(degenerate.path()); }, "Degenerate face calibration must be rejected");
    raw = calibrationJson();
    raw["camera_matrix"][0][0] = 0;
    TempJson badFocal(raw);
    requireInvalid([&] { loadCalibration(badFocal.path()); }, "Zero focal length must be rejected");
    requireInvalid([] { markerObjectPoints(0); }, "Zero marker size must be rejected");
    requireInvalid([] { markerObjectPoints(std::numeric_limits<double>::quiet_NaN()); }, "Nonfinite marker size must be rejected");
    Json rig{{"reference_id", 2}, {"marker_size_m", .048}, {"T_reference_tag", {{"2", transformToJson(Transform::eye())}}}};
    TempJson rigFile(rig);
    require(loadRig(rigFile.path(), .048).referenceTag.size() == 1, "Matching rig size must load");
    requireInvalid([&] { loadRig(rigFile.path(), .05); }, "Changing only marker size must reject the calibrated rig");
    rig["reference_id"] = 9;
    TempJson missingReference(rig);
    requireInvalid([&] { loadRig(missingReference.path(), .048); }, "Missing reference face must reject the rig");
    auto reflection = Transform::eye();
    reflection(0, 0) = -1;
    requireInvalid([&] { transformFromJson(transformToJson(reflection)); }, "A reflected coordinate frame must not be accepted as rigid rotation");
}

void testRealSenseInverseBrownSdkSamples() {
    Camera camera = syntheticCamera();
    camera.matrix = {646.4898071289062, 0, 641.2659912109375,
                     0, 645.6320190429688, 351.31304931640625, 0, 0, 1};
    camera.distortionModel = "inverse_brown_conrady";
    camera.distortion = cv::Mat(std::vector<double>{-.053984999656677246, .06133309006690979,
                         .00021358334925025702, .0007035458693280816, -.019717831164598465}, true);
    // Independent values sampled from pyrealsense2 rs2_* SDK functions with
    // the actual original D455 metadata. The SDK uses float arithmetic.
    const std::vector<cv::Point2d> pixels{{0, 0}, {1279, 719}, {641.2659912109375, 351.31304931640625}, {200, 100}, {1100, 600}};
    const std::vector<cv::Point2d> expectedNormalized{{-1.004292368888855, -.5506967306137085}, {.99362713098526, .5738946795463562}, {0, 0}, {-.6939069032669067, -.39560240507125854}, {.7187556624412537, .3902806341648102}};
    const auto normalized = normalizePixels(camera, pixels);
    require(normalized.size() == pixels.size(), "Normalization must preserve sample count");
    for (std::size_t index = 0; index < normalized.size(); ++index) require(cv::norm(normalized[index] - expectedNormalized[index]) < 2e-6, "Inverse Brown normalization must match official SDK");
    const std::vector<cv::Point3d> points{{0, 0, 1}, {.1, -.05, .5}, {-.2, .15, .7}, {.45, .28, .6}};
    const std::vector<cv::Point2d> expectedPixels{{641.2659912109375, 351.31304931640625}, {770.2879028320312, 286.90594482421875}, {457.7644348144531, 488.8176574707031}, {1120.21923828125, 648.820068359375}};
    const auto projected = projectCamera(camera, points);
    require(projected.size() == points.size(), "Projection must preserve sample count");
    for (std::size_t index = 0; index < projected.size(); ++index) require(cv::norm(projected[index] - expectedPixels[index]) < 1e-3, "Inverse Brown projection must match official SDK");
}
void testRecordedClockFallbackAndInvalidTime() {
    TempJson csvFile(Json::object());
    const auto write = [&](const std::string& text) {
        std::ofstream output(csvFile.path());
        output << text;
    };
    write("frame_index,device_timestamp_ms,host_timestamp_s\n0,1000,10\n1,1000,10.03\n2,1033,10.06\n");
    auto times = loadFrameTimes("", 30, csvFile.path());
    require(times.source.find("host_timestamp_s") != std::string::npos &&
            times.source.find("fallback") != std::string::npos,
            "Repeated device clock must use and declare the recorded monotonic host clock");
    require(std::abs(times.at(1) - .03) < 1e-10 && std::abs(times.at(2) - .06) < 1e-10,
            "Fallback must preserve the measured host clock intervals");
    write("frame_index,device_timestamp_ms,host_timestamp_s\n0,1000,10\n1,1033,10\n2,1066,10.06\n");
    times = loadFrameTimes("", 30, csvFile.path());
    require(times.source.find("device_timestamp_ms") != std::string::npos &&
            std::abs(times.at(1) - .033) < 1e-10,
            "A valid device clock must stay preferred even if optional host clock is invalid");
    write("frame_index,device_timestamp_ms,host_timestamp_s\n0,1000,10\n1,1000,10\n");
    requireInvalid([&] { loadFrameTimes("", 30, csvFile.path()); },
                   "Two non-increasing clocks must not invent a valid temporal interval");
    write("frame_index,timestamp_s,host_timestamp_s\n0,0,10\n1,0,10.03\n");
    requireInvalid([&] { loadFrameTimes("", 30, csvFile.path()); },
                   "Explicit repeated timestamp_s must be rejected");
    write("frame_index,timestamp_s\n0,0\n2,.06\n");
    requireInvalid([&] { loadFrameTimes("", 30, csvFile.path()); },
                   "Missing frame timestamp must be rejected");
}
} // namespace

int main() {
    cv::setNumThreads(1);
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"calibrated joint pose and TCP composition", testLocalizerPoseAndTcp},
        {"unknown IDs, duplicate ambiguity and image dimensions", testUnknownDuplicateAndImageSize},
        {"single and multi-face rig pose", testRigSingleAndMultipleFaces},
        {"whole-face consensus and geometry quality gates", testWholeFaceConsensusAndGeometryGates},
        {"continuity jump rejection and recovery", testContinuityRejectsJumpAndRecovers},
        {"relative origin, raw/smooth outputs and trajectory segmentation", testRelativeOriginRawAndSmoothedData},
        {"metric calibration, rig size and geometry validation", testInvalidCalibrationAndRig},
        {"official RealSense inverse Brown projection samples", testRealSenseInverseBrownSdkSamples},
        {"recorded clock fallback and invalid temporal input", testRecordedClockFallbackAndInvalidTime},
    };
    int failed = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS: " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "FAIL: " << test.first << ": " << error.what() << '\n';
        }
    }
    std::cout << tests.size() - failed << '/' << tests.size() << " test groups passed\n";
    return failed ? 1 : 0;
}
