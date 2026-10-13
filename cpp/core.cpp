#include "marker6dof/core.hpp"

#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
#include <opencv2/objdetect/aruco_detector.hpp>
#else
#include <opencv2/aruco.hpp>
#endif
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace marker6dof {
namespace {
constexpr double pi = 3.14159265358979323846;

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument(message);
}

double finiteNumber(const Json& value, const std::string& name) {
    if (!value.is_number()) invalid(name + " must be a finite number");
    const double number = value.get<double>();
    if (!std::isfinite(number)) invalid(name + " must be a finite number");
    return number;
}

int integer(const Json& value, const std::string& name) {
    const double number = finiteNumber(value, name);
    if (std::trunc(number) != number ||
        number < std::numeric_limits<int>::min() ||
        number > std::numeric_limits<int>::max()) {
        invalid(name + " must be an integer");
    }
    return static_cast<int>(number);
}

int tagId(const std::string& value) {
    std::size_t end = 0;
    long number = 0;
    try {
        number = std::stol(value, &end);
    } catch (const std::exception&) {
        invalid("invalid marker ID: " + value);
    }
    if (end != value.size() || number < 0 || number > std::numeric_limits<int>::max()) {
        invalid("invalid marker ID: " + value);
    }
    return static_cast<int>(number);
}

void requireObject(const Json& value, const std::string& name) {
    if (!value.is_object()) invalid(name + " must be an object");
}

template <int Rows, int Cols>
cv::Matx<double, Rows, Cols> matrixFromJson(const Json& value, const std::string& name) {
    if (!value.is_array() || value.size() != Rows) {
        invalid(name + " must have shape " + std::to_string(Rows) + "x" + std::to_string(Cols));
    }
    cv::Matx<double, Rows, Cols> matrix;
    for (int row = 0; row < Rows; ++row) {
        if (!value[row].is_array() || value[row].size() != Cols) {
            invalid(name + " must have shape " + std::to_string(Rows) + "x" + std::to_string(Cols));
        }
        for (int col = 0; col < Cols; ++col) {
            matrix(row, col) = finiteNumber(value[row][col], name);
        }
    }
    return matrix;
}

template <int Rows, int Cols>
Json matrixToJson(const cv::Matx<double, Rows, Cols>& matrix) {
    Json result = Json::array();
    for (int row = 0; row < Rows; ++row) {
        Json values = Json::array();
        for (int col = 0; col < Cols; ++col) values.push_back(matrix(row, col));
        result.push_back(std::move(values));
    }
    return result;
}

template <int Count>
Json vectorToJson(const cv::Vec<double, Count>& vector) {
    Json result = Json::array();
    for (int index = 0; index < Count; ++index) result.push_back(vector[index]);
    return result;
}

void flattenNumbers(const Json& value, std::vector<double>& output, const std::string& name) {
    if (value.is_array()) {
        for (const auto& child : value) flattenNumbers(child, output, name);
    } else {
        output.push_back(finiteNumber(value, name));
    }
}

cv::Matx33d rotation(const Transform& transform) {
    return {transform(0, 0), transform(0, 1), transform(0, 2),
            transform(1, 0), transform(1, 1), transform(1, 2),
            transform(2, 0), transform(2, 1), transform(2, 2)};
}

bool finiteTransform(const Transform& transform) {
    for (double value : transform.val) if (!std::isfinite(value)) return false;
    return true;
}

bool finiteCorners(const Corners& corners) {
    for (const auto& corner : corners) {
        if (!std::isfinite(corner.x) || !std::isfinite(corner.y)) return false;
    }
    return true;
}

bool positiveDepth(const Transform& transform, const std::vector<cv::Point3d>& points,
                   double minimum = 0.0) {
    if (!finiteTransform(transform)) return false;
    for (const auto& point : points) {
        const auto cameraPoint = transformPoint(transform, point);
        if (!std::isfinite(cameraPoint.z) || cameraPoint.z <= minimum) return false;
    }
    return true;
}

std::vector<cv::Point2d> project(const Transform& pose,
                               const std::vector<cv::Point3d>& points,
                               const Camera& camera) {
    std::vector<cv::Point2d> projected;
    cv::projectPoints(points, rotationVector(pose), translation(pose),
                      camera.matrix, camera.distortion, projected);
    return projected;
}

double reprojectionRms(const Transform& pose, const std::vector<cv::Point3d>& points,
                       const std::vector<cv::Point2d>& pixels, const Camera& camera) {
    const auto projected = project(pose, points, camera);
    double sum = 0;
    for (std::size_t index = 0; index < projected.size(); ++index) {
        const auto difference = projected[index] - pixels[index];
        sum += difference.dot(difference);
    }
    return std::sqrt(sum / projected.size());
}

cv::Vec3d matVector(const cv::Mat& input) {
    cv::Mat converted;
    input.convertTo(converted, CV_64F);
    converted = converted.reshape(1, 3);
    return {converted.at<double>(0, 0), converted.at<double>(1, 0), converted.at<double>(2, 0)};
}

double rotationDistance(const Transform& first, const Transform& second) {
    cv::Vec3d vector;
    cv::Rodrigues(rotation(first).t() * rotation(second), vector);
    return cv::norm(vector);
}

Json optionalNumber(const std::optional<double>& number) {
    return number && std::isfinite(*number) ? Json(*number) : Json(nullptr);
}

Json optionalTransform(const std::optional<Transform>& transform) {
    return transform ? transformToJson(*transform) : Json(nullptr);
}

void appendPoseFields(Json& result, const std::optional<Transform>& pose) {
    for (const char* name : {"tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw"}) {
        result[name] = nullptr;
    }
    if (!pose) return;
    const cv::Vec3d position = translation(*pose);
    const cv::Vec4d quaternion = quaternionXyzw(*pose);
    result["tx_m"] = position[0];
    result["ty_m"] = position[1];
    result["tz_m"] = position[2];
    result["qx"] = quaternion[0];
    result["qy"] = quaternion[1];
    result["qz"] = quaternion[2];
    result["qw"] = quaternion[3];
}
} // namespace

Json readJson(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open JSON file: " + path);
    try {
        return Json::parse(stream);
    } catch (const Json::exception& error) {
        throw std::invalid_argument("invalid JSON in " + path + ": " + error.what());
    }
}

Camera cameraFromJson(const Json& value) {
    requireObject(value, "camera");
    Camera camera;
    camera.width = integer(value.at("image_width"), "image_width");
    camera.height = integer(value.at("image_height"), "image_height");
    if (camera.width <= 0 || camera.height <= 0) invalid("image dimensions must be positive");
    camera.matrix = matrixFromJson<3, 3>(value.at("camera_matrix"), "camera_matrix");
    if (camera.matrix(0, 0) <= 0 || camera.matrix(1, 1) <= 0 ||
        std::abs(camera.matrix(2, 0)) > 1e-9 ||
        std::abs(camera.matrix(2, 1)) > 1e-9 ||
        std::abs(camera.matrix(2, 2) - 1) > 1e-9) {
        invalid("camera_matrix must contain positive focal lengths and K[2]=[0,0,1]");
    }
    if (camera.matrix(0, 2) < 0 || camera.matrix(0, 2) >= camera.width ||
        camera.matrix(1, 2) < 0 || camera.matrix(1, 2) >= camera.height) {
        invalid("camera principal point must lie inside the configured image");
    }
    const auto& rawDistortion = value.at("distortion");
    if (!rawDistortion.is_array()) invalid("distortion must be an array");
    std::vector<double> coefficients;
    flattenNumbers(rawDistortion, coefficients, "distortion");
    const std::set<std::size_t> sizes{4, 5, 8, 12, 14};
    if (!sizes.count(coefficients.size())) {
        invalid("distortion must contain 4, 5, 8, 12, or 14 finite coefficients");
    }
    camera.distortion = cv::Mat(coefficients, true).reshape(1, static_cast<int>(coefficients.size()));
    camera.distortionModel = value.value("distortion_model", std::string("opencv"));
    const std::set<std::string> models{"opencv", "none", "brown_conrady", "inverse_brown_conrady"};
    if (!models.count(camera.distortionModel)) invalid("unsupported distortion_model: " + camera.distortionModel);
    if (camera.distortionModel != "opencv" && camera.distortionModel != "none" && coefficients.size() != 5) {
        invalid("RealSense Brown-Conrady models require 5 distortion coefficients");
    }
    if (camera.distortionModel == "none") camera.distortion.setTo(0);
    return camera;
}

Json cameraToJson(const Camera& camera) {
    Json coefficients = Json::array();
    if (!camera.distortion.empty()) {
        cv::Mat distortion;
        camera.distortion.convertTo(distortion, CV_64F);
        distortion = distortion.reshape(1, static_cast<int>(distortion.total()));
        for (int index = 0; index < distortion.rows; ++index) {
            coefficients.push_back(distortion.at<double>(index, 0));
        }
    }
    return {{"image_width", camera.width}, {"image_height", camera.height},
            {"camera_matrix", matrixToJson(camera.matrix)}, {"distortion", coefficients},
            {"distortion_model", camera.distortionModel}};
}

Transform transformFromJson(const Json& value, const std::string& name) {
    const Transform transform = matrixFromJson<4, 4>(value, name);
    for (int col = 0; col < 4; ++col) {
        if (std::abs(transform(3, col) - (col == 3 ? 1.0 : 0.0)) > 1e-9) {
            invalid(name + " must have homogeneous last row [0, 0, 0, 1]");
        }
    }
    const auto matrix = rotation(transform);
    const auto residual = matrix.t() * matrix - cv::Matx33d::eye();
    for (double value : residual.val) {
        if (std::abs(value) > 1e-5) invalid(name + " rotation must be orthonormal");
    }
    if (std::abs(cv::determinant(matrix) - 1.0) > 1e-5) {
        invalid(name + " rotation determinant must be +1");
    }
    return transform;
}

Json transformToJson(const Transform& transform) {
    return matrixToJson(transform);
}

Calibration loadCalibration(const std::string& path) {
    const Json raw = readJson(path);
    requireObject(raw, "calibration");
    if (!raw.contains("schema_version") || integer(raw.at("schema_version"), "schema_version") != 1) {
        invalid("calibration schema_version must be 1");
    }
    if (!raw.contains("units") || raw.at("units") != "m") invalid("calibration units must be 'm'");
    Calibration calibration;
    calibration.camera = cameraFromJson(raw);
    if (calibration.camera.distortionModel != "opencv" &&
        calibration.camera.distortionModel != "brown_conrady" &&
        calibration.camera.distortionModel != "none") {
        invalid("body calibration requires an OpenCV-compatible camera distortion model");
    }
    if (calibration.camera.distortionModel == "none") calibration.camera.distortion.setTo(0);
    const Json aruco = raw.value("aruco", Json::object());
    requireObject(aruco, "aruco");
    calibration.dictionary = aruco.value("dictionary", std::string());
    dictionaryId(calibration.dictionary);
    const Json bodies = raw.value("bodies", Json::object());
    requireObject(bodies, "bodies");
    std::map<int, std::string> owners;
    for (const auto& entry : bodies.items()) {
        requireObject(entry.value(), "body " + entry.key());
        Body body;
        body.name = entry.key();
        const Json tags = entry.value().value("tag_corners_gripper_m", Json::object());
        requireObject(tags, body.name + ".tag_corners_gripper_m");
        for (const auto& tag : tags.items()) {
            const int id = tagId(tag.key());
            if (owners.count(id)) {
                invalid("tag " + std::to_string(id) + " belongs to both " + owners.at(id) + " and " + body.name);
            }
            const std::string name = body.name + ".tag[" + std::to_string(id) + "]";
            const auto coordinates = matrixFromJson<4, 3>(tag.value(), name);
            std::array<cv::Point3d, 4> corners;
            cv::Vec3d center(0, 0, 0);
            for (int index = 0; index < 4; ++index) {
                corners[index] = {coordinates(index, 0), coordinates(index, 1), coordinates(index, 2)};
                center += cv::Vec3d(coordinates(index, 0), coordinates(index, 1), coordinates(index, 2)) / 4.0;
            }
            cv::Mat centered(4, 3, CV_64F);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 3; ++col) centered.at<double>(row, col) = coordinates(row, col) - center[col];
            }
            cv::Mat singularValues;
            cv::SVD::compute(centered, singularValues);
            const double tolerance = singularValues.at<double>(0) * 4 * std::numeric_limits<double>::epsilon();
            int rank = 0;
            for (int index = 0; index < 3; ++index) rank += singularValues.at<double>(index) > tolerance;
            if (rank != 2) invalid(name + " corners must define one non-degenerate plane");
            for (int index = 0; index < 4; ++index) {
                if (cv::norm(corners[index] - corners[(index + 3) % 4]) <= 1e-5) {
                    invalid(name + " has a degenerate edge");
                }
            }
            if (!body.corners.emplace(id, corners).second) invalid("duplicate marker ID " + std::to_string(id));
            owners[id] = body.name;
        }
        if (body.corners.empty()) invalid("body " + body.name + " must define at least one tag");
        body.bodyTcp = transformFromJson(entry.value().at("T_gripper_tcp"), body.name + ".T_gripper_tcp");
        calibration.bodies.emplace(body.name, std::move(body));
    }
    if (calibration.bodies.empty()) invalid("calibration must define at least one body");
    const Json quality = raw.value("quality", Json::object());
    requireObject(quality, "quality");
    auto numberOption = [&](const char* name, double defaultValue) {
        return quality.contains(name) ? finiteNumber(quality.at(name), std::string("quality.") + name) : defaultValue;
    };
    auto integerOption = [&](const char* name, int defaultValue) {
        return quality.contains(name) ? integer(quality.at(name), std::string("quality.") + name) : defaultValue;
    };
    auto& config = calibration.quality;
    config.minMarkers = integerOption("min_markers", config.minMarkers);
    config.minInlierCorners = integerOption("min_inlier_corners", config.minInlierCorners);
    config.ransacThresholdPx = numberOption("ransac_threshold_px", config.ransacThresholdPx);
    config.ransacConfidence = numberOption("ransac_confidence", config.ransacConfidence);
    config.ransacIterations = integerOption("ransac_iterations", config.ransacIterations);
    config.maxRmsePx = numberOption("max_reprojection_rmse_px", config.maxRmsePx);
    config.maxErrorPx = numberOption("max_reprojection_error_px", config.maxErrorPx);
    if (config.minMarkers < 1 || config.minInlierCorners < 4) {
        invalid("quality min_markers must be >=1 and min_inlier_corners must be >=4");
    }
    if (config.ransacConfidence <= 0 || config.ransacConfidence >= 1) {
        invalid("quality ransac_confidence must be in (0,1)");
    }
    if (config.ransacIterations < 1 || config.ransacThresholdPx <= 0 ||
        config.maxRmsePx <= 0 || config.maxErrorPx <= 0) {
        invalid("quality iterations and reprojection thresholds must be positive");
    }
    return calibration;
}

Rig loadRig(const std::string& path, double markerSizeM) {
    markerObjectPoints(markerSizeM);
    const Json raw = readJson(path);
    requireObject(raw, "rig");
    Rig rig;
    rig.referenceId = integer(raw.at("reference_id"), "reference_id");
    if (rig.referenceId < 0) invalid("reference_id must be non-negative");
    rig.markerSizeM = markerSizeM;
    const Json& transforms = raw.at("T_reference_tag");
    requireObject(transforms, "T_reference_tag");
    for (const auto& tag : transforms.items()) {
        const int id = tagId(tag.key());
        const Transform transform = transformFromJson(tag.value(), "rig transform for ID " + std::to_string(id));
        if (!rig.referenceTag.emplace(id, transform).second) invalid("duplicate marker ID " + std::to_string(id));
    }
    if (!rig.referenceTag.count(rig.referenceId)) invalid("Reference ID is missing from rig");
    if (raw.contains("marker_size_m")) {
        const double size = finiteNumber(raw.at("marker_size_m"), "marker_size_m");
        if (size <= 0 || std::abs(size - markerSizeM) > 1e-4 * std::max(size, markerSizeM)) {
            invalid("Marker size differs from rig calibration; recalibrate the rig first");
        }
    }
    return rig;
}

Transform inverseRigid(const Transform& transform) {
    const cv::Matx33d inverseRotation = rotation(transform).t();
    const cv::Vec3d inverseTranslation = -(inverseRotation * translation(transform));
    Transform inverse = Transform::eye();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) inverse(row, col) = inverseRotation(row, col);
        inverse(row, 3) = inverseTranslation[row];
    }
    return inverse;
}

cv::Point3d transformPoint(const Transform& transform, const cv::Point3d& point) {
    const cv::Vec3d result = rotation(transform) * cv::Vec3d(point.x, point.y, point.z) + translation(transform);
    return {result[0], result[1], result[2]};
}

Transform makeTransform(const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
    cv::Matx33d matrix;
    cv::Rodrigues(rvec, matrix);
    Transform transform = Transform::eye();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) transform(row, col) = matrix(row, col);
        transform(row, 3) = tvec[row];
    }
    return transform;
}

cv::Vec3d rotationVector(const Transform& transform) {
    cv::Vec3d vector;
    cv::Rodrigues(rotation(transform), vector);
    return vector;
}

cv::Vec3d translation(const Transform& transform) {
    return {transform(0, 3), transform(1, 3), transform(2, 3)};
}

cv::Vec4d quaternionXyzw(const Transform& transform) {
    const auto matrix = rotation(transform);
    const double trace = matrix(0, 0) + matrix(1, 1) + matrix(2, 2);
    cv::Vec4d quaternion;
    if (trace > 0) {
        const double scale = 2.0 * std::sqrt(trace + 1.0);
        quaternion = {(matrix(2, 1) - matrix(1, 2)) / scale,
                      (matrix(0, 2) - matrix(2, 0)) / scale,
                      (matrix(1, 0) - matrix(0, 1)) / scale, 0.25 * scale};
    } else {
        int index = 0;
        for (int i = 1; i < 3; ++i) if (matrix(i, i) > matrix(index, index)) index = i;
        if (index == 0) {
            const double scale = 2 * std::sqrt(std::max(1 + matrix(0, 0) - matrix(1, 1) - matrix(2, 2), 1e-15));
            quaternion = {0.25 * scale, (matrix(0, 1) + matrix(1, 0)) / scale,
                          (matrix(0, 2) + matrix(2, 0)) / scale, (matrix(2, 1) - matrix(1, 2)) / scale};
        } else if (index == 1) {
            const double scale = 2 * std::sqrt(std::max(1 + matrix(1, 1) - matrix(0, 0) - matrix(2, 2), 1e-15));
            quaternion = {(matrix(0, 1) + matrix(1, 0)) / scale, 0.25 * scale,
                          (matrix(1, 2) + matrix(2, 1)) / scale, (matrix(0, 2) - matrix(2, 0)) / scale};
        } else {
            const double scale = 2 * std::sqrt(std::max(1 + matrix(2, 2) - matrix(0, 0) - matrix(1, 1), 1e-15));
            quaternion = {(matrix(0, 2) + matrix(2, 0)) / scale,
                          (matrix(1, 2) + matrix(2, 1)) / scale, 0.25 * scale,
                          (matrix(1, 0) - matrix(0, 1)) / scale};
        }
    }
    return quaternion / cv::norm(quaternion);
}

std::array<cv::Point3d, 4> markerObjectPoints(double sizeM) {
    if (!std::isfinite(sizeM) || sizeM <= 0) invalid("marker_size_m must be positive and finite");
    const double half = sizeM / 2;
    return {{{-half, half, 0}, {half, half, 0}, {half, -half, 0}, {-half, -half, 0}}};
}

int dictionaryId(const std::string& name) {
    static const std::map<std::string, int> dictionaries{
        {"DICT_4X4_50", cv::aruco::DICT_4X4_50}, {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
        {"DICT_4X4_250", cv::aruco::DICT_4X4_250}, {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
        {"DICT_5X5_50", cv::aruco::DICT_5X5_50}, {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
        {"DICT_5X5_250", cv::aruco::DICT_5X5_250}, {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
        {"DICT_6X6_50", cv::aruco::DICT_6X6_50}, {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
        {"DICT_6X6_250", cv::aruco::DICT_6X6_250}, {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
        {"DICT_7X7_50", cv::aruco::DICT_7X7_50}, {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
        {"DICT_7X7_250", cv::aruco::DICT_7X7_250}, {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
        {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
        {"DICT_APRILTAG_16h5", cv::aruco::DICT_APRILTAG_16h5},
        {"DICT_APRILTAG_25h9", cv::aruco::DICT_APRILTAG_25h9},
        {"DICT_APRILTAG_36h10", cv::aruco::DICT_APRILTAG_36h10},
        {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11}};
    const auto found = dictionaries.find(name);
    if (found == dictionaries.end()) invalid("unsupported OpenCV ArUco dictionary: " + name);
    return found->second;
}

Detections detectMarkers(const cv::Mat& image, const std::string& dictionary) {
    if (image.empty()) invalid("image must not be empty");
    cv::Mat gray;
    if (image.channels() == 1) gray = image;
    else if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if (image.channels() == 4) cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else invalid("image must have 1, 3, or 4 channels");
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
    cv::aruco::DetectorParameters parameters;
    parameters.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    const cv::aruco::ArucoDetector detector(
        cv::aruco::getPredefinedDictionary(dictionaryId(dictionary)), parameters);
    detector.detectMarkers(gray, corners, ids);
#else
    auto parameters = cv::aruco::DetectorParameters::create();
    parameters->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    cv::aruco::detectMarkers(gray, cv::aruco::getPredefinedDictionary(dictionaryId(dictionary)),
                            corners, ids, parameters);
#endif
    Detections detections;
    for (std::size_t index = 0; index < ids.size(); ++index) {
        Corners pixels;
        for (int corner = 0; corner < 4; ++corner) pixels[corner] = corners[index][corner];
        if (!detections.emplace(ids[index], pixels).second) {
            // The fixed public API uses a map. Keep a duplicate ID present but
            // unusable so it can never silently overwrite an observed marker.
            const double nan = std::numeric_limits<double>::quiet_NaN();
            detections[ids[index]].fill(cv::Point2d(nan, nan));
        }
    }
    return detections;
}

Json poseToJson(const PoseEstimate& pose) {
    Json result{{"body", pose.body}, {"status", pose.status}, {"reason", pose.reason},
                {"visible_ids", pose.visibleIds}, {"used_ids", pose.usedIds},
                {"corner_count", pose.cornerCount}, {"inlier_count", pose.inlierCount},
                {"reprojection_rmse_px", optionalNumber(pose.rmsePx)},
                {"max_reprojection_error_px", optionalNumber(pose.maxErrorPx)},
                {"T_camera_gripper", optionalTransform(pose.cameraBody)},
                {"T_camera_tcp", optionalTransform(pose.cameraTcp)}};
    appendPoseFields(result, pose.cameraTcp);
    return result;
}

Json trackingToJson(const TrackingResult& result) {
    Json tagErrors = Json::object();
    for (const auto& entry : result.tagErrors) tagErrors[std::to_string(entry.first)] = optionalNumber(entry.second);
    return {{"status", result.status}, {"reason", result.reason},
            {"visible_ids", result.visibleIds}, {"used_ids", result.usedIds},
            {"rejected_ids", result.rejectedIds},
            {"reprojection_error_px", optionalNumber(result.reprojectionErrorPx)},
            {"tag_errors", tagErrors}, {"pose", optionalTransform(result.pose)}};
}

MarkerLocalizer::MarkerLocalizer(Calibration calibration) : calibration_(std::move(calibration)) {
    dictionaryId(calibration_.dictionary);
    if (calibration_.camera.distortionModel != "opencv" &&
        calibration_.camera.distortionModel != "brown_conrady" &&
        calibration_.camera.distortionModel != "none") {
        invalid("body calibration requires an OpenCV-compatible camera distortion model");
    }
    if (calibration_.camera.distortionModel == "none") {
        calibration_.camera.distortion = cv::Mat::zeros(5, 1, CV_64F);
    }
}

const Calibration& MarkerLocalizer::calibration() const { return calibration_; }

Detections MarkerLocalizer::detect(const cv::Mat& image) const {
    if (image.empty()) invalid("image must not be empty");
    if (image.cols != calibration_.camera.width || image.rows != calibration_.camera.height) {
        invalid("image size does not match calibration");
    }
    return detectMarkers(image, calibration_.dictionary);
}

std::vector<PoseEstimate> MarkerLocalizer::estimate(const Detections& detections) const {
    std::vector<PoseEstimate> estimates;
    const auto& quality = calibration_.quality;
    const auto& camera = calibration_.camera;
    for (const auto& bodyEntry : calibration_.bodies) {
        const Body& body = bodyEntry.second;
        PoseEstimate result;
        result.body = body.name;
        bool duplicate = false;
        std::vector<cv::Point3d> objectPoints;
        std::vector<cv::Point2d> imagePoints;
        for (const auto& detection : detections) {
            result.visibleIds.push_back(detection.first);
            if (!finiteCorners(detection.second)) {
                duplicate = true;
                continue;
            }
            const auto registered = body.corners.find(detection.first);
            if (registered == body.corners.end()) continue;
            result.usedIds.push_back(detection.first);
            objectPoints.insert(objectPoints.end(), registered->second.begin(), registered->second.end());
            imagePoints.insert(imagePoints.end(), detection.second.begin(), detection.second.end());
        }
        if (static_cast<int>(result.usedIds.size()) < quality.minMarkers) {
            result.reason = duplicate ? "duplicate_marker_id" : "insufficient_registered_markers";
            estimates.push_back(std::move(result));
            continue;
        }
        result.cornerCount = static_cast<int>(objectPoints.size());
        cv::Vec3d rvec, tvec;
        std::vector<int> inliers;
        bool solved = false;
        try {
            if (objectPoints.size() == 4) {
                std::vector<cv::Mat> rotations, translations;
                cv::solvePnPGeneric(objectPoints, imagePoints, camera.matrix, camera.distortion,
                                    rotations, translations, false, cv::SOLVEPNP_IPPE);
                double bestError = std::numeric_limits<double>::infinity();
                for (std::size_t index = 0; index < rotations.size(); ++index) {
                    const auto candidateRvec = matVector(rotations[index]);
                    const auto candidateTvec = matVector(translations[index]);
                    const auto pose = makeTransform(candidateRvec, candidateTvec);
                    if (!positiveDepth(pose, objectPoints)) continue;
                    const double error = reprojectionRms(pose, objectPoints, imagePoints, camera);
                    if (error < bestError) {
                        bestError = error;
                        rvec = candidateRvec;
                        tvec = candidateTvec;
                        solved = true;
                    }
                }
                if (solved) inliers = {0, 1, 2, 3};
            } else {
                cv::Mat inlierMatrix;
                solved = cv::solvePnPRansac(objectPoints, imagePoints, camera.matrix, camera.distortion,
                                           rvec, tvec, false, quality.ransacIterations,
                                           static_cast<float>(quality.ransacThresholdPx), quality.ransacConfidence,
                                           inlierMatrix, cv::SOLVEPNP_EPNP);
                if (solved && inlierMatrix.total() >= 4) {
                    std::vector<cv::Point3d> inlierObjects;
                    std::vector<cv::Point2d> inlierImages;
                    for (int row = 0; row < inlierMatrix.rows; ++row) {
                        const int index = inlierMatrix.at<int>(row, 0);
                        inliers.push_back(index);
                        inlierObjects.push_back(objectPoints[index]);
                        inlierImages.push_back(imagePoints[index]);
                    }
                    cv::solvePnPRefineLM(inlierObjects, inlierImages, camera.matrix, camera.distortion, rvec, tvec);
                } else solved = false;
            }
        } catch (const cv::Exception&) {
            solved = false;
        }
        if (!solved) {
            result.reason = "pnp_failed";
            estimates.push_back(std::move(result));
            continue;
        }
        result.inlierCount = static_cast<int>(inliers.size());
        if (result.inlierCount < quality.minInlierCorners) {
            result.reason = "too_few_inlier_corners";
            estimates.push_back(std::move(result));
            continue;
        }
        const Transform pose = makeTransform(rvec, tvec);
        if (!positiveDepth(pose, objectPoints)) {
            result.reason = "non_positive_depth";
            estimates.push_back(std::move(result));
            continue;
        }
        const auto projected = project(pose, objectPoints, camera);
        double squaredError = 0;
        double maxError = 0;
        for (std::size_t index = 0; index < projected.size(); ++index) {
            const double error = cv::norm(projected[index] - imagePoints[index]);
            squaredError += error * error;
            maxError = std::max(maxError, error);
        }
        const double rmse = std::sqrt(squaredError / objectPoints.size());
        result.rmsePx = rmse;
        result.maxErrorPx = maxError;
        if (!std::isfinite(rmse) || rmse > quality.maxRmsePx) result.reason = "reprojection_rmse_rejected";
        else if (!std::isfinite(maxError) || maxError > quality.maxErrorPx) result.reason = "reprojection_max_rejected";
        else {
            result.status = "MEASURED";
            result.reason = "ok";
            result.cameraBody = pose;
            result.cameraTcp = pose * body.bodyTcp;
        }
        estimates.push_back(std::move(result));
    }
    return estimates;
}

struct RigTracker::Impl {
    Rig rig;
    std::array<cv::Point3d, 4> localPoints;
    std::map<int, std::array<cv::Point3d, 4>> points;
    double maxError;
    double minEdge;
    std::optional<Transform> previous;
    std::optional<double> previousTime;

    Impl(Rig configured, double maxErrorPx, double minEdgePx)
        : rig(std::move(configured)), localPoints(markerObjectPoints(rig.markerSizeM)),
          maxError(maxErrorPx), minEdge(minEdgePx) {
        if (!std::isfinite(maxError) || maxError <= 0 || !std::isfinite(minEdge) || minEdge <= 0) {
            invalid("tracking error and edge thresholds must be positive and finite");
        }
        if (!rig.referenceTag.count(rig.referenceId)) invalid("Reference ID is missing from rig");
        for (const auto& tag : rig.referenceTag) {
            transformFromJson(transformToJson(tag.second), "rig transform");
            auto& corners = points[tag.first];
            for (int index = 0; index < 4; ++index) corners[index] = transformPoint(tag.second, localPoints[index]);
        }
    }

    void reset() {
        previous.reset();
        previousTime.reset();
    }

    std::map<int, double> errors(const Transform& pose, const Detections& valid, const Camera& camera) const {
        std::map<int, double> report;
        for (const auto& tag : valid) {
            const auto& corners = points.at(tag.first);
            const std::vector<cv::Point3d> objects(corners.begin(), corners.end());
            if (!positiveDepth(pose, objects, .01)) {
                report[tag.first] = std::numeric_limits<double>::infinity();
            } else {
                try {
                    std::vector<cv::Point3d> cameraPoints;
                    for (const auto& point : objects) cameraPoints.push_back(transformPoint(pose, point));
                    const auto projected = projectCamera(camera, cameraPoints);
                    double squared = 0;
                    for (std::size_t index = 0; index < projected.size(); ++index) {
                        const auto difference = projected[index] - tag.second[index];
                        squared += difference.dot(difference);
                    }
                    report[tag.first] = std::sqrt(squared / projected.size());
                } catch (const cv::Exception&) {
                    report[tag.first] = std::numeric_limits<double>::infinity();
                }
            }
        }
        return report;
    }

    Transform refine(const Transform& seed, const std::vector<int>& tags,
                     const std::map<int, std::vector<cv::Point2d>>& normalized) const {
        std::vector<cv::Point3d> objects;
        std::vector<cv::Point2d> pixels;
        for (int tag : tags) {
            const auto& corners = points.at(tag);
            const auto& image = normalized.at(tag);
            objects.insert(objects.end(), corners.begin(), corners.end());
            pixels.insert(pixels.end(), image.begin(), image.end());
        }
        cv::Vec3d rvec = rotationVector(seed);
        cv::Vec3d tvec = translation(seed);
        try {
            if (cv::solvePnP(objects, pixels, cv::Matx33d::eye(), cv::noArray(), rvec, tvec,
                             true, cv::SOLVEPNP_ITERATIVE)) {
                const auto pose = makeTransform(rvec, tvec);
                if (finiteTransform(pose)) return pose;
            }
        } catch (const cv::Exception&) {
        }
        return seed;
    }
};

RigTracker::RigTracker(Rig rig, double maxErrorPx, double minEdgePx)
    : impl_(std::make_unique<Impl>(std::move(rig), maxErrorPx, minEdgePx)) {}

RigTracker::~RigTracker() = default;

void RigTracker::reset() { impl_->reset(); }

TrackingResult RigTracker::solve(const Detections& detections, const Camera& camera, double timestamp) {
    if (!std::isfinite(timestamp)) invalid("timestamp must be finite");
    auto& state = *impl_;
    TrackingResult result;
    result.reason = "没有检测到已标定 Tag";
    Detections valid;
    std::map<int, std::vector<cv::Point2d>> normalized;
    std::vector<Transform> seeds;
    bool registeredVisible = false;
    for (const auto& tag : detections) {
        result.visibleIds.push_back(tag.first);
        const auto registered = state.rig.referenceTag.find(tag.first);
        if (registered == state.rig.referenceTag.end()) continue;
        registeredVisible = true;
        const Corners& corners = tag.second;
        if (!finiteCorners(corners)) continue;
        double minEdge = std::numeric_limits<double>::infinity();
        bool inside = true;
        for (int index = 0; index < 4; ++index) {
            minEdge = std::min(minEdge, cv::norm(corners[index] - corners[(index + 3) % 4]));
            inside = inside && corners[index].x > 3 && corners[index].y > 3 &&
                     corners[index].x < camera.width - 4 && corners[index].y < camera.height - 4;
        }
        if (minEdge < state.minEdge || !inside) continue;
        const std::vector<cv::Point3d> objects(state.localPoints.begin(), state.localPoints.end());
        std::vector<cv::Mat> rotations, translations;
        try {
            const auto pixels = normalizePixels(camera, corners);
            const int solutions = cv::solvePnPGeneric(objects, pixels, cv::Matx33d::eye(), cv::noArray(),
                                                      rotations, translations, false, cv::SOLVEPNP_IPPE_SQUARE);
            if (!solutions) continue;
            valid.emplace(tag.first, corners);
            normalized.emplace(tag.first, pixels);
            for (std::size_t index = 0; index < rotations.size(); ++index) {
                const auto cameraTag = makeTransform(matVector(rotations[index]), matVector(translations[index]));
                const auto cameraReference = cameraTag * inverseRigid(registered->second);
                if (finiteTransform(cameraReference)) seeds.push_back(cameraReference);
            }
        } catch (const cv::Exception&) {
            continue;
        }
    }
    if (valid.empty()) {
        if (registeredVisible) {
            result.status = "WARN";
            result.reason = "Tag 太小、靠近画面边缘或位姿求解失败";
        }
        state.reset();
        return result;
    }
    const bool recent = state.previous && state.previousTime &&
        timestamp > *state.previousTime && timestamp - *state.previousTime < .2;
    if (recent) seeds.push_back(*state.previous);
    struct Hypothesis {
        Transform pose;
        std::vector<int> used;
        std::map<int, double> report;
        double rms;
    };
    std::vector<Hypothesis> hypotheses;
    for (const auto& seed : seeds) {
        const auto initialErrors = state.errors(seed, valid, camera);
        std::vector<int> tags;
        for (const auto& error : initialErrors) if (error.second <= state.maxError * 2) tags.push_back(error.first);
        if (tags.empty()) continue;
        Transform pose = state.refine(seed, tags, normalized);
        auto report = state.errors(pose, valid, camera);
        std::vector<int> used;
        for (const auto& error : report) if (error.second <= state.maxError) used.push_back(error.first);
        if (used.empty()) continue;
        if (used != tags) {
            pose = state.refine(pose, used, normalized);
            report = state.errors(pose, valid, camera);
            used.erase(std::remove_if(used.begin(), used.end(), [&](int tag) {
                return !(report.at(tag) <= state.maxError);
            }), used.end());
        }
        if (used.empty()) continue;
        double squared = 0;
        for (int tag : used) squared += report.at(tag) * report.at(tag);
        const double rms = std::sqrt(squared / used.size());
        hypotheses.push_back({pose, std::move(used), std::move(report), rms});
    }
    if (hypotheses.empty()) {
        result.status = "WARN";
        result.reason = "重投影误差过大，已拒绝该帧";
        state.reset();
        return result;
    }
    std::stable_sort(hypotheses.begin(), hypotheses.end(), [](const auto& first, const auto& second) {
        if (first.used.size() != second.used.size()) return first.used.size() > second.used.size();
        return first.rms < second.rms;
    });
    std::size_t bestIndex = 0;
    if (recent) {
        double bestAngle = std::numeric_limits<double>::infinity();
        const std::size_t bestCount = hypotheses.front().used.size();
        const double bestRms = hypotheses.front().rms;
        for (std::size_t index = 0; index < hypotheses.size(); ++index) {
            const auto& hypothesis = hypotheses[index];
            if (hypothesis.used.size() == bestCount && hypothesis.rms <= bestRms + .15) {
                const double angle = rotationDistance(*state.previous, hypothesis.pose);
                if (angle < bestAngle) {
                    bestIndex = index;
                    bestAngle = angle;
                }
            }
        }
    }
    const auto& best = hypotheses[bestIndex];
    result.usedIds = best.used;
    result.reprojectionErrorPx = best.rms;
    for (const auto& tag : best.report) {
        result.tagErrors[tag.first] = std::isfinite(tag.second) ? std::optional<double>(tag.second) : std::nullopt;
    }
    for (int tag : result.visibleIds) {
        if (!std::binary_search(best.used.begin(), best.used.end(), tag)) result.rejectedIds.push_back(tag);
    }
    if (state.previous && state.previousTime) {
        const double dt = timestamp - *state.previousTime;
        const double distance = cv::norm(translation(best.pose) - translation(*state.previous));
        const double angle = rotationDistance(*state.previous, best.pose);
        if (dt > 0 && dt < .2 && (distance > .02 + 3 * dt || angle > (15 + 720 * dt) * pi / 180)) {
            result.status = "WARN";
            result.reason = "短时位姿突跳，轨迹在此断开";
            state.reset();
            return result;
        }
    }
    state.previous = best.pose;
    state.previousTime = timestamp;
    result.pose = best.pose;
    result.status = best.used.size() > 1 ? "MULTI" : "SINGLE";
    result.reason = best.used.size() > 1 ? "多面角点联合定位" : "单面定位，深度与姿态约束较弱";
    return result;
}

RelativeTrace::RelativeTrace(std::size_t limit) : limit_(limit) { reset(); }

void RelativeTrace::reset() {
    initialInverse_.reset();
    filtered_.reset();
    lastRaw_.reset();
    filteredQuaternion_.reset();
    lastTime_.reset();
    trace_.clear();
    segment_ = 0;
    broken_ = true;
}

Json RelativeTrace::update(const TrackingResult& result, double timestamp) {
    if (!std::isfinite(timestamp)) invalid("timestamp must be finite");
    Json output = trackingToJson(result);
    output.erase("pose");
    output["position_m"] = nullptr;
    output["smooth_position_m"] = nullptr;
    output["quaternion_xyzw"] = nullptr;
    output["smooth_quaternion_xyzw"] = nullptr;
    output["segment"] = segment_;
    output["timestamp_s"] = timestamp;
    if (!result.pose) {
        broken_ = true;
        return output;
    }
    if (!finiteTransform(*result.pose)) invalid("trace pose must be finite");
    if (!initialInverse_) initialInverse_ = inverseRigid(*result.pose);
    const Transform relative = *initialInverse_ * *result.pose;
    const cv::Vec3d raw = translation(relative);
    cv::Vec4d quaternion = quaternionXyzw(relative);
    const double dt = lastTime_ ? timestamp - *lastTime_ : 0;
    if (broken_ || dt <= 0 || dt > .2) {
        ++segment_;
        filtered_ = raw;
        filteredQuaternion_ = quaternion;
    } else {
        const double speed = cv::norm(raw - *lastRaw_) / dt;
        const double cutoff = 2.0 + 8.0 * speed;
        const double alpha = 1 - std::exp(-2 * pi * cutoff * dt);
        *filtered_ += alpha * (raw - *filtered_);
        if (quaternion.dot(*filteredQuaternion_) < 0) quaternion *= -1;
        *filteredQuaternion_ += alpha * (quaternion - *filteredQuaternion_);
        *filteredQuaternion_ /= cv::norm(*filteredQuaternion_);
    }
    broken_ = false;
    lastTime_ = timestamp;
    lastRaw_ = raw;
    output["position_m"] = vectorToJson(raw);
    output["smooth_position_m"] = vectorToJson(*filtered_);
    output["quaternion_xyzw"] = vectorToJson(quaternion);
    output["smooth_quaternion_xyzw"] = vectorToJson(*filteredQuaternion_);
    output["segment"] = segment_;
    trace_.push_back({{"t", timestamp}, {"raw", vectorToJson(raw)},
                      {"smooth", vectorToJson(*filtered_)}, {"q_raw", vectorToJson(quaternion)},
                      {"q_smooth", vectorToJson(*filteredQuaternion_)},
                      {"segment", segment_}, {"quality", result.status}});
    while (trace_.size() > limit_) trace_.pop_front();
    return output;
}

Json RelativeTrace::trace() const {
    Json result = Json::array();
    for (const auto& point : trace_) result.push_back(point);
    return result;
}
} // namespace marker6dof
