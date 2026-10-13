#include "marker6dof/camera.hpp"

#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace marker6dof {
namespace {
std::string modelName(const Camera& camera) {
    auto name = camera.distortionModel;
    if (name.rfind("distortion.", 0) == 0) name.erase(0, 11);
    return name;
}

bool realsenseInverse(const Camera& camera) {
    const auto name = modelName(camera);
    return name == "inverse_brown_conrady";
}

std::array<float, 5> coefficients(const Camera& camera) {
    std::array<float, 5> result{};
    if (camera.distortion.total() != 0 && camera.distortion.total() != 5) {
        throw std::invalid_argument("RealSense Brown models require five coefficients");
    }
    for (int index = 0; index < static_cast<int>(camera.distortion.total()); ++index) {
        result[index] = static_cast<float>(camera.distortion.at<double>(index));
    }
    return result;
}

void checkModel(const Camera& camera) {
    const auto name = modelName(camera);
    if (name != "opencv" && name != "none" && name != "brown_conrady" &&
        name != "inverse_brown_conrady") {
        throw std::invalid_argument("unsupported camera distortion model: " + name);
    }
}

std::vector<std::string> csvFields(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char letter = line[index];
        if (letter == '"') {
            if (quoted && index + 1 < line.size() && line[index + 1] == '"') {
                field.push_back('"');
                ++index;
            } else quoted = !quoted;
        } else if (letter == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else if (letter != '\r') field.push_back(letter);
    }
    if (quoted) throw std::invalid_argument("unterminated quoted timestamp CSV field");
    fields.push_back(field);
    return fields;
}

double numeric(const std::string& value, const std::string& name) {
    std::size_t end = 0;
    double result = 0;
    try { result = std::stod(value, &end); }
    catch (const std::exception&) { throw std::invalid_argument("invalid " + name + ": " + value); }
    if (end != value.size() || !std::isfinite(result)) {
        throw std::invalid_argument("invalid " + name + ": " + value);
    }
    return result;
}
} // namespace

std::vector<cv::Point2d> normalizePixels(const Camera& camera,
                                       const std::vector<cv::Point2d>& pixels) {
    checkModel(camera);
    std::vector<cv::Point2d> normalized;
    if (pixels.empty()) return normalized;
    if (!realsenseInverse(camera)) {
        cv::undistortPoints(pixels, normalized, camera.matrix,
                            modelName(camera) == "none" ? cv::Mat() : camera.distortion);
        return normalized;
    }
    const auto k = coefficients(camera);
    // This implements librealsense's float, ten-step inverse Brown equations,
    // including its radial adjustment before tangential correction. Ordinary
    // OpenCV Brown undistortion omits that adjustment and is not equivalent.
    // Reference: librealsense v2.58.4 src/rs.cpp, rs2_deproject_pixel_to_point.
    const float fx = static_cast<float>(camera.matrix(0, 0));
    const float fy = static_cast<float>(camera.matrix(1, 1));
    const float cx = static_cast<float>(camera.matrix(0, 2));
    const float cy = static_cast<float>(camera.matrix(1, 2));
    for (const auto& pixel : pixels) {
        const float observedX = (static_cast<float>(pixel.x) - cx) / fx;
        const float observedY = (static_cast<float>(pixel.y) - cy) / fy;
        float x = observedX, y = observedY;
        for (int step = 0; step < 10; ++step) {
            const float radius2 = x * x + y * y;
            const float inverseRadial = 1.0f /
                (1.0f + ((k[4] * radius2 + k[1]) * radius2 + k[0]) * radius2);
            const float scaledX = x / inverseRadial, scaledY = y / inverseRadial;
            const float tangentX = 2 * k[2] * scaledX * scaledY +
                k[3] * (radius2 + 2 * scaledX * scaledX);
            const float tangentY = 2 * k[3] * scaledX * scaledY +
                k[2] * (radius2 + 2 * scaledY * scaledY);
            x = (observedX - tangentX) * inverseRadial;
            y = (observedY - tangentY) * inverseRadial;
        }
        if (!std::isfinite(x) || !std::isfinite(y)) {
            throw std::invalid_argument("inverse Brown deprojection failed");
        }
        normalized.emplace_back(x, y);
    }
    return normalized;
}

std::vector<cv::Point2d> normalizePixels(const Camera& camera, const Corners& corners) {
    return normalizePixels(camera,
        std::vector<cv::Point2d>(corners.begin(), corners.end()));
}

std::vector<cv::Point2d> projectCamera(const Camera& camera,
                                     const std::vector<cv::Point3d>& points) {
    checkModel(camera);
    std::vector<cv::Point2d> pixels;
    if (points.empty()) return pixels;
    if (!realsenseInverse(camera)) {
        cv::projectPoints(points, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0),
                          camera.matrix, modelName(camera) == "none" ? cv::Mat() : camera.distortion, pixels);
        return pixels;
    }
    const auto k = coefficients(camera);
    for (const auto& point : points) {
        if (point.z <= 0) throw std::invalid_argument("cannot project a point behind the camera");
        float x = static_cast<float>(point.x) / static_cast<float>(point.z);
        float y = static_cast<float>(point.y) / static_cast<float>(point.z);
        const float radius2 = x * x + y * y;
        const float radial = 1 + k[0] * radius2 + k[1] * radius2 * radius2 +
            k[4] * radius2 * radius2 * radius2;
        x *= radial;
        y *= radial;
        const float correctedX = x + 2 * k[2] * x * y + k[3] * (radius2 + 2 * x * x);
        const float correctedY = y + 2 * k[3] * x * y + k[2] * (radius2 + 2 * y * y);
        pixels.emplace_back(correctedX * static_cast<float>(camera.matrix(0, 0)) +
                            static_cast<float>(camera.matrix(0, 2)),
                            correctedY * static_cast<float>(camera.matrix(1, 1)) +
                            static_cast<float>(camera.matrix(1, 2)));
    }
    return pixels;
}

Camera cameraFromMetadata(const Json& metadata) {
    const auto& intrinsics = metadata.at("color_intrinsics");
    auto model = intrinsics.value("distortion_model", std::string("none"));
    if (model.rfind("distortion.", 0) == 0) model.erase(0, 11);
    const Json value = {
        {"image_width", intrinsics.at("width")},
        {"image_height", intrinsics.at("height")},
        {"camera_matrix", {{intrinsics.at("fx"), 0.0, intrinsics.at("cx")},
                           {0.0, intrinsics.at("fy"), intrinsics.at("cy")}, {0.0, 0.0, 1.0}}},
        {"distortion", intrinsics.value("coefficients", Json::array())},
        {"distortion_model", model}
    };
    auto camera = cameraFromJson(value);
    checkModel(camera);
    if (modelName(camera) == "none") camera.distortion.setTo(0);
    if (metadata.contains("stream")) {
        const auto& stream = metadata.at("stream");
        if (stream.value("width", camera.width) != camera.width ||
            stream.value("height", camera.height) != camera.height) {
            throw std::invalid_argument("metadata stream size differs from color intrinsics");
        }
    }
    return camera;
}

void checkImageSize(const Camera& camera, const cv::Mat& image) {
    if (image.cols != camera.width || image.rows != camera.height) {
        throw std::invalid_argument("image size " + std::to_string(image.cols) + "x" +
            std::to_string(image.rows) + " differs from calibrated size " +
            std::to_string(camera.width) + "x" + std::to_string(camera.height));
    }
}

double FrameTimes::at(std::size_t frame) const {
    if (elapsed.empty()) return static_cast<double>(frame) / fps;
    const auto value = elapsed.find(frame);
    if (value == elapsed.end()) throw std::invalid_argument("timestamp CSV misses frame " + std::to_string(frame));
    return value->second;
}

FrameTimes loadFrameTimes(const std::string& session, double fps, const std::string& explicitCsv) {
    namespace fs = std::filesystem;
    FrameTimes result;
    result.fps = fps;
    fs::path file(explicitCsv);
    if (file.empty() && !session.empty()) {
        for (const auto& name : {"timestamps.csv", "frames.csv"}) {
            if (fs::exists(fs::path(session) / name)) { file = fs::path(session) / name; break; }
        }
    }
    if (file.empty()) {
        if (!std::isfinite(fps) || fps <= 0) throw std::invalid_argument("no recorded timestamps and video FPS is invalid");
        return result;
    }
    std::ifstream input(file);
    if (!input) throw std::invalid_argument("cannot read timestamp CSV: " + file.string());
    std::string line;
    if (!std::getline(input, line)) throw std::invalid_argument("timestamp CSV is empty");
    auto header = csvFields(line);
    if (header.front().rfind("\xEF\xBB\xBF", 0) == 0) header.front().erase(0, 3);
    const auto column = [&header](const std::string& key) {
        return std::find(header.begin(), header.end(), key) - header.begin();
    };
    const auto frameColumn = column("frame_index");
    if (frameColumn == static_cast<std::ptrdiff_t>(header.size())) {
        throw std::invalid_argument("timestamp CSV needs frame_index column");
    }
    std::string timeColumn;
    for (const auto& name : {"device_timestamp_ms", "timestamp_s", "host_timestamp_s"}) {
        if (column(name) != static_cast<std::ptrdiff_t>(header.size())) { timeColumn = name; break; }
    }
    if (timeColumn.empty()) throw std::invalid_argument("timestamp CSV needs device_timestamp_ms or timestamp_s or host_timestamp_s");
    const auto timeIndex = column(timeColumn);
    const double scale = timeColumn == "device_timestamp_ms" ? 0.001 : 1.0;
    std::vector<double> timestamps, hostTimestamps;
    const auto hostIndex = column("host_timestamp_s");
    bool usableHost = hostIndex != static_cast<std::ptrdiff_t>(header.size());
    std::size_t nextFrame = 0;
    while (std::getline(input, line)) {
        if (line.empty() || line == "\r") continue;
        const auto fields = csvFields(line);
        if (fields.size() != header.size()) throw std::invalid_argument("timestamp CSV row has wrong column count");
        const double frame = numeric(fields[frameColumn], "frame_index");
        if (frame != static_cast<double>(nextFrame)) throw std::invalid_argument("timestamp CSV frame_index must start at zero and be contiguous");
        const double timestamp = numeric(fields[timeIndex], timeColumn) * scale;
        timestamps.push_back(timestamp);
        if (usableHost) {
            try { hostTimestamps.push_back(numeric(fields[hostIndex], "host_timestamp_s")); }
            catch (const std::invalid_argument&) { usableHost = false; }
        }
        ++nextFrame;
    }
    if (input.bad()) throw std::runtime_error("failed while reading timestamp CSV");
    if (timestamps.empty()) throw std::invalid_argument("timestamp CSV has no frame records");
    const auto strictlyIncreasing = [](const std::vector<double>& values) {
        for (std::size_t index = 1; index < values.size(); ++index) {
            if (values[index] <= values[index - 1]) return false;
        }
        return true;
    };
    bool fallback = false;
    if (!strictlyIncreasing(timestamps)) {
        if (timeColumn == "device_timestamp_ms" && usableHost &&
            hostTimestamps.size() == timestamps.size() && strictlyIncreasing(hostTimestamps)) {
            timestamps = std::move(hostTimestamps);
            timeColumn = "host_timestamp_s";
            fallback = true;
        } else throw std::invalid_argument("recorded timestamps must be strictly increasing");
    }
    for (std::size_t index = 0; index < timestamps.size(); ++index) {
        result.elapsed.emplace(index, timestamps[index] - timestamps.front());
    }
    result.source = file.filename().string() + ":" + timeColumn;
    if (fallback) result.source += " (fallback: non-increasing device clock)";
    return result;
}
} // namespace marker6dof
