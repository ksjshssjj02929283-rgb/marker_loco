#include "marker6dof/camera.hpp"

#include <opencv2/videoio.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#ifndef MARKER6DOF_WITH_REALSENSE
#define MARKER6DOF_WITH_REALSENSE 0
#endif
#if MARKER6DOF_WITH_REALSENSE
#include <librealsense2/rs.hpp>
#endif

namespace {
using namespace marker6dof;
namespace fs = std::filesystem;
volatile std::sig_atomic_t interrupted = 0;
void stop(int) { interrupted = 1; }

void help() {
    std::cout <<
        "marker_tracker: D455 / ArUco relative 6DoF algorithms (metres, xyzw)\n\n"
        "  replay --session DIR --rig FILE --output-dir DIR\n"
        "    [--video FILE] [--metadata FILE] [--timestamps CSV]\n"
        "    [--marker-size-mm 48] [--max-frames N] [--min-coverage 0.5]\n"
        "    [--max-error-px 2.5] [--min-edge-px 24]\n"
        "  run-video --config FILE --video FILE --output-dir DIR\n"
        "    [--timestamps CSV] [--max-frames N] [--min-coverage 0.5]\n"
        "  live --rig FILE [--serial SERIAL] [--marker-size-mm 48]\n"
        "    [--max-frames N] [--max-error-px 2.5] [--min-edge-px 24]\n\n"
        "replay also accepts --video FILE --metadata FILE without --session.\n"
        "Missing/rejected observations produce status records without positions.\n"
        "live requires the optional native librealsense2 C++ SDK.\n";
}

class Options {
public:
    Options(int argc, char** argv, const std::set<std::string>& allowed) {
        for (int index = 2; index < argc; ++index) {
            const std::string key = argv[index];
            if (!allowed.count(key)) throw std::invalid_argument("unknown option: " + key);
            if (++index >= argc || std::string(argv[index]).rfind("--", 0) == 0) {
                throw std::invalid_argument("missing value for " + key);
            }
            if (!values_.emplace(key, argv[index]).second) throw std::invalid_argument("duplicate option: " + key);
        }
    }
    std::string text(const std::string& key, const std::string& fallback = "") const {
        const auto found = values_.find(key);
        return found == values_.end() ? fallback : found->second;
    }
    std::string required(const std::string& key) const {
        const auto value = text(key);
        if (value.empty()) throw std::invalid_argument("required option: " + key);
        return value;
    }
    double number(const std::string& key, double fallback) const {
        if (!values_.count(key)) return fallback;
        std::size_t end = 0;
        double value;
        try { value = std::stod(text(key), &end); }
        catch (const std::exception&) { throw std::invalid_argument("invalid number for " + key); }
        if (end != text(key).size() || !std::isfinite(value)) throw std::invalid_argument("invalid number for " + key);
        return value;
    }
    std::size_t frameLimit() const {
        if (!values_.count("--max-frames")) return std::numeric_limits<std::size_t>::max();
        const double limit = number("--max-frames", 0);
        if (limit <= 0 || std::trunc(limit) != limit || limit > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("--max-frames must be a positive integer");
        }
        return static_cast<std::size_t>(limit);
    }
    double coverage() const {
        const double value = number("--min-coverage", .5);
        if (value < 0 || value > 1) throw std::invalid_argument("--min-coverage must be within [0,1]");
        return value;
    }
    double positive(const std::string& key, double fallback) const {
        const double value = number(key, fallback);
        if (value <= 0) throw std::invalid_argument(key + " must be positive");
        return value;
    }
private:
    std::map<std::string, std::string> values_;
};

std::ofstream outputFile(const fs::path& file) {
    std::ofstream output(file);
    if (!output) throw std::runtime_error("cannot write file: " + file.string());
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output << std::setprecision(12);
    return output;
}

std::string cell(const Json& value) {
    if (value.is_null()) return "";
    if (value.is_number()) return value.dump();
    std::string raw = value.is_string() ? value.get<std::string>() : value.dump();
    std::string escaped = "\"";
    for (char letter : raw) { if (letter == '"') escaped += '"'; escaped += letter; }
    return escaped + "\"";
}

void csvRow(std::ostream& output, const Json& row, const std::vector<std::string>& fields) {
    for (std::size_t index = 0; index < fields.size(); ++index) {
        if (index) output << ',';
        output << cell(row.value(fields[index], Json(nullptr)));
    }
    output << '\n';
}

void csvHeader(std::ostream& output, const std::vector<std::string>& fields) {
    for (std::size_t index = 0; index < fields.size(); ++index) {
        if (index) output << ',';
        output << fields[index];
    }
    output << '\n';
}

void flattenVector(Json& row, const std::string& key, const std::vector<std::string>& names) {
    const auto value = row.value(key, Json(nullptr));
    for (std::size_t index = 0; index < names.size(); ++index) {
        row[names[index]] = value.is_array() ? value.at(index) : Json(nullptr);
    }
}

Json percentile(std::vector<double> numbers, double fraction) {
    if (numbers.empty()) return nullptr;
    std::sort(numbers.begin(), numbers.end());
    const double offset = fraction * (numbers.size() - 1);
    const auto low = static_cast<std::size_t>(offset);
    const auto high = std::min(low + 1, numbers.size() - 1);
    return numbers[low] + (numbers[high] - numbers[low]) * (offset - low);
}

int finish(Json summary, const fs::path& directory, double minimum) {
    const double coverage = summary.at("frame_coverage").get<double>();
    summary["required_frame_coverage"] = minimum;
    summary["coverage_passed"] = coverage >= minimum;
    summary["validation_passed"] = coverage >= minimum && summary.value("completed", false);
    auto output = outputFile(directory / "summary.json");
    output << summary.dump(2) << '\n';
    std::cout << summary.dump(2) << '\n';
    if (!summary.value("completed", false)) {
        std::cerr << "replay did not complete the requested frames; validation failed\n";
        return 1;
    }
    if (coverage < minimum) {
        std::cerr << "frame coverage " << coverage << " is below required " << minimum << '\n';
        return 1;
    }
    return 0;
}

double videoFps(cv::VideoCapture& capture, const Json& metadata = Json::object()) {
    double fps = capture.get(cv::CAP_PROP_FPS);
    if ((!std::isfinite(fps) || fps <= 0) && metadata.contains("stream")) {
        fps = metadata.at("stream").value("fps", 0.0);
    }
    return fps;
}

Json completion(cv::VideoCapture& capture, const FrameTimes& times,
                std::size_t frames, std::size_t limit) {
    const double reported = capture.get(cv::CAP_PROP_FRAME_COUNT);
    std::optional<std::size_t> expected;
    if (!times.elapsed.empty()) expected = times.elapsed.size();
    else if (std::isfinite(reported) && reported > 0 &&
             reported < static_cast<double>(std::numeric_limits<std::size_t>::max())) {
        expected = static_cast<std::size_t>(reported);
    }
    if (expected) *expected = std::min(*expected, limit);
    return {
        {"expected_frames", expected ? Json(*expected) : Json(nullptr)},
        {"completed", !interrupted && (!expected || frames == *expected)},
        {"interrupted", static_cast<bool>(interrupted)},
        {"requested_frame_limit", limit == std::numeric_limits<std::size_t>::max() ? Json(nullptr) : Json(limit)}
    };
}

int replay(const Options& options) {
    const auto session = options.text("--session");
    const auto video = options.text("--video", session.empty() ? "" : (fs::path(session) / "color_raw.mp4").string());
    const auto metadataFile = options.text("--metadata", session.empty() ? "" : (fs::path(session) / "metadata.json").string());
    if (video.empty() || metadataFile.empty()) throw std::invalid_argument("replay needs --session, or both --video and --metadata");
    const auto rigFile = options.required("--rig");
    const fs::path directory(options.required("--output-dir"));
    const auto limit = options.frameLimit();
    const auto minimum = options.coverage();
    const double markerSize = options.positive("--marker-size-mm", 48) / 1000;
    const auto metadata = readJson(metadataFile);
    const auto camera = cameraFromMetadata(metadata);
    const auto dictionary = metadata.value("aruco", Json::object()).value("dictionary", std::string("DICT_4X4_50"));
    dictionaryId(dictionary);
    RigTracker tracker(loadRig(rigFile, markerSize), options.positive("--max-error-px", 2.5),
                       options.positive("--min-edge-px", 24));
    RelativeTrace trace;
    cv::VideoCapture capture(video);
    if (!capture.isOpened()) throw std::runtime_error("cannot open video: " + video);
    const auto times = loadFrameTimes(session, videoFps(capture, metadata), options.text("--timestamps"));
    fs::create_directories(directory);
    auto trajectory = outputFile(directory / "trajectory.jsonl");
    auto csv = outputFile(directory / "poses.csv");
    const std::vector<std::string> fields = {
        "frame_index", "timestamp_s", "status", "reason", "segment", "visible_ids", "used_ids", "rejected_ids",
        "reprojection_error_px", "tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw",
        "smooth_tx_m", "smooth_ty_m", "smooth_tz_m", "smooth_qx", "smooth_qy", "smooth_qz", "smooth_qw"};
    csvHeader(csv, fields);
    std::size_t frames = 0, measured = 0;
    Json statuses = Json::object();
    std::vector<double> errors;
    cv::Mat image;
    while (!interrupted && frames < limit && capture.read(image)) {
        checkImageSize(camera, image);
        const double timestamp = times.at(frames);
        const auto result = tracker.solve(detectMarkers(image, dictionary), camera, timestamp);
        auto row = trace.update(result, timestamp);
        row["frame_index"] = frames;
        row["T_camera_reference"] = result.pose ? transformToJson(*result.pose) : Json(nullptr);
        trajectory << row.dump() << '\n';
        flattenVector(row, "position_m", {"tx_m", "ty_m", "tz_m"});
        flattenVector(row, "quaternion_xyzw", {"qx", "qy", "qz", "qw"});
        flattenVector(row, "smooth_position_m", {"smooth_tx_m", "smooth_ty_m", "smooth_tz_m"});
        flattenVector(row, "smooth_quaternion_xyzw", {"smooth_qx", "smooth_qy", "smooth_qz", "smooth_qw"});
        csvRow(csv, row, fields);
        statuses[result.status] = statuses.value(result.status, 0) + 1;
        if (result.pose) {
            ++measured;
            if (result.reprojectionErrorPx) errors.push_back(*result.reprojectionErrorPx);
        }
        ++frames;
    }
    if (!frames) throw std::runtime_error("video produced no frames");
    Json summary = {{"input", video}, {"metadata", metadataFile}, {"rig", rigFile},
        {"frames", frames}, {"measured_frames", measured}, {"frame_coverage", static_cast<double>(measured) / frames},
        {"status_counts", statuses}, {"marker_size_m", markerSize}, {"camera", cameraToJson(camera)},
        {"timestamp_source", times.source}, {"timestamp_origin", "first recorded frame"},
        {"median_reprojection_error_px", percentile(errors, .5)}, {"p95_reprojection_error_px", percentile(errors, .95)},
        {"reprojection_coordinate_space", "original distorted image pixels"},
        {"coordinate_convention", "relative pose maps reference-frame points into the first valid reference frame"},
        {"translation_unit", "m"}, {"quaternion_order", "xyzw"}, {"interrupted", static_cast<bool>(interrupted)},
        {"csv", (directory / "poses.csv").string()}, {"jsonl", (directory / "trajectory.jsonl").string()}};
    summary.update(completion(capture, times, frames, limit));
    return finish(std::move(summary), directory, minimum);
}

int runVideo(const Options& options) {
    const auto config = options.required("--config");
    const auto video = options.required("--video");
    const fs::path directory(options.required("--output-dir"));
    const auto limit = options.frameLimit();
    const auto minimum = options.coverage();
    MarkerLocalizer localizer(loadCalibration(config));
    cv::VideoCapture capture(video);
    if (!capture.isOpened()) throw std::runtime_error("cannot open video: " + video);
    const auto times = loadFrameTimes("", videoFps(capture), options.text("--timestamps"));
    fs::create_directories(directory);
    auto csv = outputFile(directory / "tcp_poses.csv");
    const std::vector<std::string> fields = {
        "frame_index", "timestamp_s", "body", "status", "reason", "visible_ids", "used_ids", "corner_count",
        "inlier_count", "reprojection_rmse_px", "max_reprojection_error_px", "tx_m", "ty_m", "tz_m",
        "qx", "qy", "qz", "qw", "runtime_ms"};
    csvHeader(csv, fields);
    std::size_t frames = 0, measuredFrames = 0, records = 0, measuredRecords = 0;
    Json statuses = Json::object();
    std::vector<double> errors;
    cv::Mat image;
    while (!interrupted && frames < limit && capture.read(image)) {
        checkImageSize(localizer.calibration().camera, image);
        const auto started = std::chrono::steady_clock::now();
        const auto estimates = localizer.estimate(localizer.detect(image));
        const double runtime = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        const double timestamp = times.at(frames);
        bool frameMeasured = false;
        for (const auto& estimate : estimates) {
            auto row = poseToJson(estimate);
            row["frame_index"] = frames;
            row["timestamp_s"] = timestamp;
            row["runtime_ms"] = runtime;
            csvRow(csv, row, fields);
            statuses[estimate.status] = statuses.value(estimate.status, 0) + 1;
            ++records;
            if (estimate.status == "MEASURED" && estimate.cameraTcp) {
                frameMeasured = true;
                ++measuredRecords;
                if (estimate.rmsePx) errors.push_back(*estimate.rmsePx);
            }
        }
        if (frameMeasured) ++measuredFrames;
        ++frames;
    }
    if (!frames || !records) throw std::runtime_error("video produced no body-frame records");
    Json summary = {{"input", video}, {"config", config}, {"frames", frames},
        {"measured_frames", measuredFrames}, {"frame_coverage", static_cast<double>(measuredFrames) / frames},
        {"body_frame_records", records}, {"measured_records", measuredRecords},
        {"measured_fraction", static_cast<double>(measuredRecords) / records}, {"status_counts", statuses},
        {"timestamp_source", times.source}, {"median_reprojection_rmse_px", percentile(errors, .5)},
        {"p95_reprojection_rmse_px", percentile(errors, .95)},
        {"coordinate_convention", "T_camera_tcp maps TCP-frame points into the camera optical frame"},
        {"translation_unit", "m"}, {"quaternion_order", "xyzw"}, {"interrupted", static_cast<bool>(interrupted)},
        {"csv", (directory / "tcp_poses.csv").string()}};
    summary.update(completion(capture, times, frames, limit));
    return finish(std::move(summary), directory, minimum);
}

int live(const Options& options) {
    const auto rigFile = options.required("--rig");
    const auto limit = options.frameLimit();
    const double size = options.positive("--marker-size-mm", 48) / 1000;
    const auto rig = loadRig(rigFile, size);
    const double maximumError = options.positive("--max-error-px", 2.5);
    const double minimumEdge = options.positive("--min-edge-px", 24);
#if MARKER6DOF_WITH_REALSENSE
    rs2::context context;
    std::vector<rs2::device> devices;
    for (auto device : context.query_devices()) {
        if (device.supports(RS2_CAMERA_INFO_PRODUCT_LINE) &&
            std::string(device.get_info(RS2_CAMERA_INFO_PRODUCT_LINE)) == "D400") devices.push_back(device);
    }
    auto serial = options.text("--serial");
    if (serial.empty()) {
        if (devices.empty()) throw std::runtime_error("no RealSense D400 camera found");
        if (devices.size() != 1) throw std::invalid_argument("multiple RealSense cameras found; specify --serial");
        serial = devices.front().get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    } else {
        const bool found = std::any_of(devices.begin(), devices.end(), [&](const auto& device) {
            return serial == device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
        });
        if (!found) throw std::runtime_error("requested RealSense D400 serial not found: " + serial);
    }
    rs2::pipeline pipeline(context);
    rs2::config stream;
    stream.enable_device(serial);
    stream.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
    const auto profile = pipeline.start(stream);
    const auto intrinsic = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    std::string distortion;
    switch (intrinsic.model) {
        case RS2_DISTORTION_NONE: distortion = "none"; break;
        case RS2_DISTORTION_BROWN_CONRADY: distortion = "brown_conrady"; break;
        case RS2_DISTORTION_INVERSE_BROWN_CONRADY: distortion = "inverse_brown_conrady"; break;
        default: throw std::runtime_error("unsupported RealSense color distortion model");
    }
    const Json metadata{{"color_intrinsics", {
        {"width", intrinsic.width}, {"height", intrinsic.height}, {"fx", intrinsic.fx}, {"fy", intrinsic.fy},
        {"cx", intrinsic.ppx}, {"cy", intrinsic.ppy}, {"distortion_model", distortion},
        {"coefficients", {intrinsic.coeffs[0], intrinsic.coeffs[1], intrinsic.coeffs[2], intrinsic.coeffs[3], intrinsic.coeffs[4]}}}}};
    const auto camera = cameraFromMetadata(metadata);
    RigTracker tracker(rig, maximumError, minimumEdge);
    RelativeTrace trace;
    std::optional<double> origin;
    for (std::size_t frame = 0; !interrupted && frame < limit; ++frame) {
        const auto color = pipeline.wait_for_frames().get_color_frame();
        if (!color) throw std::runtime_error("RealSense color frame unavailable");
        cv::Mat image(color.get_height(), color.get_width(), CV_8UC3,
                      const_cast<void*>(color.get_data()), color.get_stride_in_bytes());
        checkImageSize(camera, image);
        const double recorded = color.get_timestamp() / 1000;
        if (!origin) origin = recorded;
        const double timestamp = recorded - *origin;
        const auto result = tracker.solve(detectMarkers(image), camera, timestamp);
        auto row = trace.update(result, timestamp);
        row["frame_index"] = frame;
        row["T_camera_reference"] = result.pose ? transformToJson(*result.pose) : Json(nullptr);
        std::cout << row.dump() << std::endl;
    }
    pipeline.stop();
    return 0;
#else
    (void)rig; (void)limit; (void)maximumError; (void)minimumEdge;
    throw std::runtime_error("live requires native librealsense2 development headers/library; install the C++ SDK and rebuild with -DMARKER6DOF_ENABLE_REALSENSE=ON");
#endif
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") { help(); return 0; }
        const std::string command = argv[1];
        if (command != "replay" && command != "run-video" && command != "live") {
            throw std::invalid_argument("unknown command: " + command);
        }
        for (int index = 2; index < argc; ++index) {
            if (std::string(argv[index]) == "--help" || std::string(argv[index]) == "-h") { help(); return 0; }
        }
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        if (command == "replay") return replay(Options(argc, argv, {
            "--session", "--rig", "--metadata", "--video", "--timestamps", "--output-dir",
            "--marker-size-mm", "--max-frames", "--min-coverage", "--max-error-px", "--min-edge-px"}));
        if (command == "run-video") return runVideo(Options(argc, argv, {
            "--config", "--video", "--timestamps", "--output-dir", "--max-frames", "--min-coverage"}));
        return live(Options(argc, argv, {"--rig", "--serial", "--marker-size-mm", "--max-frames", "--max-error-px", "--min-edge-px"}));
    } catch (const std::exception& error) {
        std::cerr << "marker_tracker: " << error.what() << '\n';
        return 2;
    }
}
