#include "dfmarker_api.h"
#include "marker6dof/core.hpp"
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
#include <opencv2/objdetect/aruco_detector.hpp>
#else
#include <opencv2/aruco.hpp>
#endif
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace marker6dof;
namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
struct Fixture {
    std::filesystem::path directory;
    Camera camera;
    Rig rig;
    Json config;
    Fixture() {
        static unsigned sequence = 0;
        directory = std::filesystem::temp_directory_path() /
            ("dfmarker_api_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
             "_" + std::to_string(sequence++));
        std::filesystem::create_directory(directory);
        camera.width = 640; camera.height = 480;
        camera.matrix = {620, 0, 319.5, 0, 620, 239.5, 0, 0, 1};
        camera.distortion = cv::Mat::zeros(5, 1, CV_64F);
        camera.distortionModel = "none";
        rig.referenceTag[2] = Transform::eye();
        rig.referenceTag[3] = makeTransform({0, .6, 0}, {.075, 0, .01});
        Json tags = Json::object();
        for (const auto& tag : rig.referenceTag) tags[std::to_string(tag.first)] = transformToJson(tag.second);
        write("rig.json", {{"reference_id", 2}, {"marker_size_m", .048}, {"T_reference_tag", tags}});
        config = {{"schema_version", 1}, {"units", "m"}, {"camera", cameraToJson(camera)},
                  {"rig_file", "rig.json"}}; // exercise SDK defaults and relative path resolution
        write("config.json", config);
    }
    ~Fixture() { std::error_code ignored; std::filesystem::remove_all(directory, ignored); }
    void write(const char *file, const Json& value) {
        std::ofstream out(directory / file);
        out << value.dump();
        if (!out) throw std::runtime_error("cannot write fixture");
    }
    std::string path() const { return (directory / "config.json").string(); }
    cv::Mat render(double x = 0) const {
        cv::Mat output(camera.height, camera.width, CV_8UC1, cv::Scalar(255));
        const Transform pose = makeTransform({3.0, .06, .08}, {x - .035, .015, .55});
        const auto square = markerObjectPoints(.048);
        const auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        for (const auto& tag : rig.referenceTag) {
            std::vector<cv::Point3d> points;
            for (const auto& point : square) points.push_back(transformPoint(pose * tag.second, point));
            const auto projected = projectCamera(camera, points);
            std::vector<cv::Point2f> destination;
            for (const auto& pixel : projected) destination.emplace_back(pixel);
            cv::Mat marker, warped;
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
            cv::aruco::generateImageMarker(dictionary, tag.first, 200, marker);
#else
            cv::aruco::drawMarker(dictionary, tag.first, 200, marker);
#endif
            const std::vector<cv::Point2f> source{{0, 0}, {199, 0}, {199, 199}, {0, 199}};
            cv::warpPerspective(marker, warped, cv::getPerspectiveTransform(source, destination), output.size(),
                                cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
            cv::min(output, warped, output);
        }
        return output;
    }
};
struct Handle {
    DFMARKER_HANDLE value = nullptr;
    explicit Handle(const Fixture& fixture) {
        require(dfmarker_create(fixture.path().c_str(), &value) == 1, "create fixture handle");
    }
    ~Handle() { dfmarker_destroy(&value); }
};
struct Snapshot {
    dfmarker_result_t result{};
    std::string reason;
    std::vector<int32_t> visible, used, rejected;
    std::vector<dfmarker_detection_t> detections;
    std::vector<dfmarker_tag_error_t> errors;
};
struct Capture {
    std::vector<Snapshot> frames;
    DFMARKER_HANDLE *owner = nullptr;
    bool reentryRejected = true;
};
void callback(const dfmarker_result_t *r, void *data) {
    auto& capture = *static_cast<Capture*>(data);
    Snapshot frame;
    frame.result = *r;
    frame.reason = r->reason;
    if (r->visible_count) frame.visible.assign(r->visible_ids, r->visible_ids + r->visible_count);
    if (r->used_count) frame.used.assign(r->used_ids, r->used_ids + r->used_count);
    if (r->rejected_count) frame.rejected.assign(r->rejected_ids, r->rejected_ids + r->rejected_count);
    if (r->detection_count) frame.detections.assign(r->detections, r->detections + r->detection_count);
    if (r->tag_error_count) frame.errors.assign(r->tag_errors, r->tag_errors + r->tag_error_count);
    capture.frames.push_back(std::move(frame));
    if (capture.owner) {
        const auto before = *capture.owner;
        dfmarker_status_t status;
        capture.reentryRejected = capture.reentryRejected &&
            !dfmarker_get_status(before, &status) && !dfmarker_start(before) && !dfmarker_stop(before) &&
            !dfmarker_register_result_callback(before, nullptr, nullptr) &&
            !dfmarker_feed_image(before, nullptr) && !dfmarker_destroy(capture.owner) && *capture.owner == before;
    }
}
dfmarker_image_t image(const cv::Mat& pixels, uint64_t time, dfmarker_image_format_t format = DFMARKER_MONO8) {
    return {time, static_cast<uint32_t>(pixels.cols), static_cast<uint32_t>(pixels.rows), pixels.step[0], pixels.data,
            pixels.step[0] * static_cast<uint64_t>(pixels.rows - 1) + pixels.cols * pixels.elemSize(), format};
}
void state(DFMARKER_HANDLE handle, dfmarker_status_t expected) {
    dfmarker_status_t status = DFMARKER_ERROR;
    require(dfmarker_get_status(handle, &status) && status == expected, "unexpected lifecycle status");
}

void testLifecycleAndValidation() {
    Fixture fixture;
    Handle handle(fixture);
    Capture capture;
    capture.owner = &handle.value;
    require(!dfmarker_create(fixture.path().c_str(), &handle.value), "create must not overwrite existing handle");
    state(handle.value, DFMARKER_CREATED);
    const cv::Mat pixels = fixture.render();
    const uint64_t epoch = 1700000000000000000ULL;
    auto frame = image(pixels, epoch);
    require(!dfmarker_feed_image(handle.value, &frame), "feed before start rejected");
    require(dfmarker_register_result_callback(handle.value, callback, &capture), "register callback");
    require(dfmarker_start(handle.value) && dfmarker_start(handle.value), "idempotent start");
    auto invalid = frame;
    invalid.width++;
    require(!dfmarker_feed_image(handle.value, &invalid), "resolution mismatch rejected");
    invalid = frame; invalid.data = nullptr;
    require(!dfmarker_feed_image(handle.value, &invalid), "NULL data rejected");
    invalid = frame; invalid.format = static_cast<dfmarker_image_format_t>(99);
    require(!dfmarker_feed_image(handle.value, &invalid), "unknown format rejected");
    invalid = frame; invalid.stride_bytes = 1;
    require(!dfmarker_feed_image(handle.value, &invalid), "short stride rejected");
    invalid = frame; invalid.stride_bytes = std::numeric_limits<uint64_t>::max();
    require(!dfmarker_feed_image(handle.value, &invalid), "overflowing stride rejected");
    invalid = frame; invalid.data_size_bytes--;
    require(!dfmarker_feed_image(handle.value, &invalid), "short buffer rejected");
    require(!dfmarker_feed_image(handle.value, nullptr) && !dfmarker_get_status(handle.value, nullptr), "NULL arguments rejected");
    state(handle.value, DFMARKER_RUNNING);
    require(capture.frames.empty(), "invalid input must not invoke callback");
    require(dfmarker_feed_image(handle.value, &frame), "valid frame accepted after invalid frames");
    require(capture.frames.size() == 1 && capture.frames[0].result.frame_index == 0 &&
            capture.frames[0].result.timestamp_ns == epoch, "invalid frames must not consume timestamp/index");
    require(capture.reentryRejected, "all same-handle calls rejected inside callback");
    require(!dfmarker_feed_image(handle.value, &frame), "repeated timestamp rejected");
    invalid = frame; --invalid.timestamp_ns;
    require(!dfmarker_feed_image(handle.value, &invalid) && capture.frames.size() == 1, "backward timestamp rejected");
    require(dfmarker_register_result_callback(handle.value, nullptr, nullptr), "unregister callback");
    frame.timestamp_ns += 33333333;
    require(dfmarker_feed_image(handle.value, &frame) && capture.frames.size() == 1, "unregistered frames still processed");
    require(dfmarker_register_result_callback(handle.value, callback, &capture), "reregister callback");
    frame.timestamp_ns += 33333333;
    require(dfmarker_feed_image(handle.value, &frame) && capture.frames.back().result.frame_index == 2, "frame index continues without callback");
    require(dfmarker_stop(handle.value) && dfmarker_stop(handle.value), "idempotent stop");
    state(handle.value, DFMARKER_STOPPED);
    require(!dfmarker_start(handle.value) && !dfmarker_feed_image(handle.value, &frame), "stopped handle cannot restart/feed");
    require(dfmarker_destroy(&handle.value) && !handle.value && dfmarker_destroy(&handle.value), "idempotent destroy clears handle");
    require(!dfmarker_destroy(nullptr) && !dfmarker_start(nullptr), "NULL handle arguments rejected");
}

void testImagePoseAndSegments() {
    Fixture fixture;
    Handle handle(fixture);
    Capture capture;
    dfmarker_register_result_callback(handle.value, callback, &capture);
    dfmarker_start(handle.value);
    const cv::Mat mono = fixture.render();
    auto frame = image(mono, 0);
    require(dfmarker_feed_image(handle.value, &frame), "rendered tag image accepted");
    const auto first = capture.frames.back();
    require(first.result.has_pose && first.result.quality == DFMARKER_MULTI && first.used == std::vector<int32_t>({2, 3}),
            "rendered multi-face image must produce a supported pose");
    require(first.detections.size() == 2 && first.errors.size() == 2 && first.result.has_reprojection_error,
            "callback includes detections and per-tag errors");
    const cv::Vec3d expected(-.035, .015, .55);
    cv::Vec3d measured(first.result.camera_pose.position);
    require(cv::norm(measured - expected) < .015, "independent rendering recovers metric camera pose");
    require(cv::norm(cv::Vec3d(first.result.relative_pose.position)) < 1e-10 && first.result.segment == 1,
            "first pose establishes relative origin");

    cv::Mat bgr;
    cv::cvtColor(mono, bgr, cv::COLOR_GRAY2BGR);
    // Coloured image verifies the RGB channel conversion against equivalent BGR.
    bgr.setTo(cv::Scalar(180, 220, 250), mono == 255);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    const uint64_t stride = bgr.cols * 3 + 17;
    std::vector<uint8_t> padded(stride * bgr.rows, 77);
    for (int row = 0; row < bgr.rows; ++row)
        std::copy(bgr.ptr<uint8_t>(row), bgr.ptr<uint8_t>(row) + bgr.cols * 3, padded.data() + row * stride);
    dfmarker_image_t colour{33333333, 640, 480, stride, padded.data(), padded.size(), DFMARKER_BGR8};
    require(dfmarker_feed_image(handle.value, &colour), "padded BGR frame accepted");
    const auto bgrResult = capture.frames.back();
    frame = image(rgb, 66666666, DFMARKER_RGB8);
    require(dfmarker_feed_image(handle.value, &frame), "RGB frame accepted");
    const auto rgbResult = capture.frames.back();
    require(rgbResult.result.has_pose && bgrResult.result.has_pose && rgbResult.used == bgrResult.used,
            "both colour formats recover pose");
    for (int i = 0; i < 3; ++i)
        require(std::abs(rgbResult.result.camera_pose.position[i] - bgrResult.result.camera_pose.position[i]) < 1e-8,
                "equivalent RGB/BGR produce equivalent poses");
    require(cv::norm(cv::Vec3d(rgbResult.result.relative_pose.position)) < .015, "format changes preserve original origin");

    cv::Mat blank(480, 640, CV_8UC1, cv::Scalar(255));
    frame = image(blank, 100000000);
    require(dfmarker_feed_image(handle.value, &frame), "loss is successful feed");
    const auto missing = capture.frames.back().result;
    require(missing.quality == DFMARKER_LOST && !missing.has_pose && !missing.has_reprojection_error &&
            missing.detection_count == 0 && !missing.detections, "loss has explicit absent values");
    for (double p : missing.camera_pose.position) require(p == 0, "absent poses zero-filled");
    const cv::Mat moved = fixture.render(.005);
    frame = image(moved, 133333333);
    require(dfmarker_feed_image(handle.value, &frame), "recovery accepted");
    const auto resumed = capture.frames.back().result;
    require(resumed.has_pose && resumed.segment == 2, "recovery starts new segment");
    require(cv::norm(cv::Vec3d(resumed.relative_pose.position)) > .001, "recovery preserves original origin");
    for (int i = 0; i < 3; ++i)
        require(resumed.relative_pose.position[i] == resumed.smooth_relative_pose.position[i], "segment resets display smoothing");
    frame.timestamp_ns = 500000000;
    require(dfmarker_feed_image(handle.value, &frame) && capture.frames.back().result.segment == 3, "long gap splits trajectory");
}

void throwingCallback(const dfmarker_result_t *, void *) { throw std::runtime_error("injected callback failure"); }
void testIndependentHandlesAndFailures() {
    Fixture fixture;
    Handle first(fixture), second(fixture);
    Capture a, b;
    dfmarker_register_result_callback(first.value, callback, &a);
    dfmarker_register_result_callback(second.value, callback, &b);
    dfmarker_start(first.value); dfmarker_start(second.value);
    cv::Mat pixels = fixture.render();
    auto frame = image(pixels, 9000000000000000000ULL);
    require(dfmarker_feed_image(first.value, &frame), "first independent handle feed");
    pixels = fixture.render(.015);
    frame = image(pixels, 0);
    require(dfmarker_feed_image(second.value, &frame), "second independent epoch and pose");
    require(a.frames[0].result.frame_index == 0 && b.frames[0].result.frame_index == 0 &&
            cv::norm(cv::Vec3d(b.frames[0].result.relative_pose.position)) < 1e-10, "handles have independent origins and clocks");
    dfmarker_register_result_callback(second.value, throwingCallback, nullptr);
    frame.timestamp_ns = 33333333;
    require(!dfmarker_feed_image(second.value, &frame), "unexpected callback exception cannot escape C API");
    state(second.value, DFMARKER_ERROR);
    require(!dfmarker_start(second.value) && !dfmarker_feed_image(second.value, &frame), "ERROR handle must be recreated");
    state(first.value, DFMARKER_RUNNING);

    DFMARKER_HANDLE output = nullptr;
    require(!dfmarker_create(nullptr, &output) && !output && !dfmarker_create("/missing/dfmarker_config.json", &output),
            "invalid creation leaves output NULL");
    auto invalid = fixture.config;
    invalid["units"] = "mm"; fixture.write("config.json", invalid);
    require(!dfmarker_create(fixture.path().c_str(), &output) && !output, "nonmetric configuration rejected");
    invalid = fixture.config; invalid["marker_size_m"] = .05; fixture.write("config.json", invalid);
    require(!dfmarker_create(fixture.path().c_str(), &output), "rig size mismatch rejected");
    invalid = fixture.config; invalid["quality"] = {{"max_error_px", -1}}; fixture.write("config.json", invalid);
    require(!dfmarker_create(fixture.path().c_str(), &output), "invalid thresholds rejected");
    invalid = fixture.config; invalid["camera"]["distortion_model"] = "unknown"; fixture.write("config.json", invalid);
    require(!dfmarker_create(fixture.path().c_str(), &output), "unsupported distortion rejected at creation");
}
}

int main() {
    cv::setNumThreads(1);
    const std::pair<const char *, std::function<void()>> tests[] = {
        {"API lifecycle, validation, time and callback reentry", testLifecycleAndValidation},
        {"rendered image detection, formats, pose and segments", testImagePoseAndSegments},
        {"independent handles, config errors and exception boundary", testIndependentHandlesAndFailures}
    };
    int failed = 0;
    for (const auto& test : tests) {
        try { test.second(); std::cout << "PASS: " << test.first << '\n'; }
        catch (const std::exception& error) { ++failed; std::cerr << "FAIL: " << test.first << ": " << error.what() << '\n'; }
    }
    return failed ? 1 : 0;
}
