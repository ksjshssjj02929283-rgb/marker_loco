#include "dfmarker_api.h"
#include "marker6dof/core.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>

static_assert(NLOHMANN_JSON_VERSION_MAJOR > 3 ||
              (NLOHMANN_JSON_VERSION_MAJOR == 3 && NLOHMANN_JSON_VERSION_MINOR >= 2),
              "dfmarker requires nlohmann_json >= 3.2");

namespace {
using namespace marker6dof;

struct Configuration {
    Camera camera;
    Rig rig;
    std::string dictionary;
    double maxError;
    double minEdge;
};

Configuration loadConfiguration(const char *path) {
    if (!path || !*path) throw std::invalid_argument("config_path must not be empty");
    const Json raw = readJson(path);
    if (!raw.is_object() || !raw.at("schema_version").is_number_integer() ||
        raw.at("schema_version") != 1 || raw.at("units") != "m") {
        throw std::invalid_argument("SDK config requires schema_version 1 and units m");
    }
    Configuration result;
    result.camera = cameraFromJson(raw.at("camera"));
    const auto rigName = raw.at("rig_file").get<std::string>();
    if (rigName.empty()) throw std::invalid_argument("rig_file must not be empty");
    std::filesystem::path rigPath(rigName);
    if (rigPath.is_relative()) rigPath = std::filesystem::path(path).parent_path() / rigPath;
    result.rig = loadRig(rigPath.string(), raw.value("marker_size_m", 0.048));
    result.dictionary = raw.value("aruco", Json::object()).value("dictionary", std::string("DICT_4X4_50"));
    dictionaryId(result.dictionary);
    const Json quality = raw.value("quality", Json::object());
    result.maxError = quality.value("max_error_px", 2.5);
    result.minEdge = quality.value("min_edge_px", 24.0);
    if (!std::isfinite(result.maxError) || result.maxError <= 0 ||
        !std::isfinite(result.minEdge) || result.minEdge <= 0) {
        throw std::invalid_argument("quality thresholds must be positive and finite");
    }
    return result;
}

template<class T> const T *dataOrNull(const std::vector<T>& values) {
    return values.empty() ? nullptr : values.data();
}

void jsonPose(const Json& row, const char *position, const char *orientation, dfmarker_pose_t& output) {
    for (int i = 0; i < 3; ++i) output.position[i] = row.at(position).at(i).get<double>();
    for (int i = 0; i < 4; ++i) output.orientation[i] = row.at(orientation).at(i).get<double>();
}

class MarkerSystem {
public:
    explicit MarkerSystem(Configuration config)
        : config_(std::move(config)), tracker_(config_.rig, config_.maxError, config_.minEdge), trace_(1) {}

    bool busy() const { return busy_; }

    template<class Action> int32_t invoke(Action action) noexcept {
        if (busy_) return 0;
        struct Guard {
            bool& busy;
            explicit Guard(bool& b) : busy(b) { busy = true; }
            ~Guard() { busy = false; }
        } guard(busy_);
        try {
            action();
            return 1;
        } catch (const std::invalid_argument& error) {
            std::fprintf(stderr, "[DFMARKER] %s\n", error.what());
        } catch (const std::exception& error) {
            status_ = DFMARKER_ERROR;
            std::fprintf(stderr, "[DFMARKER] processing failed: %s\n", error.what());
        } catch (...) {
            status_ = DFMARKER_ERROR;
            std::fprintf(stderr, "[DFMARKER] unexpected processing failure\n");
        }
        return 0;
    }

    int32_t start() noexcept {
        return invoke([&] {
            if (status_ != DFMARKER_CREATED && status_ != DFMARKER_RUNNING)
                throw std::invalid_argument("stopped/error handles cannot restart");
            status_ = DFMARKER_RUNNING;
        });
    }
    int32_t stop() noexcept {
        return invoke([&] {
            if (status_ == DFMARKER_ERROR) throw std::invalid_argument("handle is in ERROR; destroy it");
            status_ = DFMARKER_STOPPED;
        });
    }
    int32_t getStatus(dfmarker_status_t *output) noexcept {
        return invoke([&] {
            if (!output) throw std::invalid_argument("status output must not be NULL");
            *output = status_;
        });
    }
    int32_t registerCallback(dfmarker_result_callback_t callback, void *data) noexcept {
        return invoke([&] { callback_ = callback; userData_ = data; });
    }
    int32_t feed(const dfmarker_image_t *image) noexcept {
        return invoke([&] {
            validate(image);
            // From this point on, errors are processing failures: state may have
            // advanced and must not be reused as if the frame were unconsumed.
            try { process(*image); }
            catch (...) { status_ = DFMARKER_ERROR; throw; }
        });
    }

private:
    void validate(const dfmarker_image_t *image) const {
        if (status_ != DFMARKER_RUNNING) throw std::invalid_argument("handle must be running");
        if (!image || !image->data) throw std::invalid_argument("image/data must not be NULL");
        if (image->width != static_cast<uint32_t>(config_.camera.width) ||
            image->height != static_cast<uint32_t>(config_.camera.height))
            throw std::invalid_argument("image resolution differs from calibration");
        uint64_t channels;
        switch (image->format) {
            case DFMARKER_MONO8: channels = 1; break;
            case DFMARKER_RGB8: case DFMARKER_BGR8: channels = 3; break;
            default: throw std::invalid_argument("unsupported image format");
        }
        const uint64_t row = static_cast<uint64_t>(image->width) * channels;
        const uint64_t rows = image->height - 1;
        const uint64_t max = std::numeric_limits<uint64_t>::max();
        if (image->stride_bytes < row || image->stride_bytes > std::numeric_limits<std::size_t>::max() ||
            (rows && image->stride_bytes > (max - row) / rows) ||
            image->stride_bytes > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / image->height)
            throw std::invalid_argument("invalid/overflowing image stride");
        const uint64_t required = rows * image->stride_bytes + row;
        if (required > image->data_size_bytes ||
            required > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()))
            throw std::invalid_argument("image buffer is too small or too large");
        if (lastTime_ && image->timestamp_ns <= *lastTime_)
            throw std::invalid_argument("image timestamps must strictly increase");
    }

    void process(const dfmarker_image_t& image) {
        const bool mono = image.format == DFMARKER_MONO8;
        cv::Mat pixels(config_.camera.height, config_.camera.width, mono ? CV_8UC1 : CV_8UC3,
                       const_cast<uint8_t*>(image.data), static_cast<std::size_t>(image.stride_bytes));
        cv::Mat gray;
        if (mono) gray = pixels;
        else cv::cvtColor(pixels, gray, image.format == DFMARKER_RGB8 ? cv::COLOR_RGB2GRAY : cv::COLOR_BGR2GRAY);
        const auto detections = detectMarkers(gray, config_.dictionary);
        const uint64_t origin = firstTime_.value_or(image.timestamp_ns);
        const double elapsed = static_cast<double>(image.timestamp_ns - origin) * 1e-9;
        const auto tracking = tracker_.solve(detections, config_.camera, elapsed);
        const Json row = trace_.update(tracking, elapsed);

        dfmarker_result_t result{};
        result.timestamp_ns = image.timestamp_ns;
        result.frame_index = frameIndex_;
        result.segment = row.at("segment").get<uint64_t>();
        result.reason = tracking.reason.c_str();
        if (tracking.status == "MULTI") result.quality = DFMARKER_MULTI;
        else if (tracking.status == "SINGLE") result.quality = DFMARKER_SINGLE;
        else if (tracking.status == "WARN") result.quality = DFMARKER_WARN;
        else result.quality = DFMARKER_LOST;
        if (tracking.pose) {
            result.has_pose = 1;
            const auto p = translation(*tracking.pose);
            const auto q = quaternionXyzw(*tracking.pose);
            for (int i = 0; i < 3; ++i) result.camera_pose.position[i] = p[i];
            for (int i = 0; i < 4; ++i) result.camera_pose.orientation[i] = q[i];
            jsonPose(row, "position_m", "quaternion_xyzw", result.relative_pose);
            jsonPose(row, "smooth_position_m", "smooth_quaternion_xyzw", result.smooth_relative_pose);
        }
        if (tracking.reprojectionErrorPx) {
            result.has_reprojection_error = 1;
            result.reprojection_error_px = *tracking.reprojectionErrorPx;
        }
        std::vector<dfmarker_detection_t> observed;
        for (const auto& detection : detections) {
            dfmarker_detection_t entry{};
            entry.tag_id = detection.first;
            entry.has_corners = std::all_of(detection.second.begin(), detection.second.end(), [](const auto& point) {
                return std::isfinite(point.x) && std::isfinite(point.y);
            });
            if (entry.has_corners) for (int i = 0; i < 4; ++i) {
                entry.corners_px[2*i] = detection.second[i].x;
                entry.corners_px[2*i+1] = detection.second[i].y;
            }
            observed.push_back(entry);
        }
        std::vector<int32_t> visible(tracking.visibleIds.begin(), tracking.visibleIds.end());
        std::vector<int32_t> used(tracking.usedIds.begin(), tracking.usedIds.end());
        std::vector<int32_t> rejected(tracking.rejectedIds.begin(), tracking.rejectedIds.end());
        std::vector<dfmarker_tag_error_t> errors;
        for (const auto& error : tracking.tagErrors)
            errors.push_back({error.first, error.second ? 1 : 0, error.second.value_or(0)});
        result.detections = dataOrNull(observed); result.detection_count = static_cast<uint32_t>(observed.size());
        result.visible_ids = dataOrNull(visible); result.visible_count = static_cast<uint32_t>(visible.size());
        result.used_ids = dataOrNull(used); result.used_count = static_cast<uint32_t>(used.size());
        result.rejected_ids = dataOrNull(rejected); result.rejected_count = static_cast<uint32_t>(rejected.size());
        result.tag_errors = dataOrNull(errors); result.tag_error_count = static_cast<uint32_t>(errors.size());
        firstTime_ = origin;
        lastTime_ = image.timestamp_ns;
        ++frameIndex_;
        if (callback_) callback_(&result, userData_);
    }

    Configuration config_;
    RigTracker tracker_;
    RelativeTrace trace_;
    dfmarker_status_t status_ = DFMARKER_CREATED;
    bool busy_ = false;
    dfmarker_result_callback_t callback_ = nullptr;
    void *userData_ = nullptr;
    std::optional<uint64_t> firstTime_, lastTime_;
    uint64_t frameIndex_ = 0;
};
} // namespace

extern "C" {
int32_t dfmarker_create(const char *path, DFMARKER_HANDLE *handle) {
    if (!handle || *handle) return 0;
    try {
        auto system = std::make_unique<MarkerSystem>(loadConfiguration(path));
        *handle = system.release();
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[DFMARKER] create failed: %s\n", error.what());
    } catch (...) { std::fprintf(stderr, "[DFMARKER] create failed\n"); }
    return 0;
}
int32_t dfmarker_destroy(DFMARKER_HANDLE *handle) {
    if (!handle) return 0;
    auto *system = static_cast<MarkerSystem*>(*handle);
    if (system && system->busy()) return 0;
    delete system;
    *handle = nullptr;
    return 1;
}
int32_t dfmarker_start(DFMARKER_HANDLE handle) {
    return handle ? static_cast<MarkerSystem*>(handle)->start() : 0;
}
int32_t dfmarker_stop(DFMARKER_HANDLE handle) {
    return handle ? static_cast<MarkerSystem*>(handle)->stop() : 0;
}
int32_t dfmarker_get_status(DFMARKER_HANDLE handle, dfmarker_status_t *status) {
    return handle ? static_cast<MarkerSystem*>(handle)->getStatus(status) : 0;
}
int32_t dfmarker_register_result_callback(DFMARKER_HANDLE handle, dfmarker_result_callback_t callback, void *data) {
    return handle ? static_cast<MarkerSystem*>(handle)->registerCallback(callback, data) : 0;
}
int32_t dfmarker_feed_image(DFMARKER_HANDLE handle, const dfmarker_image_t *image) {
    return handle ? static_cast<MarkerSystem*>(handle)->feed(image) : 0;
}
}
