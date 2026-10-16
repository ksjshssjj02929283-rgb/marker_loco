#include "session_input.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace demo {
namespace {
namespace fs = std::filesystem;

cv::FileStorage readJson(const fs::path& path) {
    cv::FileStorage storage(path.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened()) throw std::invalid_argument("cannot read JSON: " + path.string());
    return storage;
}

double number(const cv::FileNode& node, const std::string& name) {
    if ((!node.isInt() && !node.isReal()) || !std::isfinite(static_cast<double>(node))) {
        throw std::invalid_argument(name + " must be a finite number");
    }
    return static_cast<double>(node);
}

int dimension(const cv::FileNode& node, const std::string& name) {
    const double value = number(node, name);
    if (value < 1 || value > std::numeric_limits<int>::max() || std::floor(value) != value) {
        throw std::invalid_argument(name + " must be a positive integer");
    }
    return static_cast<int>(value);
}

std::string stringValue(const cv::FileNode& node, const std::string& name,
                        const std::string& defaultValue = "") {
    if (node.empty()) return defaultValue;
    if (!node.isString()) throw std::invalid_argument(name + " must be a string");
    return static_cast<std::string>(node);
}

void flatten(const cv::FileNode& node, std::vector<double>& values, const std::string& name) {
    if (!node.isSeq()) throw std::invalid_argument(name + " must be an array");
    for (const auto& entry : node) {
        if (entry.isSeq()) flatten(entry, values, name);
        else values.push_back(number(entry, name));
    }
}

void validateCamera(CameraModel& camera) {
    if (camera.width <= 0 || camera.height <= 0) {
        throw std::invalid_argument("camera dimensions must be positive");
    }
    for (const auto value : camera.matrix.val) {
        if (!std::isfinite(value)) throw std::invalid_argument("camera matrix must be finite");
    }
    if (camera.matrix(0, 0) <= 0 || camera.matrix(1, 1) <= 0 ||
        std::abs(camera.matrix(2, 0)) > 1e-9 || std::abs(camera.matrix(2, 1)) > 1e-9 ||
        std::abs(camera.matrix(2, 2) - 1.0) > 1e-9 ||
        camera.matrix(0, 2) < 0 || camera.matrix(0, 2) >= camera.width ||
        camera.matrix(1, 2) < 0 || camera.matrix(1, 2) >= camera.height) {
        throw std::invalid_argument("invalid calibrated camera matrix or principal point");
    }
    auto& model = camera.distortionModel;
    if (model.rfind("distortion.", 0) == 0) model.erase(0, 11);
    if (model != "opencv" && model != "none" && model != "brown_conrady" &&
        model != "inverse_brown_conrady") {
        throw std::invalid_argument("unsupported distortion model: " + model);
    }
    const auto size = camera.distortion.size();
    if (size != 4 && size != 5 && size != 8 && size != 12 && size != 14) {
        throw std::invalid_argument("distortion requires 4, 5, 8, 12, or 14 coefficients");
    }
    if ((model == "brown_conrady" || model == "inverse_brown_conrady") && size != 5) {
        throw std::invalid_argument("RealSense Brown models require five distortion coefficients");
    }
    for (auto& value : camera.distortion) {
        if (!std::isfinite(value)) throw std::invalid_argument("distortion coefficients must be finite");
        if (model == "none") value = 0;
    }
    if (!std::isfinite(camera.markerSizeM) || camera.markerSizeM <= 0) {
        throw std::invalid_argument("marker_size_m must be positive and finite");
    }
}

CameraModel sdkCamera(const cv::FileStorage& storage) {
    const auto node = storage["camera"];
    if (!node.isMap()) throw std::invalid_argument("SDK configuration requires a camera object");
    CameraModel camera;
    camera.width = dimension(node["image_width"], "image_width");
    camera.height = dimension(node["image_height"], "image_height");
    const auto matrix = node["camera_matrix"];
    if (!matrix.isSeq() || matrix.size() != 3) {
        throw std::invalid_argument("camera_matrix must be a 3x3 array");
    }
    for (int row = 0; row < 3; ++row) {
        const auto values = matrix[row];
        if (!values.isSeq() || values.size() != 3) {
            throw std::invalid_argument("camera_matrix must be a 3x3 array");
        }
        for (int col = 0; col < 3; ++col) camera.matrix(row, col) = number(values[col], "camera_matrix");
    }
    flatten(node["distortion"], camera.distortion, "distortion");
    camera.distortionModel = stringValue(node["distortion_model"], "distortion_model", "opencv");
    if (!storage["marker_size_m"].empty()) {
        camera.markerSizeM = number(storage["marker_size_m"], "marker_size_m");
    }
    validateCamera(camera);
    return camera;
}

CameraModel metadataCamera(const cv::FileStorage& storage, double markerSize) {
    const auto intrinsics = storage["color_intrinsics"];
    if (!intrinsics.isMap()) throw std::invalid_argument("metadata requires color_intrinsics");
    CameraModel camera;
    camera.width = dimension(intrinsics["width"], "color_intrinsics.width");
    camera.height = dimension(intrinsics["height"], "color_intrinsics.height");
    camera.matrix = cv::Matx33d(number(intrinsics["fx"], "fx"), 0, number(intrinsics["cx"], "cx"),
                               0, number(intrinsics["fy"], "fy"), number(intrinsics["cy"], "cy"),
                               0, 0, 1);
    if (intrinsics["coefficients"].empty()) camera.distortion.assign(5, 0.0);
    else flatten(intrinsics["coefficients"], camera.distortion, "coefficients");
    camera.distortionModel = stringValue(intrinsics["distortion_model"], "distortion_model", "none");
    camera.markerSizeM = markerSize;
    validateCamera(camera);
    const auto stream = storage["stream"];
    if (!stream.empty() &&
        (dimension(stream["width"], "stream.width") != camera.width ||
         dimension(stream["height"], "stream.height") != camera.height)) {
        throw std::invalid_argument("metadata stream size differs from color intrinsics");
    }
    return camera;
}

std::vector<std::string> csvFields(const std::string& line) {
    std::vector<std::string> result;
    std::string field;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char value = line[index];
        if (value == '"') {
            if (quoted && index + 1 < line.size() && line[index + 1] == '"') {
                field.push_back('"');
                ++index;
            } else quoted = !quoted;
        } else if (value == ',' && !quoted) {
            result.push_back(field);
            field.clear();
        } else if (value != '\r') field.push_back(value);
    }
    if (quoted) throw std::invalid_argument("unterminated quoted field in timestamp CSV");
    result.push_back(field);
    return result;
}

long double csvNumber(const std::string& value, const std::string& name) {
    std::size_t end = 0;
    long double result = 0;
    try { result = std::stold(value, &end); }
    catch (const std::exception&) { throw std::invalid_argument("invalid " + name + ": " + value); }
    if (end != value.size() || !std::isfinite(result) || result < 0) {
        throw std::invalid_argument(name + " must be finite and nonnegative: " + value);
    }
    return result;
}

bool strictlyIncreasing(const std::vector<long double>& values) {
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (values[index] <= values[index - 1]) return false;
    }
    return true;
}

void loadTimestamps(SessionInput& result, const fs::path& csv) {
    if (csv.empty()) {
        result.timestampSource = "video_fps";
        return;
    }
    std::ifstream input(csv);
    if (!input) throw std::invalid_argument("cannot read timestamp CSV: " + csv.string());
    std::string line;
    if (!std::getline(input, line)) throw std::invalid_argument("timestamp CSV is empty");
    auto header = csvFields(line);
    if (header.front().rfind("\xEF\xBB\xBF", 0) == 0) header.front().erase(0, 3);
    const auto column = [&header](const std::string& name) {
        const auto found = std::find(header.begin(), header.end(), name);
        return static_cast<std::size_t>(found - header.begin());
    };
    const auto frameColumn = column("frame_index");
    if (frameColumn == header.size()) throw std::invalid_argument("timestamp CSV requires frame_index");
    std::string timeColumn;
    for (const auto& candidate : {"device_timestamp_ms", "timestamp_s", "host_timestamp_s"}) {
        if (column(candidate) != header.size()) { timeColumn = candidate; break; }
    }
    if (timeColumn.empty()) throw std::invalid_argument("timestamp CSV has no supported timestamp column");
    const auto timeIndex = column(timeColumn);
    const auto hostIndex = column("host_timestamp_s");
    bool validHost = hostIndex != header.size();
    std::vector<long double> timestamps, hostTimestamps;
    while (std::getline(input, line)) {
        if (line.empty() || line == "\r") continue;
        const auto fields = csvFields(line);
        if (fields.size() != header.size()) throw std::invalid_argument("timestamp CSV row has wrong column count");
        if (csvNumber(fields[frameColumn], "frame_index") != static_cast<long double>(timestamps.size())) {
            throw std::invalid_argument("timestamp CSV frame_index must start at zero and be contiguous");
        }
        timestamps.push_back(csvNumber(fields[timeIndex], timeColumn));
        if (validHost) {
            try { hostTimestamps.push_back(csvNumber(fields[hostIndex], "host_timestamp_s")); }
            catch (const std::invalid_argument&) { validHost = false; }
        }
    }
    if (input.bad()) throw std::runtime_error("failed while reading timestamp CSV");
    if (timestamps.empty()) throw std::invalid_argument("timestamp CSV has no frame records");
    bool fallback = false;
    if (!strictlyIncreasing(timestamps)) {
        if (timeColumn == "device_timestamp_ms" && validHost &&
            hostTimestamps.size() == timestamps.size() && strictlyIncreasing(hostTimestamps)) {
            timestamps = std::move(hostTimestamps);
            timeColumn = "host_timestamp_s";
            fallback = true;
        } else throw std::invalid_argument("recorded timestamps must be strictly increasing");
    }
    const long double toNanoseconds = timeColumn == "device_timestamp_ms" ? 1000000.0L : 1000000000.0L;
    for (const auto value : timestamps) {
        const long double relative = std::round((value - timestamps.front()) * toNanoseconds);
        if (!std::isfinite(relative) || relative < 0 ||
            relative >= std::ldexp(1.0L, 64)) {
            throw std::invalid_argument("relative timestamp exceeds uint64 nanosecond range");
        }
        const auto nanoseconds = static_cast<uint64_t>(relative);
        if (!result.timestampsNs.empty() && nanoseconds <= result.timestampsNs.back()) {
            throw std::invalid_argument("recorded timestamps are not distinct at nanosecond resolution");
        }
        result.timestampsNs.push_back(nanoseconds);
    }
    result.timestampSource = fs::absolute(csv).lexically_normal().string() + ":" + timeColumn;
    if (fallback) result.timestampSource += " (fallback: non-increasing device clock)";
    result.hasRecordedTimes = true;
}

struct TemporaryConfig {
    fs::path directory;
    ~TemporaryConfig() {
        std::error_code error;
        fs::remove_all(directory, error);
    }
};

std::shared_ptr<TemporaryConfig> temporaryConfig() {
    std::random_device random;
    for (int attempt = 0; attempt < 32; ++attempt) {
        std::ostringstream name;
        name << "dfmarker-video-" << std::hex << random() << random();
        const auto directory = fs::temp_directory_path() / name.str();
        std::error_code error;
        if (!fs::create_directory(directory, error)) {
            if (error && error != std::errc::file_exists) {
                throw std::runtime_error("cannot create temporary SDK config: " + error.message());
            }
            continue;
        }
        auto owner = std::make_shared<TemporaryConfig>();
        owner->directory = directory;
        fs::permissions(directory, fs::perms::owner_all, fs::perm_options::replace);
        return owner;
    }
    throw std::runtime_error("cannot allocate unique temporary SDK configuration directory");
}

void writeConfiguration(const fs::path& path, const CameraModel& camera,
                        const fs::path& rig, const std::string& dictionary) {
    cv::FileStorage output(path.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!output.isOpened()) throw std::runtime_error("cannot create temporary SDK configuration");
    output << "schema_version" << 1 << "units" << "m";
    output << "camera" << "{" << "image_width" << camera.width << "image_height" << camera.height;
    output << "camera_matrix" << "[";
    for (int row = 0; row < 3; ++row) {
        output << "[:";
        for (int col = 0; col < 3; ++col) output << camera.matrix(row, col);
        output << "]";
    }
    output << "]" << "distortion" << "[:";
    for (const auto value : camera.distortion) output << value;
    output << "]" << "distortion_model" << camera.distortionModel << "}";
    output << "rig_file" << fs::absolute(rig).lexically_normal().string();
    output << "marker_size_m" << camera.markerSizeM;
    output << "aruco" << "{" << "dictionary" << dictionary << "}";
    output << "quality" << "{" << "max_error_px" << 2.5 << "min_edge_px" << 24.0 << "}";
    output.release();
}

fs::path regularFile(const fs::path& path, const std::string& name) {
    if (!fs::is_regular_file(path)) throw std::invalid_argument(name + " does not exist: " + path.string());
    return fs::absolute(path).lexically_normal();
}

int referenceId(const cv::FileStorage& rig) {
    const double value = number(rig["reference_id"], "rig.reference_id");
    if (value < 0 || value > std::numeric_limits<int>::max() || std::floor(value) != value) {
        throw std::invalid_argument("rig.reference_id must be a nonnegative integer");
    }
    return static_cast<int>(value);
}
} // namespace

SessionInput loadVideoInput(const std::string& configPath, const std::string& videoPath,
                           const std::string& timestampCsv) {
    SessionInput result;
    result.configPath = regularFile(configPath, "SDK configuration").string();
    result.videoPath = regularFile(videoPath, "video").string();
    auto storage = readJson(result.configPath);
    result.camera = sdkCamera(storage);
    auto rig = fs::path(stringValue(storage["rig_file"], "rig_file"));
    if (rig.empty()) throw std::invalid_argument("SDK configuration requires rig_file");
    if (rig.is_relative()) rig = fs::path(result.configPath).parent_path() / rig;
    result.rigPath = regularFile(rig, "rig calibration").string();
    auto rigStorage = readJson(result.rigPath);
    result.referenceId = referenceId(rigStorage);
    if (!rigStorage["marker_size_m"].empty()) {
        const double size = number(rigStorage["marker_size_m"], "rig.marker_size_m");
        if (size <= 0 || std::abs(size - result.camera.markerSizeM) >
            1e-4 * std::max(size, result.camera.markerSizeM)) {
            throw std::invalid_argument("Marker size differs from rig calibration; recalibrate the rig first");
        }
    }
    loadTimestamps(result, timestampCsv);
    return result;
}

SessionInput loadPackageInput(const std::string& packagePath, const std::string& sessionName,
                             const std::string& rigOverride) {
    const auto package = fs::absolute(packagePath).lexically_normal();
    if (!fs::is_directory(package)) throw std::invalid_argument("test package is not a directory: " + package.string());
    fs::path rig;
    if (!rigOverride.empty()) rig = regularFile(rigOverride, "rig calibration");
    else {
        for (const auto& directory : {"configs", "config"}) {
            for (const auto& filename : {"rig.json", "rig_calibration_id2.json"}) {
                const auto candidate = package / directory / filename;
                if (fs::is_regular_file(candidate)) { rig = candidate; break; }
            }
            if (!rig.empty()) break;
        }
        if (rig.empty()) throw std::invalid_argument("test package has no configs/config rig calibration");
    }
    const auto data = package / "test_data";
    fs::path session;
    if (!sessionName.empty()) {
        const fs::path name(sessionName);
        if (name.is_absolute() || name.has_parent_path() || name == "." || name == "..") {
            throw std::invalid_argument("session must be a directory name under test_data");
        }
        session = data / name;
    } else {
        if (!fs::is_directory(data)) throw std::invalid_argument("test package has no test_data directory");
        for (const auto& entry : fs::directory_iterator(data)) {
            if (entry.is_directory() && fs::is_regular_file(entry.path() / "color_raw.mp4") &&
                fs::is_regular_file(entry.path() / "metadata.json")) {
                if (!session.empty()) throw std::invalid_argument("multiple sessions found; specify --session");
                session = entry.path();
            }
        }
        if (session.empty()) throw std::invalid_argument("test package contains no usable video session");
    }
    SessionInput result;
    result.rigPath = regularFile(rig, "rig calibration").string();
    result.videoPath = regularFile(session / "color_raw.mp4", "session video").string();
    auto rigStorage = readJson(rig);
    result.referenceId = referenceId(rigStorage);
    const double markerSize = rigStorage["marker_size_m"].empty() ? 0.048 :
        number(rigStorage["marker_size_m"], "rig.marker_size_m");
    auto metadata = readJson(regularFile(session / "metadata.json", "session metadata"));
    result.camera = metadataCamera(metadata, markerSize);
    const auto dictionary = stringValue(metadata["aruco"]["dictionary"], "aruco.dictionary", "DICT_4X4_50");
    fs::path timestamps;
    for (const auto& filename : {"timestamps.csv", "frames.csv"}) {
        if (fs::is_regular_file(session / filename)) { timestamps = session / filename; break; }
    }
    if (timestamps.empty()) throw std::invalid_argument("test session requires timestamps.csv or frames.csv");
    loadTimestamps(result, timestamps);
    auto owner = temporaryConfig();
    result.configPath = (owner->directory / "sdk_config.json").string();
    result.temporaryConfigOwner = owner;
    writeConfiguration(result.configPath, result.camera, rig, dictionary);
    return result;
}

} // namespace demo
