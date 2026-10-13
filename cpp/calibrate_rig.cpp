#include "marker6dof/core.hpp"
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
using namespace marker6dof;
namespace fs = std::filesystem;
constexpr double pi = 3.14159265358979323846;
struct Options {
    std::string poses, output;
    int reference = 2, minOverlap = 60;
    double sizeMm = 48;
};

std::vector<std::string> csvRecord(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') { field += '"'; ++i; }
            else if (quoted || field.empty()) quoted = !quoted;
            else throw std::runtime_error("unexpected quote in CSV field");
        } else if (c == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else if (c != '\r' || quoted) field += c;
    }
    if (quoted) throw std::runtime_error("unterminated quoted CSV field");
    fields.push_back(field);
    return fields;
}

double number(const std::string& value, const std::string& name) {
    std::size_t end = 0;
    double result;
    try { result = std::stod(value, &end); }
    catch (const std::exception&) { throw std::runtime_error(name + " must be a finite number"); }
    while (end < value.size() && std::isspace(static_cast<unsigned char>(value[end]))) ++end;
    if (end != value.size() || !std::isfinite(result)) throw std::runtime_error(name + " must be a finite number");
    return result;
}

int integer(const std::string& value, const std::string& name) {
    const double result = number(value, name);
    if (result < 0 || result > std::numeric_limits<int>::max() || std::trunc(result) != result) {
        throw std::runtime_error(name + " must be a non-negative integer");
    }
    return static_cast<int>(result);
}

Options options(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout << "Usage: marker_calibrate_rig --poses poses.csv --output rig.json\n"
                      << "       [--reference-id 2] [--min-overlap 60] [--marker-size-mm 48]\n"
                      << "Fits fixed tag geometry from simultaneous observations (metres).\n";
            std::exit(0);
        }
        if (i + 1 == argc) throw std::runtime_error("missing value for " + flag);
        const std::string value = argv[++i];
        if (flag == "--poses") result.poses = value;
        else if (flag == "--output") result.output = value;
        else if (flag == "--reference-id") result.reference = integer(value, flag);
        else if (flag == "--min-overlap") result.minOverlap = integer(value, flag);
        else if (flag == "--marker-size-mm") result.sizeMm = number(value, flag);
        else throw std::runtime_error("unknown option: " + flag);
    }
    if (result.poses.empty() || result.output.empty()) throw std::runtime_error("--poses and --output are required");
    if (result.minOverlap < 1 || result.sizeMm <= 0) throw std::runtime_error("--min-overlap and --marker-size-mm must be positive");
    if (fs::absolute(result.poses).lexically_normal() == fs::absolute(result.output).lexically_normal()) {
        throw std::runtime_error("CSV input and JSON output must be different paths");
    }
    return result;
}

using Frames = std::map<int, std::map<int, Transform>>;
Frames loadRows(const std::string& path, std::size_t& rowCount) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open poses CSV: " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("poses CSV is empty");
    if (line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
    const auto headers = csvRecord(line);
    std::map<std::string, std::size_t> columns;
    for (std::size_t i = 0; i < headers.size(); ++i) {
        if (!columns.emplace(headers[i], i).second) throw std::runtime_error("duplicate CSV column: " + headers[i]);
    }
    for (const char* name : {"frame_index", "tag_id", "rx_rad", "ry_rad", "rz_rad", "tx_m", "ty_m", "tz_m", "reprojection_error_px"}) {
        if (!columns.count(name)) throw std::runtime_error(std::string("missing CSV column: ") + name);
    }
    Frames frames;
    std::size_t lineIndex = 1;
    while (std::getline(input, line)) {
        ++lineIndex;
        if (line.empty() || line == "\r") continue;
        try {
            const auto fields = csvRecord(line);
            if (fields.size() != headers.size()) throw std::runtime_error("CSV field count does not match header");
            auto field = [&](const std::string& name) -> const std::string& { return fields.at(columns.at(name)); };
            const int frame = integer(field("frame_index"), "frame_index");
            const int tag = integer(field("tag_id"), "tag_id");
            const cv::Vec3d rvec(number(field("rx_rad"), "rx_rad"), number(field("ry_rad"), "ry_rad"), number(field("rz_rad"), "rz_rad"));
            const cv::Vec3d tvec(number(field("tx_m"), "tx_m"), number(field("ty_m"), "ty_m"), number(field("tz_m"), "tz_m"));
            if (tvec[2] <= 0) throw std::runtime_error("tag observation must be in front of the camera");
            if (number(field("reprojection_error_px"), "reprojection_error_px") < 0) throw std::runtime_error("reprojection_error_px must be non-negative");
            // Match the initial recorder loader: the last row for a frame/tag wins.
            frames[frame][tag] = makeTransform(rvec, tvec);
            ++rowCount;
        } catch (const std::exception& error) {
            throw std::runtime_error("CSV line " + std::to_string(lineIndex) + ": " + error.what());
        }
    }
    if (input.bad()) throw std::runtime_error("failed while reading poses CSV");
    if (frames.empty()) throw std::runtime_error("poses CSV contains no observations");
    return frames;
}

cv::Matx33d rotation(const Transform& t) {
    return {t(0,0), t(0,1), t(0,2), t(1,0), t(1,1), t(1,2), t(2,0), t(2,1), t(2,2)};
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t center = values.size() / 2;
    return values.size() % 2 ? values[center] : (values[center - 1] + values[center]) / 2;
}

Transform average(const std::vector<Transform>& samples) {
    const cv::Vec4d reference = quaternionXyzw(samples.front());
    cv::Vec4d q(0, 0, 0, 0);
    std::array<std::vector<double>, 3> positions;
    for (const auto& sample : samples) {
        const auto quaternion = quaternionXyzw(sample);
        q += quaternion.dot(reference) < 0 ? -quaternion : quaternion;
        const auto position = translation(sample);
        for (int i = 0; i < 3; ++i) positions[i].push_back(position[i]);
    }
    const double norm = cv::norm(q);
    if (!std::isfinite(norm) || norm < 1e-12) throw std::runtime_error("rotation samples cannot be averaged");
    q /= norm;
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    const cv::Matx33d matrix(1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                            2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                            2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y));
    Transform result = Transform::eye();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) result(row,col) = matrix(row,col);
        result(row,3) = median(positions[row]);
    }
    return result;
}

std::optional<Transform> robustAverage(const std::vector<Transform>& samples, std::size_t& inlierCount) {
    const auto center = average(samples);
    std::vector<Transform> selected;
    for (const auto& sample : samples) {
        cv::Vec3d delta;
        cv::Rodrigues(rotation(center).t() * rotation(sample), delta);
        if (cv::norm(translation(sample) - translation(center)) <= .015 && cv::norm(delta) <= 8*pi/180) selected.push_back(sample);
    }
    inlierCount = selected.size();
    if (selected.size() < std::max<std::size_t>(3, (samples.size() + 1) / 2)) return std::nullopt;
    return average(selected);
}

Json calibrate(const Frames& frames, const Options& config, std::size_t rowCount) {
    using Pair = std::pair<int, int>;
    std::map<Pair, std::vector<Transform>> samples;
    std::map<int, int> observed;
    for (const auto& frame : frames) {
        for (auto left = frame.second.begin(); left != frame.second.end(); ++left) {
            ++observed[left->first];
            for (auto right = std::next(left); right != frame.second.end(); ++right) {
                samples[{left->first, right->first}].push_back(inverseRigid(left->second) * right->second);
            }
        }
    }
    if (!observed.count(config.reference)) throw std::runtime_error("reference ID " + std::to_string(config.reference) + " was not observed");
    std::map<int, std::map<int, Transform>> edges;
    Json edgeReport = Json::object();
    std::size_t accepted = 0;
    for (const auto& pair : samples) {
        const auto [left, right] = pair.first;
        Json report{{"overlap_frames", pair.second.size()}, {"accepted", false}};
        if (pair.second.size() >= static_cast<std::size_t>(config.minOverlap)) {
            std::size_t inliers = 0;
            const auto transform = robustAverage(pair.second, inliers);
            report["inlier_frames"] = inliers;
            if (transform) {
                edges[left][right] = *transform;
                edges[right][left] = inverseRigid(*transform);
                const auto position = translation(*transform);
                report["accepted"] = true;
                report["translation_m"] = {position[0], position[1], position[2]};
                ++accepted;
            }
        }
        edgeReport[std::to_string(left) + "-" + std::to_string(right)] = report;
    }
    std::map<int, Transform> transforms{{config.reference, Transform::eye()}};
    std::queue<int> queue;
    queue.push(config.reference);
    while (!queue.empty()) {
        const int current = queue.front(); queue.pop();
        for (const auto& neighbor : edges[current]) {
            if (transforms.count(neighbor.first)) continue;
            transforms[neighbor.first] = transforms.at(current) * neighbor.second;
            queue.push(neighbor.first);
        }
    }
    std::ostringstream unreachable;
    for (const auto& tag : observed) if (!transforms.count(tag.first)) unreachable << ' ' << tag.first;
    if (!unreachable.str().empty()) {
        throw std::runtime_error("overlap graph is disconnected; unreachable IDs:" + unreachable.str() +
                                 " (accepted edges=" + std::to_string(accepted) + ", min-overlap=" + std::to_string(config.minOverlap) +
                                 "). Record more simultaneous tag observations; no partial rig was written.");
    }
    Json geometry = Json::object(), ids = Json::array(), detections = Json::object();
    for (const auto& tag : transforms) { ids.push_back(tag.first); geometry[std::to_string(tag.first)] = transformToJson(tag.second); }
    for (const auto& tag : observed) detections[std::to_string(tag.first)] = tag.second;
    return {{"reference_id", config.reference}, {"marker_size_m", config.sizeMm/1000},
            {"min_overlap", config.minOverlap}, {"reachable_ids", ids}, {"T_reference_tag", geometry},
            {"accepted_edges", edgeReport}, {"source", config.poses}, {"detections_by_id", detections},
            {"coordinate_convention", "T_reference_tag maps points from each tag frame into the ID reference frame."},
            {"summary", {{"observation_rows", rowCount}, {"frames", frames.size()}, {"tags", observed.size()},
                         {"accepted_edge_count", accepted}, {"connected_overlap_graph", true}}}};
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto config = options(argc, argv);
        std::size_t rowCount = 0;
        const auto frames = loadRows(config.poses, rowCount);
        const auto rig = calibrate(frames, config, rowCount);
        const fs::path output(config.output);
        if (!output.parent_path().empty()) fs::create_directories(output.parent_path());
        std::ofstream stream(output);
        if (!stream) throw std::runtime_error("cannot open rig output: " + config.output);
        stream << rig.dump(2) << '\n';
        stream.close();
        if (!stream) throw std::runtime_error("failed while writing rig output: " + config.output);
        Json summary = rig.at("summary");
        summary["output"] = config.output;
        summary["reference_id"] = config.reference;
        summary["reachable_ids"] = rig.at("reachable_ids");
        std::cout << summary.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "marker_calibrate_rig: " << error.what() << '\n';
        return 1;
    }
}
