#include <dfmarker_api.h>
#include "session_input.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
namespace fs = std::filesystem;
constexpr const char* windowName = "dfmarker - pose / relative XY trajectory";
const cv::Scalar green(100, 220, 110), orange(40, 175, 255), gray(165, 165, 165);

struct Options {
    std::string config, video, package, session, rig, timestamps, csv, outputVideo;
    bool display = true, exitOnFinish = false;
};

void usage() {
    std::cerr << "Usage:\n"
        "  video_demo sdk_config.json video.mp4 [options]\n"
        "  video_demo --package TEST_PACKAGE [--session NAME] [--rig FILE] [options]\n"
        "Options:\n"
        "  --timestamps frames.csv  Use recorded times (package mode reads this automatically)\n"
        "  --csv FILE               Write pose CSV to FILE instead of stdout\n"
        "  --output-video FILE      Save annotated .mp4 or .avi (fixed source FPS)\n"
        "  --no-display             Process without a window, at full speed\n"
        "  --exit-on-finish         Close window after the last frame\n"
        "Controls: Space pause/resume; N / Right next frame; Q / Esc quit.\n";
}

Options parseOptions(int argc, char** argv) {
    Options result;
    std::vector<std::string> positional;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto value = [&]() -> std::string {
            if (++index == argc) throw std::invalid_argument("missing value for " + arg);
            return argv[index];
        };
        if (arg == "--package") result.package = value();
        else if (arg == "--session") result.session = value();
        else if (arg == "--rig") result.rig = value();
        else if (arg == "--timestamps") result.timestamps = value();
        else if (arg == "--csv") result.csv = value();
        else if (arg == "--output-video") result.outputVideo = value();
        else if (arg == "--no-display") result.display = false;
        else if (arg == "--exit-on-finish") result.exitOnFinish = true;
        else if (arg.rfind("--", 0) == 0) throw std::invalid_argument("unknown option: " + arg);
        else positional.push_back(arg);
    }
    if (!result.package.empty()) {
        if (!positional.empty() || !result.timestamps.empty())
            throw std::invalid_argument("package mode accepts no positional files or --timestamps");
    } else {
        if (positional.size() != 2 || !result.session.empty() || !result.rig.empty())
            throw std::invalid_argument("provide SDK config and video, or --package");
        result.config = positional[0]; result.video = positional[1];
    }
    return result;
}

// The C callback's borrowed arrays and strings must be copied before feed returns.
struct Snapshot {
    dfmarker_result_t result{};
    std::string reason;
    std::vector<dfmarker_detection_t> detections;
    std::vector<int32_t> visible, used, rejected;
    std::vector<dfmarker_tag_error_t> errors;
    std::exception_ptr exception;
    bool received = false;
};
template<class T> std::vector<T> copyArray(const T* data, uint32_t count) {
    return count ? std::vector<T>(data, data + count) : std::vector<T>{};
}
void callback(const dfmarker_result_t* result, void* user) noexcept {
    auto& snapshot = *static_cast<Snapshot*>(user);
    try {
        snapshot.result = *result;
        snapshot.reason = result->reason ? result->reason : "";
        snapshot.detections = copyArray(result->detections, result->detection_count);
        snapshot.visible = copyArray(result->visible_ids, result->visible_count);
        snapshot.used = copyArray(result->used_ids, result->used_count);
        snapshot.rejected = copyArray(result->rejected_ids, result->rejected_count);
        snapshot.errors = copyArray(result->tag_errors, result->tag_error_count);
        // No borrowed pointers escape into the rendering code.
        snapshot.result.reason = nullptr;
        snapshot.result.detections = nullptr;
        snapshot.result.visible_ids = snapshot.result.used_ids = snapshot.result.rejected_ids = nullptr;
        snapshot.result.tag_errors = nullptr;
        snapshot.received = true;
    } catch (...) { snapshot.exception = std::current_exception(); }
}

const char* qualityName(dfmarker_quality_t quality) {
    switch (quality) {
        case DFMARKER_SINGLE: return "SINGLE";
        case DFMARKER_MULTI: return "MULTI";
        case DFMARKER_WARN: return "WARN";
        default: return "LOST";
    }
}
std::string fixed(double value, int precision = 3) {
    std::ostringstream stream; stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}
std::string ids(const std::vector<int32_t>& values) {
    std::ostringstream stream;
    for (std::size_t i = 0; i < values.size(); ++i) stream << (i ? "," : "") << values[i];
    return values.empty() ? "-" : stream.str();
}
std::string vector3(const double* values) {
    return fixed(values[0]) + "  " + fixed(values[1]) + "  " + fixed(values[2]);
}
std::string displayReason(const Snapshot& snapshot) {
    // Hershey fonts are ASCII-only; retain the SDK reason and translate for display.
    const auto& reason = snapshot.reason;
    if (reason == "没有检测到已标定 Tag") return "No registered Tag detected";
    if (reason == "Tag 太小、靠近画面边缘或位姿求解失败") return "Tag small / near edge / pose solve failed";
    if (reason == "重投影误差过大，已拒绝该帧") return "Frame rejected: reprojection error";
    if (reason == "短时位姿突跳，轨迹在此断开") return "Pose jump: new trajectory segment";
    if (reason == "多面角点联合定位") return "Joint pose from multiple Tag faces";
    if (reason == "单面定位，深度与姿态约束较弱") return "Single face: weaker depth/orientation";
    if (std::any_of(reason.begin(), reason.end(), [](unsigned char c) { return c > 127; }))
        return snapshot.result.has_pose ? "Pose accepted" : "No accepted pose";
    return reason.empty() ? "OK" : reason;
}
cv::Matx33d rotation(const dfmarker_pose_t& pose) {
    double x = pose.orientation[0], y = pose.orientation[1], z = pose.orientation[2], w = pose.orientation[3];
    const double norm = std::sqrt(x*x + y*y + z*z + w*w);
    if (!std::isfinite(norm) || norm < 1e-12) throw std::runtime_error("invalid pose quaternion");
    x /= norm; y /= norm; z /= norm; w /= norm;
    return {1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
            2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
            2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)};
}
std::array<double, 3> rpy(const dfmarker_pose_t& pose) {
    const auto r = rotation(pose);
    const double pitch = std::asin(std::clamp(-r(2,0), -1.0, 1.0));
    const bool singular = std::abs(std::cos(pitch)) < 1e-8;
    return {180.0/CV_PI * (singular ? std::atan2(-r(1,2), r(1,1)) : std::atan2(r(2,1), r(2,2))),
            180.0/CV_PI * pitch, 180.0/CV_PI * (singular ? 0.0 : std::atan2(r(1,0), r(0,0)))};
}

// Match the SDK's camera projection, including RealSense inverse Brown semantics.
bool project(const cv::Vec3d& point, const demo::CameraModel& camera, cv::Point& pixel) {
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2]) || point[2] <= 0)
        return false;
    cv::Point2d output;
    if (camera.distortionModel == "inverse_brown_conrady") {
        float x = static_cast<float>(point[0]) / static_cast<float>(point[2]);
        float y = static_cast<float>(point[1]) / static_cast<float>(point[2]);
        const auto& d = camera.distortion;
        const float k0 = d[0], k1 = d[1], k2 = d[2], k3 = d[3], k4 = d[4];
        const float radius = x*x + y*y;
        const float radial = 1 + k0*radius + k1*radius*radius + k4*radius*radius*radius;
        x *= radial; y *= radial;
        const float correctedX = x + 2*k2*x*y + k3*(radius + 2*x*x);
        const float correctedY = y + 2*k3*x*y + k2*(radius + 2*y*y);
        output = {correctedX * static_cast<float>(camera.matrix(0,0)) + static_cast<float>(camera.matrix(0,2)),
                  correctedY * static_cast<float>(camera.matrix(1,1)) + static_cast<float>(camera.matrix(1,2))};
    } else {
        std::vector<cv::Point2d> pixels;
        cv::projectPoints(std::vector<cv::Point3d>{{point[0], point[1], point[2]}},
                          cv::Vec3d{}, cv::Vec3d{}, camera.matrix, camera.distortion, pixels);
        output = pixels.front();
    }
    if (!std::isfinite(output.x) || !std::isfinite(output.y) ||
        std::abs(output.x) > 1000000 || std::abs(output.y) > 1000000) return false;
    pixel = {cvRound(output.x), cvRound(output.y)};
    return true;
}
void drawAxes(cv::Mat& frame, const dfmarker_pose_t& pose, const demo::CameraModel& camera) {
    const auto r = rotation(pose);
    const cv::Vec3d origin(pose.position[0], pose.position[1], pose.position[2]);
    cv::Point start;
    if (!project(origin, camera, start)) return;
    const std::array<cv::Scalar, 3> colors{cv::Scalar(50,50,255), cv::Scalar(60,230,60), cv::Scalar(255,140,50)};
    for (int index = 0; index < 3; ++index) {
        cv::Vec3d axis{}; axis[index] = camera.markerSizeM * 1.25;
        cv::Point end;
        if (!project(origin + r*axis, camera, end)) continue;
        auto clippedStart = start, clippedEnd = end;
        if (cv::clipLine(frame.size(), clippedStart, clippedEnd))
            cv::arrowedLine(frame, clippedStart, clippedEnd, colors[index], 3, cv::LINE_AA, 0, 0.12);
        if (cv::Rect({}, frame.size()).contains(end))
            cv::putText(frame, std::string(1, "XYZ"[index]), end + cv::Point(5,-5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, colors[index], 2, cv::LINE_AA);
    }
}
void text(cv::Mat& image, const std::string& value, int x, int y, double scale = 0.52,
          cv::Scalar color = cv::Scalar(225,225,225), int thickness = 1) {
    cv::putText(image, value, {x,y}, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv::LINE_AA);
}
struct TracePoint { double x, y; uint64_t segment; };
void drawTrace(cv::Mat& panel, const std::deque<TracePoint>& trace, const Snapshot& snapshot) {
    const cv::Rect box(18, 490, panel.cols - 36, 200);
    cv::rectangle(panel, box, cv::Scalar(65,65,65));
    text(panel, "Relative XY (m), +Y up", box.x+8, box.y+19, 0.47);
    double xmin=0, xmax=0, ymin=0, ymax=0;
    for (const auto& point : trace) {
        xmin=std::min(xmin,point.x); xmax=std::max(xmax,point.x);
        ymin=std::min(ymin,point.y); ymax=std::max(ymax,point.y);
    }
    const double span = std::max({0.10, xmax-xmin, ymax-ymin}) * 1.2;
    const double centerX=(xmin+xmax)/2, centerY=(ymin+ymax)/2;
    const double scale=std::min(box.width-32,box.height-54)/span;
    auto pixel = [&](double x, double y) { return cv::Point(
        cvRound(box.x+box.width/2.0+(x-centerX)*scale),
        cvRound(box.y+box.height/2.0+10-(y-centerY)*scale)); };
    cv::drawMarker(panel,pixel(0,0),gray,cv::MARKER_CROSS,10,1,cv::LINE_AA);
    for (std::size_t i=1; i<trace.size(); ++i)
        if (trace[i].segment==trace[i-1].segment)
            cv::line(panel,pixel(trace[i-1].x,trace[i-1].y),pixel(trace[i].x,trace[i].y),green,1,cv::LINE_AA);
    if (snapshot.result.has_pose && !trace.empty())
        cv::circle(panel,pixel(trace.back().x,trace.back().y),4,orange,cv::FILLED,cv::LINE_AA);
    text(panel,"Span: " + fixed(span) + " m   cross = origin",box.x+8,box.y+box.height-8,0.43,gray);
}
cv::Mat render(const cv::Mat& pixels, const Snapshot& snapshot, const demo::SessionInput& input,
               const std::deque<TracePoint>& trace, uint64_t count, uint64_t total) {
    cv::Mat frame;
    if (pixels.channels()==1) cv::cvtColor(pixels,frame,cv::COLOR_GRAY2BGR);
    else pixels.copyTo(frame);
    std::vector<cv::Rect> labels;
    for (const auto& detection : snapshot.detections) {
        if (!detection.has_corners) continue;
        auto contains = [&](const auto& values) { return std::find(values.begin(),values.end(),detection.tag_id)!=values.end(); };
        const auto color=contains(snapshot.used)?green:(contains(snapshot.rejected)?orange:gray);
        std::vector<cv::Point> corners;
        for (int index=0; index<4; ++index)
            corners.emplace_back(cvRound(detection.corners_px[2*index]),cvRound(detection.corners_px[2*index+1]));
        cv::polylines(frame,corners,true,color,2,cv::LINE_AA);
        std::string label="ID " + std::to_string(detection.tag_id);
        for (const auto& error : snapshot.errors)
            if (error.tag_id==detection.tag_id && error.has_error)
                label += " / " + fixed(error.reprojection_error_px,2) + "px";
        int baseline=0;
        const auto size=cv::getTextSize(label,cv::FONT_HERSHEY_SIMPLEX,0.50,1,&baseline);
        const int width=std::min(frame.cols,size.width+12),height=size.height+baseline+10;
        cv::Rect labelBox(std::clamp(corners[0].x,0,frame.cols-width),
                          std::clamp(corners[0].y-height-8,0,std::max(0,frame.rows-height)),width,height);
        const int initialY=labelBox.y;
        // Adjacent faces often have nearly identical top corners. Stack their labels.
        for (std::size_t attempt=0; attempt<2*snapshot.detections.size()+2; ++attempt) {
            const int offset=static_cast<int>((attempt+1)/2)*(height+4)*(attempt%2?-1:1);
            const int y=initialY+offset;
            if (y<0 || y+height>frame.rows) continue;
            const cv::Rect candidate(labelBox.x,y,width,height);
            if (std::none_of(labels.begin(),labels.end(),[&](const cv::Rect& box) { return (box & candidate).area()>0; })) {
                labelBox=candidate; break;
            }
        }
        labels.push_back(labelBox);
        cv::line(frame,corners[0],labelBox.tl()+cv::Point(4,height),color,1,cv::LINE_AA);
        cv::rectangle(frame,labelBox,cv::Scalar(25,25,25),cv::FILLED);
        text(frame,label,labelBox.x+6,labelBox.y+size.height+5,0.50,color);
    }
    const auto& r=snapshot.result;
    if (r.has_pose) drawAxes(frame,r.camera_pose,input.camera);
    cv::Mat canvas(std::max(frame.rows,720),frame.cols+440,CV_8UC3,cv::Scalar(24,24,24));
    frame.copyTo(canvas(cv::Rect(0,0,frame.cols,frame.rows)));
    auto panel=canvas(cv::Rect(frame.cols,0,440,canvas.rows));
    text(panel,"DFMARKER POSE",18,30,0.74,cv::Scalar(245,245,245),2);
    const auto qualityColor=r.has_pose?green:orange;
    text(panel,qualityName(r.quality),18,67,0.85,qualityColor,2);
    text(panel,"Frame " + std::to_string(count) + (total?" / "+std::to_string(total):"") +
         "    t=" + fixed(r.timestamp_ns*1e-9) + " s",18,95);
    text(panel,"Segment " + std::to_string(r.segment) + "    reference ID " + std::to_string(input.referenceId),18,121);
    text(panel,"Visible: " + ids(snapshot.visible),18,147);
    text(panel,"Used: " + ids(snapshot.used) + "   Rejected: " + ids(snapshot.rejected),18,173);
    text(panel,"Reprojection: " + (r.has_reprojection_error?fixed(r.reprojection_error_px,3)+" px":"--"),18,199);
    text(panel,"Relative XYZ (m)",18,235,0.52,gray);
    text(panel,r.has_pose?vector3(r.relative_pose.position):"--  no current pose",18,259,0.60,qualityColor);
    text(panel,"Relative roll / pitch / yaw (deg, ZYX)",18,286,0.48,gray);
    const auto angles=r.has_pose?rpy(r.relative_pose):std::array<double,3>{};
    text(panel,r.has_pose?vector3(angles.data()):"--",18,310,0.58);
    text(panel,"Relative quaternion (x y z w)",18,337,0.5,gray);
    std::string quaternion="--";
    if (r.has_pose) { quaternion.clear(); for (double q:r.relative_pose.orientation) quaternion+=(quaternion.empty()?"":"  ")+fixed(q); }
    text(panel,quaternion,18,361,0.50);
    text(panel,"Reference in camera XYZ (m)",18,388,0.50,gray);
    text(panel,r.has_pose?vector3(r.camera_pose.position):"--",18,412,0.55);
    // Wrap the reason so long quality explanations remain readable.
    const std::string reason=displayReason(snapshot);
    text(panel,reason.substr(0,49),18,442,0.43,qualityColor);
    if (reason.size()>49) text(panel,reason.substr(49,49),18,462,0.43,qualityColor);
    drawTrace(panel,trace,snapshot);
    text(panel,"Green: used  Orange: rejected  Gray: other",18,710,0.43,gray);
    return canvas;
}

void protectOutput(const std::string& output, const demo::SessionInput& input, const Options& options) {
    if (output.empty()) return;
    const auto path=fs::weakly_canonical(output);
    const std::vector<std::string> inputs{input.videoPath,input.configPath,input.rigPath,options.timestamps,
        (fs::path(input.videoPath).parent_path()/"metadata.json").string(),
        (fs::path(input.videoPath).parent_path()/"frames.csv").string(),
        (fs::path(input.videoPath).parent_path()/"timestamps.csv").string()};
    for (const auto& source:inputs)
        if (!source.empty() && path==fs::weakly_canonical(source))
            throw std::invalid_argument("output would overwrite input: " + output);
    if (!options.csv.empty() && !options.outputVideo.empty() &&
        fs::weakly_canonical(options.csv)==fs::weakly_canonical(options.outputVideo))
        throw std::invalid_argument("CSV and video output must be different files");
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
}
void writeCsv(std::ostream& out,const dfmarker_result_t& r) {
    out << r.timestamp_ns << ',' << r.frame_index << ',' << r.segment << ',' << r.quality << ',' << r.has_pose;
    if (r.has_pose) {
        for (double p:r.relative_pose.position) out << ',' << p;
        for (double q:r.relative_pose.orientation) out << ',' << q;
    } else out << ",,,,,,,";
    out << ',';
    if (r.has_reprojection_error) out << r.reprojection_error_px;
    out << '\n';
}
bool windowClosed() {
    try {
        const double visible=cv::getWindowProperty(windowName,cv::WND_PROP_VISIBLE);
        if (visible>=0) return visible<1;
        // OpenCV 4.5's GTK backend does not implement WND_PROP_VISIBLE.
        const auto area=cv::getWindowImageRect(windowName);
        return area.width<=0 || area.height<=0;
    } catch (const cv::Exception&) { return true; }
}
} // namespace

int main(int argc,char** argv) {
    if (argc==1 || (argc==2 && std::string(argv[1])=="--help")) { usage(); return argc==1?2:0; }
    DFMARKER_HANDLE handle=nullptr;
    try {
        const auto options=parseOptions(argc,argv);
        const auto input=options.package.empty()?demo::loadVideoInput(options.config,options.video,options.timestamps):
            demo::loadPackageInput(options.package,options.session,options.rig);
        protectOutput(options.csv,input,options); protectOutput(options.outputVideo,input,options);
        cv::VideoCapture video(input.videoPath);
        const double fps=video.get(cv::CAP_PROP_FPS);
        if (!video.isOpened() || !std::isfinite(fps) || fps<=0) throw std::runtime_error("cannot read video/FPS");
        const double reportedCount=video.get(cv::CAP_PROP_FRAME_COUNT);
        const uint64_t total=input.hasRecordedTimes?input.timestampsNs.size():
            (std::isfinite(reportedCount) && reportedCount>0 && reportedCount<1e12?static_cast<uint64_t>(std::llround(reportedCount)):0);
        std::ofstream csvFile;
        if (!options.csv.empty()) { csvFile.open(options.csv); if (!csvFile) throw std::runtime_error("cannot open CSV output"); }
        std::ostream& csv=options.csv.empty()?std::cout:csvFile;
        csv << std::setprecision(12) << "timestamp_ns,frame_index,segment,quality,has_pose,x,y,z,qx,qy,qz,qw,reprojection_error_px\n";
        Snapshot snapshot;
        if (!dfmarker_create(input.configPath.c_str(),&handle) ||
            !dfmarker_register_result_callback(handle,callback,&snapshot) || !dfmarker_start(handle))
            throw std::runtime_error("SDK initialization failed");
        std::cerr << "Video: " << input.videoPath << "\nRig: " << input.rigPath << "\nMarker: "
                  << input.camera.markerSizeM*1000 << " mm\nTime: " << input.timestampSource << '\n';
        if (options.display) {
            cv::namedWindow(windowName,cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
            cv::resizeWindow(windowName,1380,600);
            std::cerr << "Space pause/resume; N / Right next; Q / Esc quit.\n";
        }
        cv::VideoWriter writer;
        cv::Mat pixels,canvas;
        std::deque<TracePoint> trace;
        uint64_t count=0,measured=0;
        bool paused=false,quit=false,finished=false,step=false;
        auto nextFrameAt=std::chrono::steady_clock::now();
        for (;;) {
            if ((!paused || step) && !finished) {
                step=false;
                const auto frameStarted=std::chrono::steady_clock::now();
                if (!video.read(pixels)) {
                    if (!count) throw std::runtime_error("no video frames");
                    if (total && count!=total) throw std::runtime_error("decoded frame count differs from video/timestamps");
                    finished=true;
                    if (!options.display || options.exitOnFinish) break;
                    std::cerr << "End of video. Q / Esc to close.\n";
                } else {
                    if (input.hasRecordedTimes && count>=input.timestampsNs.size()) throw std::runtime_error("video has more frames than timestamp CSV");
                    const long double ns=static_cast<long double>(count)*1000000000.0L/fps;
                    if (ns>=std::ldexp(1.0L,64)) throw std::runtime_error("video duration overflow");
                    const auto timestamp=input.hasRecordedTimes?input.timestampsNs[count]:static_cast<uint64_t>(std::round(ns));
                    if (pixels.type()!=CV_8UC1 && pixels.type()!=CV_8UC3) throw std::runtime_error("unsupported decoded pixel format");
                    if (pixels.cols!=input.camera.width || pixels.rows!=input.camera.height) throw std::runtime_error("video dimensions differ from calibrated camera");
                    snapshot.received=false;
                    const dfmarker_image_t image{timestamp,static_cast<uint32_t>(pixels.cols),static_cast<uint32_t>(pixels.rows),
                        pixels.step[0],pixels.data,pixels.step[0]*pixels.rows,pixels.channels()==1?DFMARKER_MONO8:DFMARKER_BGR8};
                    if (!dfmarker_feed_image(handle,&image)) throw std::runtime_error("SDK rejected video frame " + std::to_string(count));
                    if (snapshot.exception) std::rethrow_exception(snapshot.exception);
                    if (!snapshot.received) throw std::runtime_error("SDK did not return a frame result");
                    writeCsv(csv,snapshot.result);
                    if (!csv) throw std::runtime_error("failed to write pose CSV");
                    ++count;
                    if (snapshot.result.has_pose) {
                        ++measured;
                        trace.push_back({snapshot.result.relative_pose.position[0],snapshot.result.relative_pose.position[1],snapshot.result.segment});
                        if (trace.size()>3000) trace.pop_front();
                    }
                    if (options.display || !options.outputVideo.empty()) {
                        canvas=render(pixels,snapshot,input,trace,count,total);
                        if (!options.outputVideo.empty()) {
                            if (!writer.isOpened()) {
                                const auto extension=fs::path(options.outputVideo).extension().string();
                                const int codec=extension==".avi"?cv::VideoWriter::fourcc('M','J','P','G'):cv::VideoWriter::fourcc('m','p','4','v');
                                if (!writer.open(options.outputVideo,codec,fps,canvas.size())) throw std::runtime_error("cannot open annotated video output");
                            }
                            writer.write(canvas);
                        }
                    }
                    const uint64_t nextTime=input.hasRecordedTimes && count<input.timestampsNs.size()?input.timestampsNs[count]:
                        timestamp+static_cast<uint64_t>(1000000000.0/fps);
                    const double delay=std::min(1.0,(nextTime-timestamp)*1e-9);
                    nextFrameAt=frameStarted+std::chrono::milliseconds(static_cast<int>(delay*1000));
                }
            }
            if (!options.display) continue;
            cv::Mat displayed=canvas.clone();
            const std::string state=finished?"END - Q / Esc to close":paused?"PAUSED - Space resume / N next":"Space pause | N next | Q quit";
            cv::rectangle(displayed,{0,std::max(0,displayed.rows-30),std::min(550,displayed.cols),30},cv::Scalar(25,25,25),cv::FILLED);
            text(displayed,state,10,displayed.rows-9,0.50,orange);
            cv::imshow(windowName,displayed);
            // Poll frequently enough that pause, step and window-close remain responsive.
            for (;;) {
                const int key=cv::waitKeyEx(10);
                if (windowClosed() || key==27 || key=='q' || key=='Q') { quit=true; break; }
                if (finished) continue;
                if (key==' ') { paused=!paused; nextFrameAt=std::chrono::steady_clock::now(); break; }
                else if (key=='n' || key=='N' || key==0x270000 || key==65363 || key==0x1000014) {
                    paused=true; step=true; break;
                }
                if (!paused && !finished && std::chrono::steady_clock::now()>=nextFrameAt) break;
            }
            if (quit) break;
        }
        writer.release();
        csv.flush();
        if (!csv) throw std::runtime_error("failed to flush pose CSV");
        if (!dfmarker_stop(handle) || !dfmarker_destroy(&handle)) throw std::runtime_error("SDK shutdown failed");
        if (options.display) cv::destroyAllWindows();
        std::cerr << "Processed " << count << " frames; poses " << measured << " (" << fixed(100.0*measured/count,2)
                  << "%); segments " << snapshot.result.segment << (quit?"; stopped by user":"; complete") << '\n';
        return 0;
    } catch (const std::exception& error) {
        if (handle) dfmarker_destroy(&handle);
        std::cerr << "video_demo: " << error.what() << '\n';
        return 1;
    }
}
