#include <dfmarker_api.h>
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
#include <opencv2/objdetect/aruco_detector.hpp>
#else
#include <opencv2/aruco.hpp>
#endif
#include <iostream>

struct Results { int count = 0; bool pose = false; bool lost = false; };
static void callback(const dfmarker_result_t *result, void *data) {
    auto& out = *static_cast<Results*>(data);
    ++out.count;
    if (result->frame_index == 0)
        out.pose = result->has_pose && result->quality == DFMARKER_SINGLE && result->used_count == 1 &&
            result->used_ids[0] == 2 && result->segment == 1;
    if (result->frame_index == 1) out.lost = !result->has_pose && result->quality == DFMARKER_LOST;
}
int main(int argc, char **argv) {
    if (argc != 2) { std::cerr << "usage: sdk_smoke sdk_demo.json\n"; return 2; }
    cv::Mat image(480, 640, CV_8UC1, cv::Scalar(255)), tag;
    const auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
    cv::aruco::generateImageMarker(dictionary, 2, 120, tag);
#else
    cv::aruco::drawMarker(dictionary, 2, 120, tag);
#endif
    tag.copyTo(image(cv::Rect(260, 180, 120, 120)));
    DFMARKER_HANDLE handle = nullptr;
    Results results;
    bool success = dfmarker_create(argv[1], &handle) &&
        dfmarker_register_result_callback(handle, callback, &results) && dfmarker_start(handle);
    dfmarker_image_t frame{0, 640, 480, image.step[0], image.data, image.step[0] * 480, DFMARKER_MONO8};
    if (success) success = dfmarker_feed_image(handle, &frame);
    image.setTo(255);
    frame.timestamp_ns = 33333333;
    if (success) success = dfmarker_feed_image(handle, &frame);
    if (handle) dfmarker_destroy(&handle);
    success = success && results.count == 2 && results.pose && results.lost;
    std::cout << (success ? "PASS: installed C++ consumer, ArUco pose and loss\n" : "FAIL: SDK smoke\n");
    return success ? 0 : 1;
}
