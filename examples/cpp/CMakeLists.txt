cmake_minimum_required(VERSION 3.16)
project(dfmarker_cpp_consumer LANGUAGES CXX)
find_package(dfmarker 0.1 CONFIG REQUIRED)
# Programs sharing OpenCV with the SDK select the same build version.
find_package(OpenCV ${dfmarker_OPENCV_VERSION} EXACT REQUIRED COMPONENTS core imgproc)
if(OpenCV_VERSION VERSION_LESS 4.7)
    set(marker_component aruco)
else()
    set(marker_component objdetect)
endif()
find_package(OpenCV ${dfmarker_OPENCV_VERSION} EXACT REQUIRED COMPONENTS
    core imgproc videoio calib3d highgui ${marker_component})
add_executable(sdk_smoke sdk_smoke.cpp)
target_link_libraries(sdk_smoke PRIVATE dfmarker::dfmarker opencv_core opencv_${marker_component})
add_executable(video_demo video_demo.cpp session_input.cpp)
# GUI/video dependencies belong to this example, never to libdfmarker.
target_link_libraries(video_demo PRIVATE dfmarker::dfmarker
    opencv_core opencv_imgproc opencv_calib3d opencv_videoio opencv_highgui)
foreach(example sdk_smoke video_demo)
    target_compile_features(${example} PRIVATE cxx_std_17)
    target_include_directories(${example} PRIVATE ${OpenCV_INCLUDE_DIRS})
endforeach()
