cmake_minimum_required(VERSION 3.15.1)

project(drostracker)

set(CMAKE_CXX_STANDARD 17)
FIND_PACKAGE(OpenCV REQUIRED)
add_executable(drostracker src/main.cpp src/video_tracker.cpp src/tracker_single_frame_info.cpp
src/color_picker.cpp)

# add_library(tracking src/video_tracker.cpp)
INCLUDE_DIRECTORIES(include/)
TARGET_LINK_LIBRARIES(drostracker ${OpenCV_LIBS})
set(CMAKE_CXX_FLAGS_RELEASE)