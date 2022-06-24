#include "tracker_single_frame_info.hpp"
#include "geometry_utils.hpp"
#include <opencv2/imgproc.hpp>
#include "tracker.hpp"

float TrackerSingleFrameInfo::inertia()
{
    return GeometryUtils::distance_squared(current.center(), previous.center());
}

void TrackerSingleFrameInfo::reset_tracker_position(cv::Mat &img, cv::KeyPoint keypoint)
{
    current = tracker.set_tracker_position(img, keypoint);
}