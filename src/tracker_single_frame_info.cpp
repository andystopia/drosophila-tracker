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

void TrackerSingleFrameInfo::reset_tracker_position(cv::Mat &img, BBox box)
{
    current = tracker.set_tracker_position(img, box.as_rect());
}

void TrackerSingleFrameInfo::swap_tracker_positions(cv::Mat &img, TrackerSingleFrameInfo &first, TrackerSingleFrameInfo &second)
{
    BBox first_location = first.current;
    BBox second_location = second.current;

    first.reset_tracker_position(img, second_location);
    second.reset_tracker_position(img, first_location);
}