#include "video_tracker.hpp"
#include <algorithm>
#include "pairings.hpp"
#include "geometry_utils.hpp"

std::vector<cv::KeyPoint> &VideoTracker::get_current_frame_keypoints(cv::Mat &img)
{
    current_frame_keypoints.clear();
    blob_detector->detect(img, current_frame_keypoints);
    return current_frame_keypoints;
}

std::vector<TrackerSingleFrameInfo> &VideoTracker::get_current_tracker_infos(cv::Mat &img)
{
    current_frame_infos.clear();
    cv::Mat &img_ref_copy = img;
    std::transform(trackers.begin(), trackers.end(), std::back_inserter(current_frame_infos), [&](Tracker &tracker)
                   { tracker.track(img_ref_copy); });
}

static std::vector<std::pair<cv::KeyPoint &, TrackerSingleFrameInfo &>> pair_up_trackers(
    std::vector<cv::KeyPoint> &keypoints,
    std::vector<TrackerSingleFrameInfo> &info)
{
    return PairingUtils::pair_with_nearest<cv::KeyPoint, TrackerSingleFrameInfo>(
        keypoints,
        info,
        [](cv::KeyPoint const &keypoint, TrackerSingleFrameInfo const &info_a, TrackerSingleFrameInfo const &info_b)
        {
            return GeometryUtils::distance_squared(keypoint.pt, info_a.current.center()) < GeometryUtils::distance_squared(keypoint.pt, info_b.current.center());
        });
}


