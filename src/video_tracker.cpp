#include "video_tracker.hpp"
#include <algorithm>
#include "pairings.hpp"
#include "geometry_utils.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/features2d.hpp>
#include "color_picker.hpp"

using TrackerSingleFrameInfoRef = std::reference_wrapper<TrackerSingleFrameInfo>;
using KeyPointRef = std::reference_wrapper<cv::KeyPoint>;

VideoTracker::VideoTracker(cv::Mat &img, cv::Ptr<cv::SimpleBlobDetector> detector)
{
    // will initialize the current frame's keypoints.
    // we want to allow it to reuse that allocation
    // across frames
    blob_detector = detector;
    detect_current_frame_keypoints(img);
    trackers = initialize_trackers(img, current_frame_keypoints);
}

std::vector<cv::KeyPoint> &VideoTracker::detect_current_frame_keypoints(cv::Mat &img)
{
    current_frame_keypoints.clear();
    blob_detector->detect(img, current_frame_keypoints);
    return current_frame_keypoints;
}

std::vector<Tracker> VideoTracker::initialize_trackers(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints)
{
    current_frame_infos.clear();
    trackers.reserve(keypoints.size());

    for (cv::KeyPoint &keypoint : keypoints)
    {
        Tracker tracker(img, keypoint);
        trackers.push_back(tracker);
    }

    return trackers;
}

std::vector<TrackerSingleFrameInfo> &VideoTracker::detect_current_tracker_positions(cv::Mat &img)
{
    current_frame_infos.clear();

    // std::cout << " Detecting " << trackers.size() << " # of keypoints" << std::endl;
    for (Tracker &tracker : trackers)
    {
        current_frame_infos.push_back(tracker.track(img));
    }

    return current_frame_infos;
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

void VideoTracker::fix_lost_trackers(cv::Mat &img, std::vector<TrackerSingleFrameInfo> &tracks, std::vector<cv::KeyPoint> &points)
{
    // first find all the trackers which are "lost"

    std::vector<TrackerSingleFrameInfoRef> lost;

    for (TrackerSingleFrameInfo &tracker : tracks)
    {
        if (!tracker.isSuccessful)
        {
            lost.push_back(std::ref(tracker));
        }
    }

    // now that we have the lost trackers,
    // we will want to search for points which are not bounded by any trackers.

    std::vector<std::pair<cv::KeyPoint &, TrackerSingleFrameInfo &>> paired = pair_up_trackers(points, tracks);

    std::vector<KeyPointRef> unpaired_keypoints;

    for (auto &[keypoint, tracker] : paired)
    {
        // if a keypoints nearest tracker does
        // not contain that trakcer, then we must have a
        // lost tracker.
        // we also don't want to correct for trackers which are off
        // track in this step, we only want to correct trackers which are
        // are lost.
        if (!tracker.current.contains(keypoint.pt) && !tracker.isSuccessful)
        {
            unpaired_keypoints.push_back(std::ref(keypoint));
        }
    }

    // for now we will pair the tracker
    // with the nearest unowned blob
    // this will come at the advantage of simplicity
    // with the disadvantage being that
    // we will allow assigning multiple trackers to the same blob
    // rather than to unique blobs.
    // this is an optimal use case for the hungarian (kuhn-munkres)
    // algorithm but for simplicity's sake, we'll avoid that for now.

    std::vector<std::pair<TrackerSingleFrameInfoRef &, KeyPointRef &>> pairings = PairingUtils::pair_with_nearest_value<
        TrackerSingleFrameInfoRef,
        KeyPointRef,
        float>(lost, unpaired_keypoints,
               [](const TrackerSingleFrameInfoRef &tracker, const KeyPointRef &keypoint)
               { return tracker.get().previous.distance_squared(keypoint.get().pt); });

    for (auto &[tracker, keypoint] : pairings)
    {
        tracker.get().reset_tracker_position(img, keypoint.get());
        tracker.get().isSuccessful = true;
    }
}

void VideoTracker::write_frame_data()
{
    for (size_t i = 0; i < trackers.size(); i++)
    {
        trackers[i].append(current_frame_infos[i].current);
    }
}

void VideoTracker::process_frame(cv::Mat &img)
{
    detect_current_frame_keypoints(img);
    detect_current_tracker_positions(img);
    fix_lost_trackers(img, current_frame_infos, current_frame_keypoints);
    fix_overlaps(img);
    write_frame_data();
}

void VideoTracker::fix_overlaps(cv::Mat &img)
{
    // first collect the different points which
    // the keypoints are contained within.

    // scary type nesting

    using KeypointToTrackerInfoVec = std::pair<cv::KeyPoint &, std::vector<TrackerSingleFrameInfoRef>>;
    using KeypointToTrackerInfoVecRef = std::reference_wrapper<KeypointToTrackerInfoVec>;

    std::vector<KeypointToTrackerInfoVec> keypoint_mappings_to_trackers;
    keypoint_mappings_to_trackers.reserve(current_frame_keypoints.size());

    for (cv::KeyPoint &keypoint : current_frame_keypoints)
    {
        std::vector<TrackerSingleFrameInfoRef> trackers;
        for (TrackerSingleFrameInfo &info : current_frame_infos)
        {
            if (info.current.contains(keypoint.pt))
            {
                trackers.push_back(std::ref(info));
            }
        }
        keypoint_mappings_to_trackers.push_back({keypoint, trackers});
    }

    std::vector<KeypointToTrackerInfoVecRef> zeroes;

    for (KeypointToTrackerInfoVec &entry : keypoint_mappings_to_trackers)
    {
        if (entry.second.size() == 0)
        {
            zeroes.push_back(std::ref(entry));
        }
    }

    std::vector<KeypointToTrackerInfoVecRef> more_than_one;

    for (KeypointToTrackerInfoVec &entry : keypoint_mappings_to_trackers)
    {
        if (entry.second.size() > 1)
        {
            more_than_one.push_back(std::ref(entry));
        }
    }

    // i have written shorter functions than this monster
    std::vector<std::pair<KeypointToTrackerInfoVecRef &, KeypointToTrackerInfoVecRef &>>
        zeroes_to_more =
            PairingUtils::pair_with_nearest_value<
                KeypointToTrackerInfoVecRef,
                KeypointToTrackerInfoVecRef,
                float>(
                zeroes,
                more_than_one,
                [](const KeypointToTrackerInfoVecRef &map_a, const KeypointToTrackerInfoVecRef &map_b)
                {
                    return GeometryUtils::distance_squared(map_a.get().first.pt, map_b.get().first.pt);
                });

    for (std::pair<KeypointToTrackerInfoVecRef &, KeypointToTrackerInfoVecRef &> &item : zeroes_to_more)
    {
        const KeypointToTrackerInfoVecRef &zero = item.first.get();
        const KeypointToTrackerInfoVecRef &more_than_one = item.second.get();

        cv::KeyPoint &keypoint = zero.get().first;
        std::vector<TrackerSingleFrameInfoRef> &info = more_than_one.get().second;
        std::sort(info.begin(), info.end(), [&](const auto &info_a, const auto &info_b)
                  { 
            TrackerSingleFrameInfo inf_a = info_a.get();
            TrackerSingleFrameInfo inf_b = info_b.get();

            float dist_a = inf_a.current.distance_squared(keypoint.pt);
            float dist_b = inf_b.current.distance_squared(keypoint.pt);


            return inf_a.inertia() < inf_b.inertia(); });

        info.at(info.size() - 1).get().reset_tracker_position(img, zero.get().first);
    }
}

void VideoTracker::view_current_frame(cv::Mat &frame, const std::string &window_name)
{
    size_t num = 0;
    for (TrackerSingleFrameInfo &info : current_frame_infos)
    {
        cv::rectangle(frame, info.current.as_rect(),
                      ColorPicker::get_color(num), 2, 1);
        num++;
    }

    for (cv::KeyPoint &kp : current_frame_keypoints)
    {
        cv::circle(frame, kp.pt, kp.size, cv::Scalar(0, 255, 0));
    }
    cv::imshow(window_name, frame);
}
