#pragma once
#include "bbox.hpp"

class Tracker;
class BBox;

class TrackerSingleFrameInfo
{
public:
    bool isSuccessful;
    BBox current;
    BBox previous;
    Tracker &tracker;

    TrackerSingleFrameInfo(bool isSuccessful,
                           BBox current,
                           BBox previous, Tracker &tracker) : isSuccessful(isSuccessful), current(current), previous(previous), tracker(tracker)
    {
    }

    float inertia();

    void reset_tracker_position(cv::Mat& img, cv::KeyPoint keypoint);
    void reset_tracker_position(cv::Mat& img, BBox bbox);
    static void swap_tracker_positions(cv::Mat& img, TrackerSingleFrameInfo& first, TrackerSingleFrameInfo& second);
};
