#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/features2d.hpp>

#include "bbox.hpp"

class Tracker;

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
};

class Tracker
{
private:
    cv::Ptr<cv::Tracker> tracker;
    std::vector<BBox> previous_frames;

private:
    cv::Rect get_last_bbox()
    {
        return previous_frames.back();
    }

    static cv::Rect create_rect_from_circle(cv::Point2f point, float radius, float radius_k = 0.80f)
    {
        float size = keypoint.size;
        float size_k = 0.80f;

        return cv::Rect(point.x - size * size_k, point.y - size * size_k, size * size_k * 2, size * size_k * 2);
    }
    static cv::Rect create_rect_from_keypoint(cv::KeyPoint keypoint)
    {
        return create_rect_from_circle(keypoint.pt, keypoint.size)
    }

public:
    void set_tracker_position(cv::Mat &img, cv::Point2f point, float radius, float size_k = 1.4f)
    {
        tracker->init(
            img,
            create_rect_from_keypoint(point));
    }

    void set_tracker_position(cv::Mat &img, cv::KeyPoint keypoint)
    {
        set_tracker_position(img, keypoint.pt, keypoint.size);
    }

    /**
     * @brief Track the next frame ahead.
     *
     * @param img the image to track with
     */
    void track(cv::Mat &img)
    {
        cv::Rect rect = get_last_bbox();
        bool isOk = tracker->update(img, rect);
        return TrackerSingleFrameInfo(isOk, current, get_last_bbox());
    }

    void append(BBox bbox)
    {
        previous_frames.push_back(bbox);
    }
}