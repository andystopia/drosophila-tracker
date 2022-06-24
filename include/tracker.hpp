#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/features2d.hpp>
#include "tracker_single_frame_info.hpp"
#include "bbox.hpp"

class Tracker
{
private:
    cv::Ptr<cv::Tracker> tracker;
    std::vector<BBox> previous_frames;

private:
    static cv::Rect create_rect_from_circle(cv::Point2f point, float radius, float radius_k = 0.80f)
    {

        return cv::Rect(point.x - radius * radius_k, point.y - radius * radius_k, radius * radius_k * 2, radius * radius_k * 2);
    }
    static cv::Rect create_rect_from_keypoint(cv::KeyPoint keypoint)
    {
        return create_rect_from_circle(keypoint.pt, keypoint.size);
    }

public:
    Tracker(cv::Mat &img, cv::KeyPoint keypoint)
    {
        cv::Rect rect = create_rect_from_keypoint(keypoint);
        // TODO: replace with DI later
        tracker = cv::TrackerCSRT::create();

        set_tracker_position(img, rect);
        previous_frames.push_back(BBox(rect));
    }

    cv::Rect set_tracker_position(cv::Mat &img, cv::Rect rect)
    {
        tracker->init(img, rect);
        return rect;
    }

    cv::Rect set_tracker_position(cv::Mat &img, cv::Point2f point, float radius, float size_k = 1.4f)
    {
        return set_tracker_position(img, create_rect_from_circle(point, radius, size_k));
    }

    cv::Rect set_tracker_position(cv::Mat &img, cv::KeyPoint keypoint)
    {
        return set_tracker_position(img, keypoint.pt, keypoint.size);
    }

    /**
     * @brief Track the next frame ahead.
     *
     * @param img the image to track with
     */
    TrackerSingleFrameInfo track(cv::Mat &img)
    {
        BBox rect = get_last_bbox();
        cv::Rect current = rect.as_rect();
        bool isOk = tracker->update(img, current);
        return TrackerSingleFrameInfo(isOk, BBox(current), get_last_bbox(), *this);
    }

    void append(BBox bbox)
    {
        previous_frames.push_back(bbox);
    }

    BBox get_last_bbox()
    {
        return previous_frames.back();
    }
};