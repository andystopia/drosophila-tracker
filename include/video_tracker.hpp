#pragma once
#include "tracker.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/features2d.hpp>

struct VideoTrackerMetaParameters { 
    
    /**
     * @brief This meta parameter should be set
     * to whatever 
     * 
     */
    int correct_when_overlap_exceeds_percent = 80;

};

class VideoTracker
{
private:
    std::vector<Tracker> trackers;
    std::vector<cv::KeyPoint> current_frame_keypoints;
    std::vector<TrackerSingleFrameInfo> current_frame_infos;
    cv::Ptr<cv::SimpleBlobDetector> blob_detector;
    VideoTrackerMetaParameters meta;

private:
    VideoTracker() {}
    /**
     * @brief Calculate the keypoints on the current frame.
     * Returns a vector to the relevant class field, for the
     * sake of being able to use this in an async context.
     *
     * @param img the image to detect in
     * @param detector the detector to detect with
     * @return std::vector<cv::KeyPoint>&
     */
    std::vector<cv::KeyPoint> &get_current_frame_keypoints(cv::Mat &img);

    /**
     * @brief Get the current tracker infos object
     * 
     * @param img the image to do the tracking inside of
     * @return std::vector<TrackerSingleFrameInfo> 
     */
    std::vector<TrackerSingleFrameInfo>& get_current_tracker_infos(cv::Mat &img);

    void fix_lost_trackers()
    /**
     * @brief Fixes the case when a tracker swaps to another nearby object,
     * while abandoning it's current tracker. This includes whenever the tracker
     * decides to fail. 
     * 
     * If there is a keypoint which is not contained within a tracker single frame info,
     * it will attempt to "rescue" a rescue a nearby tracker. 
     * 
     * Iff a tracker is "lost", it will be paired with the nearest detected blob, and the 
     * tracker should hopefully continue from there. 
     * 
     * @param tracks 
     */
    void fix_overlaps(std::vector<std::pair<cv::KeyPoint&, TrackerSingleFrameInfo&>>& tracks, std::vector<TrackerSingleFrameInfo>& trackers);

public:
    void process_frame(cv::Mat img);
};