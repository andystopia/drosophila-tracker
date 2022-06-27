#pragma once
#include "tracker.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/features2d.hpp>

struct VideoTrackerMetaParameters
{

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
public:
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
    std::vector<cv::KeyPoint> &detect_current_frame_keypoints(cv::Mat &img);

    /**
     * @brief Get the current tracker infos object
     *
     * @param img the image to do the tracking inside of
     * @return std::vector<TrackerSingleFrameInfo>
     */
    std::vector<TrackerSingleFrameInfo> &detect_current_tracker_positions(cv::Mat &img);

    /**
     * @brief Initialize a tracking set using a set of keypoints
     *
     * @param keypoints the set of keypoints to initialize with
     * @return std::vector<Tracker>
     */
    std::vector<Tracker> initialize_trackers(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints);

    /**
     * @brief Fixes lost trackers
     *
     * Detects and attempts to correct for lost trackers by filtering through points and
     * blobs for blobs which are likely candidates for fixing lost trackers.
     *
     * @param tracks the tracks to look through
     * @param points the blobs to detect with
     */
    void fix_lost_trackers(cv::Mat &img, std::vector<TrackerSingleFrameInfo> &tracks, std::vector<cv::KeyPoint> &points);

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
    void fix_overlaps(cv::Mat& img);
    /**
     * @brief writes the frame's data back to the 
     * trackers so that they can maintain a historical log.
     * 
     */
    void write_frame_data();
    
    /**
     * @brief If two trackers are close to each other,
     * make sure that the nearest thing that they track 
     * minimizes their inertias.
     * 
     * @param img 
     */
    void minimize_nearby_swap_inertias(cv::Mat& img);
    /**
     * @brief For any tracker whose nearest tracker
     * is futher than epsilon away, attempt to recenter
     * that tracker upon it's nearest tracking point, if 
     * the tracking point is within delta.
     * 
     * @param img 
     */
    void use_local_recentering(cv::Mat& img);
public:
    /**
     * @brief Construct a new Video Tracker with a first frame
     *
     * @param first_frame the first frame of the video to track.
     */
    VideoTracker(cv::Mat &first_frame, cv::Ptr<cv::SimpleBlobDetector> detector);
    void process_frame(cv::Mat& img);
    
    void write_csv(std::string filename);
    void view_current_frame(cv::Mat& img, const std::string& window_name);
};