#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/features2d.hpp>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <optional>
#include <functional>
#include <thread>
#include "msd/channel.hpp"
#include <chrono>
#include <future>
#include "geometry_utils.hpp"
#include "pairings.hpp"
#include "video_tracker.hpp"
#include "color_picker.hpp"
#include "video-preprocessor.hpp"

struct FrameInfo
{
    std::vector<cv::Mat> tracking_regions;
    size_t frame_number;
};

void frame_reader(cv::VideoCapture capture, std::unique_ptr<VideoPreprocessor> proc, msd::channel<FrameInfo> &frame_queue)
{
    cv::Mat image;
    uint frame_number = 1;
    while (capture.read(image) && !frame_queue.closed())
    {
        std::vector<cv::Mat> parts = proc->process_frame(image);
        FrameInfo info = FrameInfo{
            .tracking_regions = parts,
            .frame_number = frame_number,
        };

        // cv::Mat after;
        // cv::threshold(image, after, 70, 255, cv::ThresholdTypes::THRESH_BINARY);
        info >> frame_queue;
        frame_number++;
    }
    frame_queue.close();
}
int main(int argc, char *argv[])
{
    std::string path;
    if (argc == 2)
    {
        std::cerr << "Attempting to read video file at: `" << argv[1] << "`" << std::endl;
        path = std::string(argv[1]);
    }
    else
    {
        path = "../60-fps-2.mp4";
    }

    // std::vector<std::pair<cv::Rect, cv::Ptr<cv::Tracker>>> tracks;

    cv::VideoCapture vid(path);

    // Exit if video is not opened
    if (!vid.isOpened())
    {
        std::cerr << "Could not read video file" << std::endl;
        return 1;
    }

    // we don't want to fill up our memory too much, so
    // we'll allow 30 frames to queue up
    msd::channel<FrameInfo> frame_queue{30};

    cv::Mat first_frame;

    vid.read(first_frame);
    VideoSplitAndDifference::write_json_rois(first_frame, path);
    cv::waitKey(0);

    std::unique_ptr<PreProcessorChain> preprocessor = std::make_unique<PreProcessorChain>();

    preprocessor
        ->chain(std::move(std::make_unique<VideoSplitAndDifference>(path + ".json")))
        // ->chain(std::move(std::make_unique<QuartileVideoPreprocessor>(first_frame)))
        .chain(std::move(std::make_unique<CvThresholdPreprocessor>(70, 255, cv::ThresholdTypes::THRESH_BINARY)));

    cv::SimpleBlobDetector::Params params;
    params.minDistBetweenBlobs = 8.0f;
    params.filterByInertia = false;
    params.filterByConvexity = false;
    params.filterByColor = false;
    params.filterByCircularity = false;
    params.filterByArea = true;
    params.minArea = 25.0f;
    params.maxArea = 5000.0f;
    params.minRepeatability = 2;
    params.thresholdStep = 10.0;
    params.minThreshold = 5.0;

    cv::Ptr<cv::SimpleBlobDetector>
        blob_detector = cv::SimpleBlobDetector::create(params);

    std::vector<cv::Mat> parts = preprocessor->process_frame(first_frame);
    std::vector<VideoTracker> trackers;
    for (size_t i = 0; i < parts.size(); i++)
    {
        trackers.push_back(VideoTracker(parts.at(i), blob_detector));
    }

    std::thread reader(frame_reader, vid, std::move(preprocessor), std::ref(frame_queue));

    // Read first frame

    FrameInfo frame;
    frame << frame_queue;

    // auto upper_left = cv::Rect(0, 0, frame.cols / 2, frame.rows / 2);
    // auto lower_left = cv::Rect(0, frame.rows / 2, frame.cols / 2, frame.rows / 2 - 1);
    // auto lower_right = cv::Rect(frame.cols / 2, frame.rows / 2, frame.cols / 2 - 1, frame.rows / 2 - 1);
    // // auto upper_right = cv::Rect(frame.cols / 2, 0, frame.cols / 2 - 1, frame.rows / 2 - 1);

    // std::vector<cv::Rect> tracking_squares = {lower_right};

    // std::vector<TrackingRegion> regions;

    // QuartileVideoPreprocessor preprocessor = QuartileVideoPreprocessor(frame);

    // cv::waitKey(0);

    // while (vid.read(frame))

    // std::vector<std::vector<cv::Point2f>> tracked_points;

    std::vector<std::future<void>> futures;

    std::cout << trackers.size() << std::endl;
    while (!frame_queue.closed())
    {
        frame << frame_queue;

        // std::cout << frame.tracking_regions.size() << std::endl;

        futures.clear();
        for (size_t i = 0; i < frame.tracking_regions.size(); i++)
        {
            futures.push_back(std::async(std::launch::async, [&, i](){ 
                                             trackers.at(i).process_frame(frame.tracking_regions.at(i)); 
                                             }));
        }


        for (size_t i = 0; i < futures.size(); i++)
        {
            futures.at(i).get();
            trackers.at(i).view_current_frame(frame.tracking_regions.at(i), std::to_string(i));
        }
        std::cerr << "frame: " << frame.frame_number << std::endl;
        int k = cv::waitKey(1);

        if (k == 27)
        {
            frame_queue.close();
            break;
        }
    }

    for (size_t i = 0; i < trackers.size(); i++) { 
        std::string filename = "output-" + std::to_string(i) + ".csv";

        trackers.at(i).write_csv(filename);
    }
    reader.join();
}
