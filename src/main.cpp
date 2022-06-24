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

void frame_reader(cv::VideoCapture capture, msd::channel<cv::Mat> &frame_queue)
{
    cv::Mat image;
    while (capture.read(image) && !frame_queue.closed())
    {
        cv::Mat after;
        cv::threshold(image, after, 70, 255, cv::ThresholdTypes::THRESH_BINARY);
        std::move(after) >> frame_queue;
    }
    frame_queue.close();
}

int main(int argc, char *argv[]){

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

    std::vector<std::pair<cv::Rect, cv::Ptr<cv::Tracker>>> tracks;

    cv::VideoCapture vid(path);

    // Exit if video is not opened
    if (!vid.isOpened())
    {
        std::cerr << "Could not read video file" << std::endl;
        return 1;
    }

    // we don't want to fill up our memory too much, so
    // we'll allow 30 frames to queue up
    msd::channel<cv::Mat> frame_queue{30};

    std::thread reader(frame_reader, vid, std::ref(frame_queue));

    // Read first frame
    cv::Mat frame;

    frame << frame_queue;

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


    auto rect = cv::Rect(0, 0, frame.cols / 2 , frame.rows / 2 );
    cv::Mat fcp = frame.operator()(rect).clone();
    VideoTracker vidtrack = VideoTracker(fcp, blob_detector);

    imshow("Tracking", frame);

    // cv::waitKey(0);

    int frame_count = 0;
    // while (vid.read(frame))

    std::vector<std::vector<cv::Point2f>> tracked_points;
    while (!frame_queue.closed())
    {

        frame << frame_queue;
        
        vidtrack.process_frame(frame);
        vidtrack.view_current_frame(frame, "modern");

        int k = cv::waitKey(1);
        if (k == 27)
        {
            break;
        }
        std::cerr << "frame: " << ++frame_count << std::endl;
    }
    reader.join();
}
