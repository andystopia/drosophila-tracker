#include "video-preprocessor.hpp"

std::vector<cv::Mat> VideoPreprocessor::process_frame(cv::Mat &img)
{
    std::vector<cv::Rect> rois = get_regions_of_interest();

    std::vector<cv::Mat> imgs;
    imgs.reserve(rois.size());
    for (cv::Rect rect : rois)
    {
        imgs.push_back(img(rect));
    }
    return imgs;
}

QuartileVideoPreprocessor::QuartileVideoPreprocessor(cv::Mat &frame)
{
    auto upper_left = cv::Rect(0, 0, frame.cols / 2, frame.rows / 2);
    auto lower_left = cv::Rect(0, frame.rows / 2, frame.cols / 2, frame.rows / 2 - 1);
    auto lower_right = cv::Rect(frame.cols / 2, frame.rows / 2, frame.cols / 2 - 1, frame.rows / 2 - 1);
    auto upper_right = cv::Rect(frame.cols / 2, 0, frame.cols / 2 - 1, frame.rows / 2 - 1);

    rois = {upper_left, lower_left, lower_right, upper_right};
    quartile_names = {
        "upper-left",
        "lower-left",
        "lower-right",
        "upper-right",
    };
}

// std::vector<cv::Mat> QuartileVideoPreprocessor::process_frame(cv::Mat &img) { 
//     std::vector<cv::Mat> areas;

//     std::vector& rois
// }
std::vector<cv::Rect> QuartileVideoPreprocessor::get_regions_of_interest()
{
    return rois;
}