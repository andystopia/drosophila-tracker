#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <fstream>
#include "json.hpp"

class VideoPreprocessor
{
public:
    virtual void init(cv::Mat &img) {}
    virtual std::vector<cv::Rect> get_regions_of_interest()
    {
        return std::vector<cv::Rect>();
    }
    virtual std::vector<cv::Mat> process_frame(cv::Mat &img);
    virtual ~VideoPreprocessor() {}
};

using json = nlohmann::json;

struct VideoDifferenceROI
{
    int x;
    int y;
    int width;
    int height;
    std::string image_path;

    cv::Rect as_rect()
    {
        return cv::Rect(x, y, width, height);
    }
};

static void to_json(json &j, VideoDifferenceROI const &roi)
{
    j = json{{"x", roi.x}, {"y", roi.y}, {"width", roi.width}, {"height", roi.height}, {"image", roi.image_path}};
}

static void from_json(const json &j, VideoDifferenceROI &roi)
{
    try
    {
        j.at("x").get_to(roi.x);
        j.at("y").get_to(roi.y);
        j.at("width").get_to(roi.width);
        j.at("height").get_to(roi.height);
        j.at("image").get_to(roi.image_path);
    }
    catch (std::exception e)
    {
        std::cout << e.what() << std::endl;
    }
}

class VideoSplitAndDifference : public VideoPreprocessor
{
private:
    std::vector<std::pair<cv::Rect, cv::Mat>> maps;

public:
    static void write_json_rois(cv::Mat img, std::string dataset_name)
    {
        std::ofstream out_stream;

        std::vector<cv::Rect> rois;
        cv::selectROIs("Choose Regions Of Interest", img, rois, false, false);

        std::vector<VideoDifferenceROI> real;
        for (size_t i = 0; i < rois.size(); i++)
        {
            std::string image_name = "image-" + std::to_string(i) + ".jpg";
            real.push_back(VideoDifferenceROI{
                .x = rois[i].x,
                .y = rois[i].y,
                .width = rois[i].width,
                .height = rois[i].height,
                .image_path = image_name,
            });

            cv::imwrite(image_name, img(rois[i]));
        }

        json areas = real;
        out_stream.open(dataset_name + ".json");
        out_stream << areas;
    }
    VideoSplitAndDifference(std::string filepath)
    {
        std::ifstream input;
        input.open(filepath);

        std::stringstream buffer;
        buffer << input.rdbuf();

        std::string data = buffer.str();

        std::cout << data << std::endl;
        try
        {
            std::cerr << "attempting to read data" << std::endl;
            json json_rois = json::parse(data);

            std::vector<VideoDifferenceROI> rois;
            for (size_t i = 0; i < 4; i++)
            {
                rois.push_back(json_rois[i].get<VideoDifferenceROI>());
            }

            std::cerr << "worked" << std::endl;
            // std::vector<VideoDifferenceROI> rois = json_rois.get<std::vector<VideoDifferenceROI>>();

            std::cerr << "parsed data" << std::endl;
            for (VideoDifferenceROI &roi : rois)
            {
                maps.push_back({roi.as_rect(), cv::imread(roi.image_path)});
            }
        }
        catch (std::exception &e)
        {
            std::cout << e.what() << std::endl;
            exit(3);
        }
    }

    std::vector<cv::Mat> process_frame(cv::Mat &img) override
    {
        std::vector<cv::Mat> imgs;
        for (std::pair<cv::Rect, cv::Mat> pair : maps)
        {
            imgs.push_back(pair.second - img(pair.first));
        }
        return imgs;
    }
};

class CvThresholdPreprocessor : public VideoPreprocessor
{
private:
    double thresh;
    double maxval;
    int type;

public:
    CvThresholdPreprocessor(double thresh, double maxval, cv::ThresholdTypes type) : thresh(thresh), maxval(maxval), type(type)
    {
    }
    virtual std::vector<cv::Rect> get_regions_of_interest()
    {
        return std::vector<cv::Rect>();
    }

    virtual std::vector<cv::Mat> process_frame(cv::Mat &img)
    {
        cv::Mat post;
        cv::threshold(img, post, thresh, maxval, type);
        std::vector<cv::Mat> imgs = {post};
        return imgs;
    }
};

class QuartileVideoPreprocessor : public VideoPreprocessor
{
private:
    std::vector<cv::Rect> rois;
    std::vector<std::string> quartile_names;

public:
    QuartileVideoPreprocessor(cv::Mat &img);
    virtual std::vector<cv::Rect> get_regions_of_interest() override;
    std::vector<std::string> &get_quartile_names()
    {
        return quartile_names;
    }
};

/**
 * @brief Chains together preprocessors,
 * since preprocessors occasionally return a
 * sequence of regions instead of just a single
 * frame, every successive processor will happen
 * on *all* those subregions.
 *
 * An empty chain is an identity transform
 */
class PreProcessorChain : public VideoPreprocessor
{
private:
    std::vector<std::unique_ptr<VideoPreprocessor>> processors;

public:
    virtual std::vector<cv::Rect> get_regions_of_interest() override
    {
        return std::vector<cv::Rect>();
    }

    virtual PreProcessorChain &chain(std::unique_ptr<VideoPreprocessor> proc)
    {
        processors.push_back(std::move(proc));
        return *this;
    }

    virtual std::vector<cv::Mat> process_frame(cv::Mat &img) override
    {
        std::vector<cv::Mat> previous_buffer = {std::move(img)};
        std::vector<cv::Mat> frame_buffer;

        for (std::unique_ptr<VideoPreprocessor> &processor : processors)
        {
            for (cv::Mat &img : previous_buffer)
            {
                std::vector<cv::Mat> mats = processor->process_frame(img);
                // std::cout << mats.size() << std::endl;
                frame_buffer.insert(frame_buffer.end(), mats.begin(), mats.end());
            }

            previous_buffer.swap(frame_buffer);
            frame_buffer.clear();
        }
        return previous_buffer;
    }
};