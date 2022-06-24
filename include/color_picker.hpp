#pragma once

#include <opencv2/imgproc.hpp>

class ColorPicker
{
private:
    // colors thanks to: https://stackoverflow.com/questions/1168260/algorithm-for-generating-unique-colors
    static const cv::Scalar colors[64];

public:
    static cv::Scalar get_color(size_t color)
    {
        size_t chosen = color % 64;
        return colors[chosen];
    }
};

