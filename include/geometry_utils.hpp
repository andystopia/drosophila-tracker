#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/features2d.hpp>

class GeometryUtils
{
public:
    static float mag_squared(cv::Point2f point)
    {
        return point.x * point.x + point.y * point.y;
    }
    static float distance_squared(cv::Point2f a, cv::Point2f b)
    {
        cv::Point2f d = a - b;
        return d.x * d.x + d.y * d.y;
    }

    static float distance(cv::Point2f a, cv::Point2f b)
    {
        return std::sqrt(distance_squared(a, b));
    }
};