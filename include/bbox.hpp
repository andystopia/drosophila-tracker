#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/features2d.hpp>

#include "geometry_utils.hpp"


class BBox
{
private:
    cv::Rect rect;

public:
    BBox(cv::Rect rect) : rect(rect)
    {
    }

    cv::Rect as_rect() const
    {
        return rect;
    }
    cv::Point2f center() const
    {
        return cv::Point2f(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0);
    }

    int left() const
    {
        return rect.x;
    }

    int right() const
    {
        return rect.x + rect.width;
    }

    int top() const
    {
        return rect.y;
    }

    int bottom() const
    {
        return rect.y + rect.height;
    }

    float area() const
    {
        return rect.area();
    }

    bool contains(cv::Point2f point) const
    {
        return rect.contains(point);
    }

    /**
     * @brief Computes what % of the area of the smaller
     * rectangle is within the larger rectange, that way
     * we can set a threshold for "instability" of a track.
     *
     * @param other the other thing to compare with
     * @return float [0, 1] depending on how much the boxes overlap
     */
    float get_overlap_score(BBox other) const
    {
        // from https://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
        float x_overlap = std::max(
            0.0f,
            std::min((float)right(), (float)other.right()) - std::max((float)left(), (float)other.left()));
        float y_overlap = std::max(
            0.0f,
            std::min((float)bottom(), (float)other.bottom()) - std::max((float)top(), (float)other.top()));
        float overlap_area = x_overlap * y_overlap;
        float min_area = std::min((float)area(), (float)other.area());
        float proportion = overlap_area / min_area;
        // std::cout << "{" << overlap_area << "," << min_area << "}" << std::endl;
        return proportion;
    }

    float distance_squared(BBox other)
    {
        return GeometryUtils::distance_squared(center(), other.center());
    }
};