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

cv::Rect create_rect_from_keypoint(cv::KeyPoint keypoint);

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

class BoundingBox
{
private:
    cv::Rect rect;

public:
    BoundingBox(cv::Rect rect) : rect(rect)
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
    float get_overlap_score(BoundingBox other) const
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

    float distance_squared(BoundingBox other)
    {
        return GeometryUtils::distance_squared(center(), other.center());
    }
};

class CountingKeyPoint;
class BoundingBoxMotion
{
private:
    BoundingBox start;
    BoundingBox end;
    cv::Ptr<cv::Tracker> tracker;
    bool is_valid_track;
    std::optional<std::reference_wrapper<CountingKeyPoint>> nearest_key_point;

public:
    BoundingBoxMotion(
        BoundingBox start,
        BoundingBox end,
        cv::Ptr<cv::Tracker> tracker,
        std::optional<std::reference_wrapper<CountingKeyPoint>> nearest_key_point,
        bool is_valid = true) : start(start), end(end), tracker(tracker), is_valid_track(is_valid), nearest_key_point(nearest_key_point)
    {
    }

    bool is_valid() const
    {
        return is_valid_track;
    }

    BoundingBox get_startpoint() const
    {
        return start;
    }

    BoundingBox get_endpoint() const
    {
        return end;
    }
    cv::Ptr<cv::Tracker> get_tracker() const
    {
        return tracker;
    }

    void reseat_tracker(cv::Mat &image, cv::KeyPoint point)
    {
        cv::Rect e = create_rect_from_keypoint(point);
        end = BoundingBox(e);
        tracker->init(image, e);
    }
    float inertia() const
    {
        return GeometryUtils::mag_squared(end.center() - start.center());
    }

    friend std::ostream &operator<<(std::ostream &out, BoundingBoxMotion const &motion)
    {
        float start_x = motion.start.center().x;
        float start_y = motion.start.center().y;
        float end_x = motion.end.center().x;
        float end_y = motion.end.center().y;

        out << "[" << start_x << ", " << start_y << "] -> [" << end_x << ", " << end_y << "] : " << motion.inertia();
        return out;
    }

    float distance_squared_between_endpoints(BoundingBoxMotion other) const
    {
        return GeometryUtils::mag_squared((end.center() - other.end.center()));
    }

    float get_endpoint_overlap_score(BoundingBoxMotion other) const
    {
        return end.get_overlap_score(other.end);
    }
};

float dist_squared(cv::Point2f a, cv::Point2f b)
{
    cv::Point2f delta = b - a;
    return delta.x * delta.x + delta.y * delta.y;
}

/**
 * @brief Calculates the pairs in a vector which are closest together.
 * Note an element will not pair with itself (not reference wise at least).
 *
 * @tparam T the element to match with
 * @param collection the collection to run this operation on
 * @param comparer a function which should return true iff the second argument is smaller than the third. The function will be given the element which will be the first in the pair, and should return true if the second argument is closer than the third, else it should
 return false.
 * @return std::vector<std::pair<T&, T&>> a vector filled with pairs of nearby elements.
 */
template <typename T>
std::vector<std::pair<T &, T &>> pair_with_nearest(
    std::vector<T> &collection,
    const std::function<bool(const T &, const T &, const T &)> &comparer)
{
    if (collection.size() < 2)
    {
        return std::vector<std::pair<T &, T &>>();
    }

    std::vector<std::pair<T &, T &>> pairs;
    pairs.reserve(collection.size());

    for (size_t i = 0; i < collection.size(); i++)
    {
        size_t smallest = i == 0 ? 1 : 0;
        for (size_t j = 0; j < collection.size(); j++)
        {
            if (i != j)
            {
                if (comparer(collection.at(i), collection.at(j), collection.at(smallest)))
                {
                    smallest = j;
                }
            }
        }
        T &first = collection.at(i);
        T &second = collection.at(smallest);
        pairs.push_back({first, second});
    }
    return pairs;
}

cv::Rect create_rect_from_keypoint(cv::KeyPoint keypoint)
{
    cv::Point2f point = keypoint.pt;

    float size = keypoint.size;
    return cv::Rect(point.x - size, point.y - size, size * 1.75, size * 1.75);
}

class CountingKeyPoint
{
private:
    cv::KeyPoint keypoint;
    int count = 0;

public:
    CountingKeyPoint(cv::KeyPoint point) : keypoint(point)
    {
    }

    void inc()
    {
        count++;
    }

    int get_count() const
    {
        return count;
    }

    cv::KeyPoint get_keypoint() const
    {
        return keypoint;
    }
};

class KeyPointCollection
{
private:
    std::vector<CountingKeyPoint> keypoints;

public:
    KeyPointCollection()
    {
    }

    void clear()
    {
        keypoints.clear();
    }

    void reset_with_keypoints(std::vector<cv::KeyPoint> &keys)
    {
        this->clear();
        std::transform(
            keys.cbegin(),
            keys.cend(),
            std::back_inserter(keypoints),
            [](cv::KeyPoint const &point)
            {
                return CountingKeyPoint(point);
            });
    }

    CountingKeyPoint const &at(size_t i)
    {
        return keypoints.at(i);
    }

    /**
     * @brief Get the index of nearest keypoint to the passed point,
     * or -1 if it doesn't exist.
     *
     * @param point the point to compare with
     * @return std::ssize_t the index of the get point
     */
    std::optional<std::reference_wrapper<CountingKeyPoint>> get_closest_point(cv::Point2f point)
    {
        if (keypoints.size() == 0)
        {
            return std::nullopt;
        }

        std::size_t index = 0;
        for (size_t i = 1; i < keypoints.size(); i++)
        {
            if (index == -1)
            {
                index = i;
            }
            else if (
                GeometryUtils::distance_squared(keypoints[i].get_keypoint().pt, point) < GeometryUtils::distance_squared(keypoints[index].get_keypoint().pt, point))
            {
                index = i;
            }
        }

        return std::optional(std::reference_wrapper(keypoints[index]));
    }

    std::vector<CountingKeyPoint> &get_keypoints()
    {
        return keypoints;
    }

    std::vector<std::reference_wrapper<CountingKeyPoint>> get_points_ordered_by_distance_to(cv::Point2f point)
    {
        std::vector<std::reference_wrapper<CountingKeyPoint>> sorted;

        std::transform(keypoints.begin(), keypoints.end(), std::back_inserter(sorted), [](CountingKeyPoint &point)
                       { return std::reference_wrapper(point); });

        std::sort(sorted.begin(), sorted.end(), [point](auto const &first, auto const &second)
                  { return GeometryUtils::distance_squared(first.get().get_keypoint().pt, point) < GeometryUtils::distance_squared(second.get().get_keypoint().pt, point); });
        return sorted;
    }

    std::vector<std::reference_wrapper<CountingKeyPoint>> get_points_with_count(int count)
    {
        std::vector<std::reference_wrapper<CountingKeyPoint>> counted;

        std::copy_if(keypoints.begin(), keypoints.end(), std::back_inserter(counted), [count](auto &keypoint)
                     { return keypoint.get_count() == count; });
        return counted;
    }
};



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

std::pair<bool, cv::Rect> execute_track(cv::Ptr<cv::Tracker> &tracker, cv::Mat const &img, cv::Rect bbox)
{
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    auto keypoint_detection_i = high_resolution_clock::now();
    cv::Rect box = bbox;
    bool is_ok = tracker->update(img, box);
    auto keypoint_detection_f = high_resolution_clock::now();

    duration<double, std::milli> keypoint_time = keypoint_detection_f - keypoint_detection_i;


    return {is_ok, box};
}

int main(int argc, char *argv[])
{
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    std::string path;
    if (argc == 2)
    {
        std::cerr << "Attempting to read video file at: `" << argv[1] << "`" << std::endl;
        path = std::string(argv[1]);
    }
    else
    {
        path = "60-fps-trail.mp4";
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
    // bool ok = vid.read(first_frame);

    // Define initial bounding box
    // cv::Rect2d bbox(287, 23, 86, 320);

    // Uncomment the line below to select a different bounding box
    // cv::Rect2d bbox = cv::selectROI(frame, true, true);
    std::vector<cv::Rect2i> rects;

    // cv::selectROIs("Select ROIs", frame, rects, false, true);

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

    std::vector<cv::KeyPoint> keypoints;

    blob_detector->detect(frame, keypoints);
    // std::cerr << keypoints.size() << std::endl;

    for (auto keypoint : keypoints)
    {
        rects.push_back(create_rect_from_keypoint(keypoint));
    }

    // Display bounding box.
    // cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2, 1);

    for (cv::Rect2i &rect : rects)
    {
        cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();
        tracker->init(frame, rect);
        tracks.push_back(std::make_pair(rect, tracker));
    }

    imshow("Tracking", frame);

    // cv::waitKey(0);
    KeyPointCollection keypoint_collection;

    int frame_count = 0;
    // while (vid.read(frame))

    std::vector<std::vector<cv::Point2f>> tracked_points;
    while (!frame_queue.closed())
    {
        // std::cerr << "Frame Queue Size: " << frame_queue.size() << std::endl;
        std::vector<BoundingBoxMotion> motions;

        frame << frame_queue;
        // cv::Mat after;
        // frame = cv::threshold(frame, after, 70, 255, cv::ThresholdTypes::THRESH_BINARY);

        // frame = after;
        // Start timer
        // double timer = (double)cv::getTickCount();

        // auto keypoint_detection_i = high_resolution_clock::now();
        keypoints.clear();
        blob_detector->detect(frame, keypoints);
        // auto keypoint_detection_f = high_resolution_clock::now();

        // duration<double, std::milli> keypoint_time = keypoint_detection_f - keypoint_detection_i;

        // std::cerr << "KeyPoint Detection: " << keypoint_time.count() << "ms\n";

        keypoint_collection.reset_with_keypoints(keypoints);

        for (cv::KeyPoint &kp : keypoints)
        {
            cv::circle(frame, kp.pt, kp.size, cv::Scalar(0, 255, 0));
        }

        auto tracker_i = high_resolution_clock::now();

        std::vector<cv::Rect> startings;

        std::transform(tracks.cbegin(), tracks.cend(), std::back_inserter(startings), [](const auto &track)
                       { 
            auto &[bbox, tracker] = track;
            return bbox; });

        std::vector<std::future<std::pair<bool, cv::Rect>>> ending_tasks;

        std::transform(
            tracks.begin(),
            tracks.end(),
            std::back_inserter(ending_tasks),
            [frame](
                std::pair<cv::Rect, cv::Ptr<cv::Tracker>> &track)
            {
                cv::Ptr<cv::Tracker> tracker = track.second;
                cv::Mat const &img = frame;
                cv::Rect bbox = track.first;
                return std::async(std::launch::async, [tracker, img, bbox]() mutable
                                  { return execute_track(tracker, img, bbox); });
            });

        std::vector<std::pair<bool, cv::Rect>> endings;

        std::transform(ending_tasks.begin(), ending_tasks.end(), std::back_inserter(endings), [](auto &val)
                       { return val.get(); });

        // std::transform(tracks.begin(), tracks.end(), std::back_inserter(endings), [frame](std::pair<cv::Rect, cv::Ptr<cv::Tracker>> &track) mutable
        //                {
        //     auto &[bbox, tracker] = track;

        //     cv::Rect start = bbox;

        //     // Update the tracking result
        //     bool ok = tracker->update(frame, bbox);
        //     return std::make_pair(ok, bbox); });

        for (int i = 0; i < tracks.size(); i++)
        {
            for (auto keypoint : keypoint_collection.get_keypoints()) { 
                keypoint.inc();
            }
            motions.push_back(BoundingBoxMotion(
                startings[i],
                endings[i].second,
                tracks[i].second,
                keypoint_collection.get_closest_point(BoundingBox(endings[i].second).center()),
                endings[i].first));
        }

        // std::transform(tracks.begin(), tracks.end(), std::back_inserter(motions), [keypoint_collection, frame](std::pair<cv::Rect, cv::Ptr<cv::Tracker>>& track) mutable {
        //     auto &[bbox, tracker] = track;

        //     cv::Rect start = bbox;

        //     // Update the tracking result
        //     bool ok = tracker->update(frame, bbox);

        //     cv::Rect end = bbox;

        //     // Calculate Frames per second (FPS)
        //     // float fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);

        //     if (ok)
        //     {
        //         for (CountingKeyPoint &point : keypoint_collection.get_keypoints())
        //         {
        //             if (end.contains(point.get_keypoint().pt))
        //             {
        //                 point.inc();
        //             }
        //         }
        //     }

        //     return BoundingBoxMotion(
        //         start,
        //         end,
        //         tracker,
        //         keypoint_collection.get_closest_point(BoundingBox(end).center()),
        //         ok
        //     );
        // });
        // for (auto &[bbox, tracker] : tracks)
        // {
        //     cv::Rect start = bbox;

        //     // Update the tracking result
        //     bool ok = tracker->update(frame, bbox);

        //     cv::Rect end = bbox;

        //     // Calculate Frames per second (FPS)
        //     // float fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);

        //     if (ok)
        //     {
        //         for (CountingKeyPoint &point : keypoint_collection.get_keypoints())
        //         {
        //             if (end.contains(point.get_keypoint().pt))
        //             {
        //                 point.inc();
        //             }
        //         }
        //     }

        //     motions.push_back(
        //         BoundingBoxMotion(start, end, tracker, keypoint_collection.get_closest_point(BoundingBox(end).center()), ok));
        // }

        auto tracker_f = high_resolution_clock::now();

        duration<double, std::milli> tracker_time = tracker_f - tracker_i;

        std::cerr << "Tracker Detection Time: " << tracker_time.count() << "ms\n";

        // optimally this would be done with the kuhn munkres (hungarian)
        // selection algorithm to properly minimize error
        for (size_t i = 0; i < motions.size(); i++)
        {
            BoundingBoxMotion &motion = motions[i];
            if (!motion.is_valid())
            {
                std::vector<std::reference_wrapper<CountingKeyPoint>> nearest = keypoint_collection.get_points_ordered_by_distance_to(motion.get_endpoint().center());

                for (std::reference_wrapper<CountingKeyPoint> p : nearest)
                {
                    CountingKeyPoint &point = p.get();
                    if (point.get_count() == 0)
                    {
                        point.inc();
                        cv::Rect correct_endpoint = create_rect_from_keypoint(point.get_keypoint());
                        motion.get_tracker()->init(frame, correct_endpoint);
                        BoundingBoxMotion new_motion(motion.get_startpoint(), correct_endpoint, motion.get_tracker(), p);
                        motions.at(i) = new_motion;
                    }
                }
            }
        }

        for (auto &&delta : motions)
        {
            rectangle(frame, delta.get_endpoint().as_rect(), cv::Scalar(255, 0, 0), 2, 1);
        }

        std::vector<std::pair<BoundingBoxMotion &, BoundingBoxMotion &>> motion_pairs = pair_with_nearest<BoundingBoxMotion>(
            motions,
            [](BoundingBoxMotion const &src, BoundingBoxMotion const &possible_a, BoundingBoxMotion const &possible_b) -> bool
            {
                return src.distance_squared_between_endpoints(possible_a) < src.distance_squared_between_endpoints(possible_b);
            });

        for (auto &&[motion, closest] : motion_pairs)
        {
            int pct_overlap = (int)(motion.get_endpoint_overlap_score(closest) * 100);

            // regions don't generally overlap by more than 80%
            if (pct_overlap >= 80)
            {
                if (motion.inertia() >= closest.inertia())
                {
                    std::vector<std::reference_wrapper<CountingKeyPoint>> ordered_points = keypoint_collection.get_points_ordered_by_distance_to(motion.get_endpoint().center());

                    for (std::reference_wrapper<CountingKeyPoint> &p : ordered_points)
                    {
                        CountingKeyPoint &point = p.get();
                        if (point.get_count() == 0)
                        {
                            // std::cerr << "Updated" << std::endl;
                            if (GeometryUtils::distance_squared(point.get_keypoint().pt, motion.get_endpoint().center()) < 100)
                            {
                                point.inc();
                                motion.reseat_tracker(frame, point.get_keypoint());
                            }
                        }
                    }
                }
            }
            // std::cout << motion << " $ " << motion.distance_squared_between_endpoints(closest) << " $ " << pct_overlap << "%" << std::endl;
        }

        for (CountingKeyPoint &point : keypoint_collection.get_keypoints())
        {
            if (point.get_count() == 0)
            {
                std::vector<std::reference_wrapper<BoundingBoxMotion>> ordered_by_closeness;

                std::transform(motions.begin(), motions.end(), std::back_inserter(ordered_by_closeness), [](auto &val)
                               { return std::reference_wrapper(val); });

                auto p = point;
                std::sort(
                    ordered_by_closeness.begin(),
                    ordered_by_closeness.end(),
                    [p](const auto &a, const auto &b)
                    {
                        return GeometryUtils::distance_squared(a.get().get_endpoint().center(), p.get_keypoint().pt) < GeometryUtils::distance_squared(b.get().get_endpoint().center(), p.get_keypoint().pt);
                    });

                CountingKeyPoint &closest_point = keypoint_collection.get_points_ordered_by_distance_to(point.get_keypoint().pt).at(1);
                // FIX ME: INSTEAD OF USING THE FIRST POINT, USE THE FIRST SHARED POINT
                // ordered_by_closeness.at(0).get().get_tracker()->init(frame, create_rect_from_keypoint(point.get_keypoint()));
                ordered_by_closeness.at(0).get().reseat_tracker(frame, point.get_keypoint());
            }
        }

        // if (failed.size() == 1 && keypoints.size() == 1)
        // {
        //     auto [rect, tracker] = failed[0];
        //     auto kp = keypoints[0].pt;
        //     rect = cv::Rect(kp.x - rect.width / 2, kp.y - rect.height / 2, rect.width, rect.height);
        //     tracker->init(frame, rect);
        //     rectangle(frame, rect, cv::Scalar(255, 0, 255), 2, 1);
        // }
        // else
        // {
        //     // for (cv::KeyPoint &keypoint : keypoints)
        //     // {
        //     //     cv::Point2f point = keypoint.pt;
        //     //     float size = keypoint.size;
        //     //     cv::Rect rect = cv::Rect(point.x - size, point.y - size, size * 2, size * 2);
        //     //     cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();
        //     //     tracker->init(frame, rect);
        //     //     tracks.push_back(std::make_pair(rect, tracker));
        //     // }
        // }

        // Display tracker type on frame
        putText(frame, "CRST Tracker", cv::Point(70, 20), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(50, 170, 50), 2);

        // Display FPS on frame
        // putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

        // Display frame.
        imshow("Tracking", frame);

        std::vector<cv::Point2f> this_frame_tracked_points;

        std::transform(motions.cbegin(), motions.cend(), std::back_inserter(this_frame_tracked_points), [](const BoundingBoxMotion &motion)
                       { return motion.get_endpoint().center(); });

        tracked_points.push_back(this_frame_tracked_points);
        // Exit if ESC pressed.
        int k = cv::waitKey(1);
        if (k == 27)
        {
            break;
        }
        // https://www.reddit.com/r/C_Programming/comments/502xun/how_do_i_clear_a_line_on_console_in_c/
        fprintf(stderr, "\x1b[1F"); // Move to beginning of previous line
        fprintf(stderr, "\x1b[2K"); // Clear entire line
        std::cerr << "frame: " << ++frame_count << std::endl;
    }

    for (size_t i = 0; i < tracked_points.size(); i++)
    {
        std::vector<cv::Point2f> &row = tracked_points.at(i);

        std::cout << i;
        for (size_t j = 0; j < row.size(); j++)
        {
            std::cout << "," << row.at(j).x << "," << row.at(j).y;
        }

        std::cout << std::endl;
    }
    // frame_queue.close();
    // reader.join();
}
