#ifndef OPTICALFLOWTRACKING_TRACKER_H
#define OPTICALFLOWTRACKING_TRACKER_H

#include <opencv2/core/mat.hpp>

class Tracker {
public:
    virtual void init(cv::Mat frame, cv::Rect2d pedestrian) = 0;

    virtual cv::Rect2d update(cv::Mat frame) = 0;

    virtual ~Tracker() = default;

    virtual void reinit(cv::Rect2d boundingBox) = 0;

    Tracker();

};

#endif //OPTICALFLOWTRACKING_TRACKER_H
