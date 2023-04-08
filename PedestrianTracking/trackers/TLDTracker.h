#ifndef TRACKING_TLDTRACKER_H
#define TRACKING_TLDTRACKER_H

#include <opencv2/tracking.hpp>
#include "Tracker.h"
#include <opencv2/tracking/tracking_legacy.hpp>

class TLDTracker : public Tracker {

public:
    TLDTracker();

    void init(cv::Mat frame, cv::Rect_<double> pedestrian) override;

    cv::Rect2d update(cv::Mat) override;

    void reinit(cv::Rect2d boundingBox);

private:
    cv::Rect_<double> pedestrianBox;
    cv::Ptr<cv::legacy::TrackerTLD> tracker;

    void denoise(cv::Mat frame);

    cv::Mat frame;
};


#endif //TRACKING_TLDTRACKER_H