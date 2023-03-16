
#ifndef TRACKING_TLDTRACKER_H
#define TRACKING_TLDTRACKER_H
#include <opencv2/tracking.hpp>
#include "Tracker.h"
#include <opencv2/tracking/tracking_legacy.hpp>

class TLDTracker : public Tracker{

public:
    TLDTracker();
    void init(const std::string& path, cv::Rect_<double> pedestrian, int nFrame) override;
    cv::Rect2d getNextPedestrianPosition() override;

    void reinit(cv::Rect2d boundingBox);

private:
    cv::VideoCapture capture;
    cv::Rect_<double> pedestrianBox;
    cv::Ptr<cv::legacy::TrackerTLD> tracker;
    void denoise(cv::Mat frame);
    cv::Mat frame;
};


#endif //TRACKING_TLDTRACKER_H
