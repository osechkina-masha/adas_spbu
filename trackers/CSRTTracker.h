
#ifndef PEDESTRIANTRACKING_CSRTTRACKER_H
#define PEDESTRIANTRACKING_CSRTTRACKER_H


#include <opencv2/tracking.hpp>
#include "Tracker.h"

class CSRTTracker : public Tracker {
    cv::VideoCapture capture;
    cv::Rect2i pedestrianBox;
    cv::Ptr<cv::TrackerCSRT> tracker;
public:
    CSRTTracker();
    void startTracking(const std::string& path, cv::Rect2d pedestrian, int nFrame) override;
    const cv::Rect2d getNextPedestrianPosition() override;
};


#endif //PEDESTRIANTRACKING_CSRTTRACKER_H
