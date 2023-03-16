#ifndef TRACKING_GOTURNTRACKER_H
#define TRACKING_GOTURNTRACKER_H

#include <opencv2/tracking.hpp>
#include "Tracker.h"

class GOTURNTracker : public Tracker {


public:
    GOTURNTracker();

    void init(const std::string &path, cv::Rect2d pedestrian, int nFrame) override;

    cv::Rect2d getNextPedestrianPosition() override;

    void reinit(cv::Rect2d boundingBox);

private:
    cv::VideoCapture capture;
    cv::Rect2i pedestrianBox;
    cv::Ptr<cv::TrackerGOTURN> tracker;
    cv::Mat frame;

    void denoise(cv::Mat frame);
};


#endif //TRACKING_GOTURNTRACKER_H
