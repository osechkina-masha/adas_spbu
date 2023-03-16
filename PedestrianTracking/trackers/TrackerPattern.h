#ifndef ADAS_SPBU_TRACKERPATTERN_H
#define ADAS_SPBU_TRACKERPATTERN_H


#include <opencv2/video/tracking.hpp>
#include "Tracker.h"

class TrackerPattern : public Tracker {
public:
    void init(const std::string &path, cv::Rect2d pedestrian, int nFrame) override;

    cv::Rect2d getNextPedestrianPosition() override;

    void reinit(cv::Rect2d boundingBox) override;

protected:
    virtual void denoise(cv::Mat frame);

    cv::Ptr<cv::Tracker> tracker;

private:
    cv::VideoCapture capture;
    cv::Rect2i pedestrianBox;
    cv::Mat frame;
};


#endif //ADAS_SPBU_TRACKERPATTERN_H
