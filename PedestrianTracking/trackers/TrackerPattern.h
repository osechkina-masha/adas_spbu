#ifndef ADAS_SPBU_TRACKERPATTERN_H
#define ADAS_SPBU_TRACKERPATTERN_H

#include <opencv2/video/tracking.hpp>
#include "Tracker.h"

class TrackerPattern : public Tracker {
public:
    void init(cv::Mat frame, cv::Rect2d pedestrian) override;

    cv::Rect2d update(cv::Mat frame) override;

    void reinit(cv::Rect2d boundingBox) override;

protected:
    virtual void denoise(cv::Mat frame);

    cv::Ptr<cv::Tracker> tracker;

private:
    cv::Rect2i pedestrianBox;
    cv::Mat frame;
};


#endif //ADAS_SPBU_TRACKERPATTERN_H
