#ifndef TRACKING_GOTURNTRACKER_H
#define TRACKING_GOTURNTRACKER_H

#include <opencv2/tracking.hpp>
#include "Tracker.h"
#include "TrackerPattern.h"

class GOTURNTracker : public TrackerPattern {


public:
    GOTURNTracker();

protected:

    void denoise(cv::Mat frame) override;

private:
};


#endif //TRACKING_GOTURNTRACKER_H
