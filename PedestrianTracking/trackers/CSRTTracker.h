

#ifndef PEDESTRIANTRACKING_CSRTTRACKER_H
#define PEDESTRIANTRACKING_CSRTTRACKER_H


#include <opencv2/tracking.hpp>
#include "Tracker.h"
#include "TrackerPattern.h"

class CSRTTracker : public TrackerPattern {

public:
    CSRTTracker();

protected:
    void denoise(cv::Mat frame) override;

private:

};

#endif //PEDESTRIANTRACKING_CSRTTRACKER_H