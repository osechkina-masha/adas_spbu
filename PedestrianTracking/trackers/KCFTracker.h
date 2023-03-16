#ifndef OPTICALFLOWTRACKING_KCFTRACKER_H
#define OPTICALFLOWTRACKING_KCFTRACKER_H


#include <opencv2/tracking.hpp>
#include "Tracker.h"
#include "TrackerPattern.h"

class KCFTracker : public TrackerPattern {

public:
    KCFTracker();

protected:

    void denoise(cv::Mat frame) override;


private:
};

#endif //OPTICALFLOWTRACKING_KCFTRACKER_H
