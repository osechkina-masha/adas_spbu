#ifndef OPTICALFLOWTRACKING_MYTRACKER_H
#define OPTICALFLOWTRACKING_MYTRACKER_H


#include <map>
#include "Tracker.h"

class MyTracker : Tracker {


public:

    MyTracker();

    //reinit every tracker by bounding box
    void reinit(cv::Rect2d boundingBox);

    //tracker initialization. Path - path to video, bounding box, nFrame - number of frame where from tracking should start
    void init(cv::Mat frame, cv::Rect2d pedestrian) override;

    //gives mean result from several trackers
    cv::Rect2d update(cv::Mat frame) override;

    //function to set weights for 4 trackers: optical flow, kcf, csrt, (tld - legacy)
    void setWeights(std::vector<double> weights);

private:
    cv::Mat currentFrame;
    std::vector<std::shared_ptr<Tracker>> trackers;
    std::vector<double> weights;
    std::map<Tracker *, double> weightOfTracker;

    void setDefaultWeights();

    cv::Rect2d getMeanResult(std::vector<cv::Rect2d> &boundingBoxes);

};

#endif //OPTICALFLOWTRACKING_MYTRACKER_H
