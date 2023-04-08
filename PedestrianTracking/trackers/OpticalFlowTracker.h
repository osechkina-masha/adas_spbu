
#ifndef OPTICALFLOWTRACKING_OPTICALFLOWTRACKER_H
#define OPTICALFLOWTRACKING_OPTICALFLOWTRACKER_H


#include "Tracker.h"

class OpticalFlowTracker : public Tracker {


public:
    OpticalFlowTracker();

    void init(cv::Mat frame, cv::Rect2d pedestrian) override;

    cv::Rect2d update(cv::Mat frame) override;

    void reinit(cv::Rect2d boundingBox);

private:
    cv::Mat oldFrame;
    cv::Mat oldGray;
    std::vector<cv::Point2f> oldFeatures, newFeatures;
    int featuresCount = 25;

    void updateBoxPosition();

    cv::Point2i getBoxMotion();

    cv::Rect2d pedestrianBox;

    void denoise(cv::Mat frame);

    std::vector<cv::Point2f> selectGoodFeatures(std::vector<uchar> &status, cv::Mat &frame);
};


#endif //OPTICALFLOWTRACKING_OPTICALFLOWTRACKER_H
