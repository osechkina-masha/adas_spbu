
#ifndef OPTICALFLOWTRACKING_OPTICALFLOWTRACKER_H
#define OPTICALFLOWTRACKING_OPTICALFLOWTRACKER_H


#include "Tracker.h"

class OpticalFlowTracker : public Tracker {


public:
    OpticalFlowTracker();
    void init(const std::string& path, cv::Rect2d pedestrian, int nFrame) override;
    cv::Rect2d getNextPedestrianPosition() override;

    void reinit(cv::Rect2d boundingBox);

private:
    cv::Mat oldFrame;
    cv::Mat oldGray;
    std::vector<cv::Point2f> oldFeatures, newFeatures;
    int featuresCount = 25;
    void updateBoxPosition();
    cv::Point2i getBoxMotion();
    cv::VideoCapture capture;
    cv::Rect2d pedestrianBox;
    void denoise(cv::Mat frame);
    std::vector<cv::Point2f> selectGoodFeatures(std::vector<uchar> &status, cv::Mat &frame);
};


#endif //OPTICALFLOWTRACKING_OPTICALFLOWTRACKER_H
