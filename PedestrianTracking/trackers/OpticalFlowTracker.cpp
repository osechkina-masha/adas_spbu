
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "OpticalFlowTracker.h"
#include "constsForOpticalFlow.h"

using namespace cv;

OpticalFlowTracker::OpticalFlowTracker() = default;

void OpticalFlowTracker::init(cv::Mat oldFrame, Rect2d pedestrian) {
    pedestrianBox = pedestrian;
    cvtColor(oldFrame, oldGray, COLOR_BGR2GRAY);
    this->oldFrame = oldFrame;
}

void OpticalFlowTracker::reinit(cv::Rect2d boundingBox) {
    pedestrianBox = boundingBox;
}

Rect2d OpticalFlowTracker::update(cv::Mat newFrame) {
    Mat newGray;
    cvtColor(newFrame, newGray, COLOR_BGR2GRAY);
    Mat flow(oldFrame.size(), CV_32FC2);

    calcOpticalFlowFarneback(oldGray, newGray, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma,
                             flags);

    Mat flow_parts[2];
    split(flow, flow_parts);


    Mat xSubMat(flow_parts[0], pedestrianBox);
    Mat ySubMat(flow_parts[1], pedestrianBox);
    double dx = sum(xSubMat)[0] / (pedestrianBox.x * pedestrianBox.y);
    double dy = sum(ySubMat)[0] / (pedestrianBox.x * pedestrianBox.y);

    pedestrianBox.x += dx;
    pedestrianBox.y += dy;

    oldFrame = newGray;
    return pedestrianBox;
}


void OpticalFlowTracker::denoise(cv::Mat frame) {
}
