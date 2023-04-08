#include "TrackerPattern.h"

void TrackerPattern::init(cv::Mat frame, cv::Rect2d pedestrian) {
    pedestrianBox = pedestrian;
    this->frame = frame;
    denoise(frame);
    tracker->init(frame, pedestrianBox);
}

cv::Rect2d TrackerPattern::update(cv::Mat frame) {
    denoise(frame);
    this->frame = frame;
    if (!tracker->update(frame, pedestrianBox)) {}
    return pedestrianBox;
}

void TrackerPattern::reinit(cv::Rect2d boundingBox) {
    tracker->init(frame, boundingBox);
    pedestrianBox = boundingBox;
}

void TrackerPattern::denoise(cv::Mat frame) {

}

