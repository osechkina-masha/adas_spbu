
#include <iostream>
#include "TLDTracker.h"
#include <opencv2/photo.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>

void TLDTracker::init(cv::Mat frame, cv::Rect2d pedestrian) {
    pedestrianBox = pedestrian;
    this->frame = frame;
    tracker = cv::legacy::TrackerTLD::create();
    denoise(frame);
    tracker->init(frame, pedestrianBox);
}

void TLDTracker::denoise(cv::Mat frame) {
    cv::fastNlMeansDenoising(frame, frame, 30, 7, 21);
}

TLDTracker::TLDTracker() = default;


cv::Rect2d TLDTracker::update(cv::Mat frame) {
    this->frame = frame;
    denoise(frame);
    if (!tracker->update(frame, pedestrianBox)) {}
    return pedestrianBox;
}

void TLDTracker::reinit(cv::Rect2d boundingBox) {
    tracker->init(frame, boundingBox);
    pedestrianBox = boundingBox;
}