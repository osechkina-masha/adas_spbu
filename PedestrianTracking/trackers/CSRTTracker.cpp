#include <iostream>
#include <opencv2/photo.hpp>
#include "CSRTTracker.h"


CSRTTracker::CSRTTracker() {
    tracker = cv::TrackerCSRT::create();
}

void CSRTTracker::denoise(cv::Mat frame) {
//    cv::fastNlMeansDenoising(frame, frame, 30, 7, 21);
}

