#include "TrackerPattern.h"

void TrackerPattern::init(const std::string &path, cv::Rect2d pedestrian, int nFrame) {
    capture = cv::VideoCapture(path);
    pedestrianBox = pedestrian;
    for (int i = 0; i < nFrame; i++) { capture >> frame; }
    denoise(frame);
    tracker->init(frame, pedestrianBox);
}

cv::Rect2d TrackerPattern::getNextPedestrianPosition() {
    capture >> frame;
    denoise(frame);
    if (!tracker->update(frame, pedestrianBox)) {
//        std::cout << "failed csrt tracking" << std::endl;
    }
    return pedestrianBox;
}

void TrackerPattern::reinit(cv::Rect2d boundingBox) {
    tracker->init(frame, boundingBox);
    pedestrianBox = boundingBox;
}

void TrackerPattern::denoise(cv::Mat frame) {

}

//void TrackerPattern::denoise(cv::Mat frame) {
//
//}
