#include <iostream>
#include "KCFTracker.h"

KCFTracker::KCFTracker() {
    tracker = cv::TrackerKCF::create();

}

void KCFTracker::denoise(cv::Mat frame) {
}
