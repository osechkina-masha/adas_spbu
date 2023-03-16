#include <iostream>
#include "GOTURNTracker.h"


GOTURNTracker::GOTURNTracker() {
    tracker = cv::TrackerGOTURN::create();

}

void GOTURNTracker::denoise(cv::Mat frame) {
}
