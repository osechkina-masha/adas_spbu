#include "MyTracker.h"
#include "OpticalFlowTracker.h"
#include "KCFTracker.h"
#include "CSRTTracker.h"
#include "TLDTracker.h"
#include "GOTURNTracker.h"
#include <utility>
#include <vector>
#include <numeric>
#include <iostream>
#include "consts.h"

MyTracker::MyTracker() {
    weightOfTracker.clear();
    trackers.clear();
    std::shared_ptr<Tracker> opticalFlowTracker(new OpticalFlowTracker());
    std::shared_ptr<Tracker> kcfTracker(new KCFTracker());
    std::shared_ptr<Tracker> csrtTracker(new CSRTTracker());
    std::shared_ptr<Tracker> tldTracker(new TLDTracker());

    trackers.push_back(opticalFlowTracker);
    trackers.push_back(kcfTracker);
    trackers.push_back(csrtTracker);
    trackers.push_back(tldTracker);

    setDefaultWeights();
}

void MyTracker::setWeights(std::vector<double> newWeights){
    weights = std::move(newWeights);
}

void MyTracker::setDefaultWeights(){
    weights = std::vector<double> {1,1,1,1};
}

cv::Rect2d MyTracker::getMeanResult(std::vector<cv::Rect2d> &boundingBoxes) {
    cv::Point2d avgCenter = {0, 0};
    cv::Point2d avgParams = {0, 0};
    double sumWeight = std::reduce(weights.begin(), weights.end());
    for (int i = 0; i < boundingBoxes.size(); i++) {
        avgCenter += {boundingBoxes[i].x * weights[i], boundingBoxes[i].y * weights[i]};
        avgParams += {boundingBoxes[i].width * weights[i], boundingBoxes[i].height * weights[i]};
    }
    avgCenter.x /= sumWeight;
    avgCenter.y /= sumWeight;
    avgParams.x /= sumWeight;
    avgParams.y /= sumWeight;
    return cv::Rect2d(avgCenter, cv::Size(avgParams.x, avgParams.y));
}

cv::Rect2d MyTracker::getNextPedestrianPosition() {
    static int nFrame = 0;
    nFrame++;
    std::vector<cv::Rect2d> boundingBoxes;
    for (const auto& tracker: trackers) {
        boundingBoxes.push_back(tracker->getNextPedestrianPosition());
    }
    cv::Rect2d newBoundingBox = getMeanResult(boundingBoxes);
    if (nFrame == framesBeforeReinitialization) {
        reinit(newBoundingBox);
        nFrame = 0;
    }
    return newBoundingBox;
}

void MyTracker::init(const std::string &path, cv::Rect2d pedestrian, int nFrame) {
    for (const auto& tracker: trackers) {
        tracker->init(path, pedestrian, nFrame);
    }
}

void MyTracker::reinit(cv::Rect2d boundingBox) {
    for (const auto& tracker: trackers) {
        tracker->reinit(boundingBox);
    }
}
