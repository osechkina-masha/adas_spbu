
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "OpticalFlowTracker.h"
#include "constsForOpticalFlow.h"

using namespace cv;

OpticalFlowTracker::OpticalFlowTracker() = default;

Point2f getMean(const std::vector<Point2f> &vec) {
    Point2f mean(0.f, 0.f);
    for (const Point2f &p: vec) {
        mean += p;
    }
    mean /= (float) vec.size();
    return mean;

}

Point2i OpticalFlowTracker::getBoxMotion() {
    Point2f oldMean = getMean(oldFeatures);
    Point2f newMean = getMean(newFeatures);
    return newMean - oldMean;
}

cv::Rect2d getIntersection(const Mat &img, const Rect2d &box) {
    Point2d leftCorner = {max(0.0, box.x - box.width / 2), max(0.0, box.y - box.width / 2)};
    Point2d rightCorner = {min(double(img.rows), box.x + box.width / 2), min(double(img.cols), box.y + box.width / 2)};
    return Rect2d(leftCorner, Size(rightCorner.x - leftCorner.x, rightCorner.y - leftCorner.y));
}

void OpticalFlowTracker::updateBoxPosition() {
    Point2f boxMotion = getBoxMotion();
    pedestrianBox.x += boxMotion.x;
    pedestrianBox.y += boxMotion.y;
}

void OpticalFlowTracker::init(cv::Mat oldFrame, Rect2d pedestrian) {
    pedestrianBox = pedestrian;
    cvtColor(oldFrame, oldGray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(oldGray(pedestrianBox), oldFeatures, featuresCount, qualityLevel, minDistance, Mat(), blockSize,
                        useHarrisDetector, hassisK);
}

void OpticalFlowTracker::reinit(cv::Rect2d boundingBox) {
    pedestrianBox = boundingBox;
}

Rect2d OpticalFlowTracker::update(cv::Mat newFrame) {
    Mat newGray;
    cvtColor(newFrame, newGray, COLOR_BGR2GRAY);
    std::vector<uchar> status;
    std::vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), termCriteriaMaxCount,
                                         termCriteriaEpsilon);
    calcOpticalFlowPyrLK(oldGray(pedestrianBox), newGray(pedestrianBox), oldFeatures, newFeatures, status, err,
                         Size(lkWindowWidth, lkWindowHeight), lkMaxDepth,
                         criteria);
    std::vector<Point2f> good_new = selectGoodFeatures(status, newFrame);
    updateBoxPosition();
    oldGray = newGray.clone();
    oldFeatures = good_new;
    if (oldFeatures.size() < minPointsToTrack) {
        goodFeaturesToTrack(newGray(pedestrianBox), oldFeatures, featuresCount, qualityLevel, minDistance, Mat(),
                            blockSize, useHarrisDetector, hassisK);
    }
    oldFrame = newFrame.clone();
    return pedestrianBox;
}

std::vector<Point2f> OpticalFlowTracker::selectGoodFeatures(std::vector<uchar> &status,
                                                            Mat &frame) {
    std::vector<Point2f> good_new;
    for (uint i = 0; i < oldFeatures.size(); i++) {
        if (status[i] == 1) {
            good_new.push_back(newFeatures[i]);
        }
    }
    return good_new;
}

void OpticalFlowTracker::denoise(cv::Mat frame) {
}
