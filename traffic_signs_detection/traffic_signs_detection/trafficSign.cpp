#include <iostream>
#include "trafficSign.h"
#include "detectorPolygons.h"
#include "imageSegmentation.h"

std::vector <std::vector<cv::Rect>> DetectorTrafficSign::detectTrafficSigns(const cv::Mat &frame)
{   
    cv::Mat red = ImageSegmentation::highlightRed(frame);
    cv::Mat white = ImageSegmentation::highlightColor(frame, cv::Scalar(0, 0, 150), cv::Scalar(360, 60, 255));
    cv::Mat blue = ImageSegmentation::highlightColor(frame, cv::Scalar(90, 50, 70), cv::Scalar(128, 255, 255));
    cv::Mat yellow = ImageSegmentation::highlightColor(frame, cv::Scalar(20, 100, 100), cv::Scalar(30, 255, 255));
    std::vector <std::vector<cv::Rect>>rectangles;
    rectangles.push_back(DetectorPolygons::detectShape(red));
    rectangles.push_back(DetectorPolygons::detectShape(white));
    rectangles.push_back(DetectorPolygons::detectShape(blue));
    rectangles.push_back(DetectorPolygons::detectShape(yellow));
    return rectangles;
}
