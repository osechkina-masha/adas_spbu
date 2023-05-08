#include <iostream>
#include "trafficSign.h"
#include "detectorPolygons.h"

void TrafficSign::showTrafficSigns(const cv::Mat &colorImage, const cv::Mat &frame)
{
    std::vector<cv::Rect> rectangles = DetectorPolygons::detectShape(colorImage);
    for (auto &rectangle : rectangles)
    {
        cv::rectangle(frame, rectangle, cv::Scalar(5, 6, 7), 3);
    }
}
