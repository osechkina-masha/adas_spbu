#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class DetectorPolygons
{

public:
    
    // Detects shapes in an image
    static std::vector<cv::Rect> detectShape(const cv::Mat &frame);

private:

    static double angle (cv::Point pt1, cv::Point pt2, cv::Point pt0);
};
