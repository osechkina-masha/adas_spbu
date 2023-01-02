#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class detectorPolygons
{

public:
    
    // Detects circles in an image
    static std::vector<cv::Vec3f> detectCircle(const cv::Mat &frame);
};
