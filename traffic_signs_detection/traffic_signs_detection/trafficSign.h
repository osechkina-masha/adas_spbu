#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class DetectorTrafficSign
{
public:

    static std::vector <std::vector<cv::Rect>> detectTrafficSigns(const cv::Mat &frame);
};
