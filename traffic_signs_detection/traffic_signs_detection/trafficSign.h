#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class TrafficSign
{

public:

    static void TrafficSign::showTrafficSigns(const cv::Mat& colorImage, const cv::Mat& frame);
};
