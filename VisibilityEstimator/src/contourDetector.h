#pragma once
#ifndef CONTOURDETECTOR_H
#define CONTOURDETECTOR_H

#include "opencv2/opencv.hpp"

namespace edge_detector
{
    /**
     * @brief Class that provides a method for detecting contours on the image.
    */
    class ContourDetector
    {
    public:
        /**
         * @brief Detects contours using Canny, Suzuki algorithm. And them applies to them algorithm for merging.
         * @param image is the image on which contours should be detected.
         * @return Vector of detected contours.
        */
        static std::vector<std::vector<cv::Point>> detectContours(const cv::Mat& image);
    };
}

#endif