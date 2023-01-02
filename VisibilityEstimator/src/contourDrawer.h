#pragma once
#ifndef CONTOURDRAWER_H
#define CONTOURDRAWER_H

#include "opencv2/opencv.hpp"

namespace edge_detector
{
    /**
     * @brief Class for debugging purposes. Provides a set of methods for drawing contours on the image.
    */
    class ContourDrawer
    {
    public:
        static void colorContours(cv::Mat frame, const std::vector<std::vector<cv::Point>>& contours);

        static void drawHorizontalLine(cv::Mat frame, cv::Point point);
    };
}

#endif