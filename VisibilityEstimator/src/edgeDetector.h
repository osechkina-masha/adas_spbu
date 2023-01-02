#pragma once
#ifndef EDGEDETECTOR_H
#define EDGEDETECTOR_H

#include "opencv2/imgproc.hpp"

#include <string>
#include <set>

namespace edge_detector
{
    /**
     * @brief Class for finding farthest visible point on the image.
    */
    class EdgeDetector
    {
    public:
        /**
         * @brief Finds farthest visible point of the road on the image.
         * @param contours is the vector of contours that was found on the image.
         * @param vanishingPoint is the vanishing point of the road borders.
         * @return Farthest visible point of the road on the image.
        */
        static cv::Point findFarthestVisiblePoint(std::vector<std::vector<cv::Point>> contours, cv::Point vanishingPoint);

    private:
        static double countContourCos(cv::Point vanishingPoint, cv::Point contourPoint);

        static double findCoefficient(cv::Point vanishingPoint, cv::Point contour);
    };
}

#endif