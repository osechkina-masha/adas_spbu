#pragma once

#include <opencv2/opencv.hpp>

namespace edge_detector
{
    /**
     * @brief Class that provides a set of methods for ordering contours.
    */
    class ContourOrdering
    {
    public:
        /**
         * @brief Finds highest point of the contour.
         * @param contour is the contour on which the point is searched.
         * @return Returns point with maximum Y value.
        */
        static cv::Point findPointWithMaxY(const std::vector<cv::Point>& contour);

        /**
         * @brief Sorts contours by their sizes int descending order 
         * @param contours is the vector of contours that should be sorted.
         * @return Sorted vector of contours.
        */
        static std::vector<int> sortContoursByTheirSizesDescending(std::vector<std::vector<cv::Point>> contours);
    };
}