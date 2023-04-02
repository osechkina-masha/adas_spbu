#pragma once

#include "opencv2/imgproc.hpp"

#include <string>
#include <set>

namespace edge_detector
{
    /**
     * @brief Class that provides a set of methods for connecting broken contours.
    */
    class ContourMerger
    {
    public:
        /**
         * @brief Applies contour merging algorithm to the longest contours.
         * @param contours Vector of contours to merge.
         * @param imageSizeX is the image width.
         * @param imageSizeY is the image height.
         * @return Vector of connected contours.
        */
        static std::vector<std::vector<cv::Point>> connectContours(
            const std::vector<std::vector<cv::Point>>& contours,
            int imageSizeX,
            int imageSizeY);

        /**
         * @brief Finds nearest contours in square neighborhood 5x5 pixels around the given point.
         * @param contours Vector of contours.
         * @param mainContourIndex Index of the contour to which the point belongs.
         * @param contourIndexMatrix Matrix which elements are contour indices.
         * @param point Point around which the search is made.
         * @return Set of nearest contours.
        */
        static std::set<int> findNearestContours(
            const std::vector<std::vector<cv::Point>>& contours,
            int mainContourIndex,
            const std::vector<std::vector<int>>& contourIndexMatrix,
            cv::Point point);

        /**
         * @brief Finds the highest point of a contour and merges origin contour with contours found around this point.
         * @param contours Vector of contours.
         * @param indexMatrix Matrix which elements are contour indices.
         * @param mainContourIndex Index of contour that should be merged with nearest contours.
         * @param mainContour Contour that should be merged with nearest contours
         * @return Vector of connected contours.
        */
        static std::vector<std::vector<cv::Point>> mergeContours(
            const std::vector<std::vector<cv::Point>>& contours,
            const std::vector<std::vector<int>>& indexMatrix,
            int mainContourIndex,
            const std::vector<cv::Point>& mainContour);

    private:
        static std::vector<std::vector<cv::Point>> removeZeroSizeContours(std::vector<std::vector<cv::Point>> contours);
    };
}