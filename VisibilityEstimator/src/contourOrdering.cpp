#include "contourOrdering.h"

namespace edge_detector
{
    std::vector<int> ContourOrdering::sortContoursByTheirSizesDescending(std::vector<std::vector<cv::Point>> contours)
    {
        std::vector<std::pair<int, int>> sizeOfContours;
        std::vector<int> sortedContoursIndices;

        for (auto contourIndex = 0; contourIndex < (int)std::ssize(contours); contourIndex++)
        {
            sizeOfContours.emplace_back( contourIndex, (int)std::ssize(contours[contourIndex]) );
        }

        sort(sizeOfContours.begin(), sizeOfContours.end(),
            [](auto& left, auto& right)
            {
                return left.second > right.second;
            });

        for (auto i = 0; i < std::ssize(sizeOfContours); i++)
        {
            sortedContoursIndices.push_back(sizeOfContours[i].first);
        }

        sizeOfContours.clear();

        return sortedContoursIndices;
    }

    cv::Point ContourOrdering::findPointWithMaxY(const std::vector<cv::Point>& contour)
    {
        return *max_element(contour.begin(), contour.end(),
            [](auto& left, auto& right)
            {
                return left.y > right.y;
            });
    }
}