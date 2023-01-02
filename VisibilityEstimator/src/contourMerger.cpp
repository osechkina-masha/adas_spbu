#include "opencv2/imgcodecs.hpp"

#include <filesystem>
#include <algorithm>
#include <set>
#include <utility>

#include "contourOrdering.h"
#include "contourMerger.h"

namespace edge_detector
{
    std::vector<std::vector<cv::Point>> ContourMerger::connectContours(
        std::vector<std::vector<cv::Point>> contours,
        int imageSizeX,
        int imageSizeY)
    {
        // Making matrix where points are -1 if they don't belong to any contour, and they can be positive,
        // if they belong to any contour; and positive value is the index of the contour they belong to.
        std::vector<std::vector<int>> indexMatrix(imageSizeX, std::vector(imageSizeY, -1));
        for (int contour = 0; contour < contours.size(); contour++)
        {
            for (int pointIndex = 0; pointIndex < contours[contour].size(); pointIndex++)
            {
                indexMatrix[contours[contour][pointIndex].x][contours[contour][pointIndex].y] = contour;
            }
        }

        // Sorting by values the dictionary where key is contour index, value is contour size.
        auto sortedContoursIndices = ContourOrdering::sortContoursByTheirSizesDescending(contours);

        // подумать, чем ограничить количество итераций
        // Merging contours.
        std::vector<std::vector<cv::Point>> newContours(contours);
        for (int i = 0; i < (sortedContoursIndices.size() < 15 ? sortedContoursIndices.size() : 15); i++)
        {
            newContours = mergeContours(newContours, indexMatrix, sortedContoursIndices[i], contours[sortedContoursIndices[i]]);
        }

        indexMatrix.clear();
        sortedContoursIndices.clear();

        return removeZeroSizeContours(newContours);
    }

    std::set<int> ContourMerger::findNearestContours(
        std::vector<std::vector<cv::Point>> contours,
        int mainContourIndex,
        std::vector<std::vector<int>> contourIndexMatrix,
        cv::Point point)
    {
        std::set<int> foundContours;
        const int neighborhoodSize = 5;

        for (int xDiff = -neighborhoodSize; xDiff <= neighborhoodSize; xDiff++)
        {
            for (int yDiff = -neighborhoodSize; yDiff <= neighborhoodSize; yDiff++)
            {
                if (point.x + xDiff <= 0 || point.y + yDiff <= 0
                    || point.x + xDiff >= contourIndexMatrix.size() || point.y + yDiff >= contourIndexMatrix[0].size())
                {
                    continue;
                }
                auto contourIndex = contourIndexMatrix[point.x + xDiff][point.y + yDiff];
                if (contourIndex != -1 && !contours[contourIndex].empty() && contourIndex != mainContourIndex)
                {
                    foundContours.insert(contourIndex);
                }
            }
        }

        return foundContours;
    }

    // Returns vector where some contours have been merged.
    std::vector<std::vector<cv::Point>> ContourMerger::mergeContours(
        std::vector<std::vector<cv::Point>> contours,
        const std::vector<std::vector<int>>& indexMatrix,
        int mainContourIndex,
        std::vector<cv::Point> mainContour)
    {
        auto pointWithMaxY = ContourOrdering::findPointWithMaxY(std::move(mainContour));

        auto nearestContours = findNearestContours(contours, mainContourIndex, indexMatrix, pointWithMaxY);

        if (nearestContours.empty())
        {
            return contours;
        }

        std::vector<cv::Point> zeroSizeVector;
        std::vector<std::vector<cv::Point>> mergedContours;
        mergedContours.reserve(contours.size() - nearestContours.size());
        for (auto i = 0; i < std::ssize(contours); i++)
        {
            if (i == mainContourIndex)
            {
                auto sizeOfMergedContour = std::ssize(contours[mainContourIndex]);
                for (auto contourIndex : nearestContours)
                {
                    sizeOfMergedContour += std::ssize(contours[contourIndex]);
                }

                std::vector<cv::Point> newContour;
                newContour.reserve(sizeOfMergedContour);
                newContour.insert(newContour.end(), contours[mainContourIndex].begin(), contours[mainContourIndex].end());
                for (auto neighbor : nearestContours)
                {
                    newContour.insert(newContour.end(), contours[neighbor].begin(), contours[neighbor].end());
                }

                mergedContours.push_back(newContour);

                continue;
            }

            if (nearestContours.find(i) != nearestContours.end())
            {
                mergedContours.push_back(zeroSizeVector);
                continue;
            }

            mergedContours.push_back(contours[i]);
        }

        return mergeContours(mergedContours, indexMatrix, mainContourIndex, mergedContours[mainContourIndex]);
    }

    std::vector<std::vector<cv::Point>> ContourMerger::removeZeroSizeContours(std::vector<std::vector<cv::Point>> contours)
    {
        auto iter = contours.begin();
        while (iter != contours.end())
        {
            if (iter->empty())
            {
                iter->clear();
                iter = contours.erase(iter);
                continue;
            }

            advance(iter, 1);
        }

        return contours;
    }
}