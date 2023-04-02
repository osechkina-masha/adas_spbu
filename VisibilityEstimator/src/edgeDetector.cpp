#include "opencv2/imgcodecs.hpp"

#include <filesystem>
#include <algorithm>

#include "edgeDetector.h"
#include "contourOrdering.h"

namespace edge_detector
{
    cv::Point EdgeDetector::findFarthestVisiblePoint(const std::vector<std::vector<cv::Point>>& contours, cv::Point vanishingPoint)
    {
        std::vector<int> sortedContoursIndices = ContourOrdering::sortContoursByTheirSizesDescending(contours);
        auto numberOfConsideredContours = std::ssize(contours) >= 10 ? 10 : std::ssize(contours);
        cv::Point farthestVisiblePoint = cv::Point(INT_MAX, INT_MAX);

        for (int i = 0; i < numberOfConsideredContours; i++)
        {
            auto pointWithMaxY = ContourOrdering::findPointWithMaxY(contours[sortedContoursIndices[i]]);
            auto coefficient = findCoefficient(vanishingPoint, pointWithMaxY);

            if (coefficient * pointWithMaxY.y < farthestVisiblePoint.y
                && pointWithMaxY.y >= vanishingPoint.y)
            {
                farthestVisiblePoint = cv::Point(pointWithMaxY.x, (int)(coefficient * pointWithMaxY.y));
            }
        }

        sortedContoursIndices.clear();

        return farthestVisiblePoint;
    }

    double EdgeDetector::countContourCos(cv::Point vanishingPoint, cv::Point contourPoint)
    {
        double hypotenuse = sqrt(pow(contourPoint.x - vanishingPoint.x, 2) + pow(contourPoint.y - vanishingPoint.y, 2));
        double cathetus = contourPoint.y - vanishingPoint.y;
        double cosine = cathetus / hypotenuse;

        return cosine;
    }

    double EdgeDetector::findCoefficient(cv::Point vanishingPoint, cv::Point point)
    {
        double cosine = countContourCos(vanishingPoint, point);
        double angle = 180 * acos(cosine) / CV_PI;
        if (angle > 70)
        {
            return 1.064;
        }
        return 1;
    }
}