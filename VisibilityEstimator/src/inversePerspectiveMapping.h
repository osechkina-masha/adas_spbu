#pragma once
#ifndef INVERSEPERSPECTIVEMAPPING_H
#define INVERSEPERSPECTIVEMAPPING_H

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <string>
#include <vector>

namespace distance_estimator
{
    /**
     * @brief Class that provides a set of methods for inverse perspective mapping.
    */
    class InversePerspectiveMapping
    {
    public:
        InversePerspectiveMapping(
            int sizeX,
            int sizeY,
            int focalLength,
            float cameraHeight) :
            sizeX(sizeX), sizeY(sizeY), focalLength(focalLength), cameraHeight(cameraHeight)
        {
        }

        cv::Mat inversePerspectiveMap(const cv::Mat& frame, cv::Point vanishingPoint, cv::Point farthestVisiblePoint);

    private:
        int sizeX;
        int sizeY;

        float cameraHeight;
        int focalLength;

        [[nodiscard]] cv::Mat getProjectiveTransfromationMatrix(cv::Size size, float pitch) const;

        static cv::Mat getMatrixWithoutSecondColumn(cv::Mat matrix);
    };
}

#endif