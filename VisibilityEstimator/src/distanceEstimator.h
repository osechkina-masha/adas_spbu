#pragma once
#ifndef DISTANCEESTIMATOR_H
#define DISTANCEESTIMATOR_H

namespace distance_estimator
{
    /**
     * @brief Class that provides a method for calculating a distance to the line on the image.
    */
    class DistanceEstimator
    {
    public:
        /**
         * @brief Calculates the distance from camera to the line on the image using formula based on intersection of two planes.
         * @param focalLength is the focal length of a camera in pixels.
         * @param lineY is the Y value of a line to which distance should be calculated.
         * @param cameraHeight is the camera height in meters.
         * @param horizon is the Y value of horizon on the image.
         * @return Returns the distance to the line in meters.
        */
        static double calculateDistance(
            int focalLength,
            int lineY,
            double cameraHeight,
            int horizon);
    };
}

#endif