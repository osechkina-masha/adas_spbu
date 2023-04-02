#include "distanceEstimator.h"

namespace distance_estimator
{
    double DistanceEstimator::calculateDistance(int focalLength, int lineY, double cameraHeight, int horizon)
    {
        return focalLength * cameraHeight / (lineY - horizon);
    }
}