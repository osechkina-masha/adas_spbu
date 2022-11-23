#ifndef VISUALIZER_PARABOLA_H
#define VISUALIZER_PARABOLA_H

#include "models.h"

namespace models {
    class [[maybe_unused]] parabola : public models {
    private:
        [[maybe_unused]] double firstCoefficient;
        [[maybe_unused]] double secondCoefficient;
        [[maybe_unused]] double thirdCoefficient;
    public:
        [[maybe_unused]] parabola(const cv::Point2i& firstPoint, const cv::Point2i& secondPoint, const cv::Point2i& thirdPoint);

        [[maybe_unused]] [[maybe_unused]] cv::Vec3d parabolaCoefficient() const;
    };
}

#endif