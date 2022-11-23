#ifndef VISUALIZER_PLANE_H
#define VISUALIZER_PLANE_H

#include "models.h"

namespace models {
    class [[maybe_unused]] plane : public models{
    private:
        double width;
        double length;
        void initializateLocalPoints();
    public:

        plane(const cv::Vec3d& planeCoordinates, double width, double length);

        // Поменять длину и ширину плоскости
        [[maybe_unused]] void changeWidthAndLength(double width, double length);

        // Соеденить плоскости
        [[maybe_unused]] std::vector<std::shared_ptr<plane>> mergePlanes(Axis axis, double maxAngle, double stepLength, double stepAngle);

        // Получить длину и ширину
        [[maybe_unused]] [[nodiscard]] double getWidth() const;

        [[maybe_unused]] [[nodiscard]] double getLength() const;
    };
}

#endif