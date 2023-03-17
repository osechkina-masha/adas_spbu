#include <iostream>
#include "trafficSign.h"
#include "detectorPolygons.h"

void TrafficSign::showTrafficSigns(const cv::Mat& colorImage, const cv::Mat &frame)
{
    std::vector<cv::Vec3f> circles = DetectorPolygons::detectCircle(colorImage);
    std::vector<cv::Vec4f> triangles = DetectorPolygons::detectTriangle(colorImage);
    cv::Mat templateTriangle = cv::imread("templates/template.png", cv::IMREAD_GRAYSCALE);
    int width = templateTriangle.cols;
    int height = templateTriangle.rows;
    for (std::vector<cv::Vec4f>::iterator iter = triangles.begin(); iter != triangles.end(); ++iter) {
        cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f((*iter)[0], (*iter)[1]),
            cv::Size2f(width * (*iter)[2], height * (*iter)[2]),
            (*iter)[3]);
        cv::Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
        {
            line(colorImage, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 6);
            line(frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 6);
        }
    }
    for (auto &circle : circles)
    {
        cv::Vec3i coordinates = circle;
        int radius = coordinates[2];
        cv::Point upperVertex = cv::Point(coordinates[0] - radius - 0.5, coordinates[1] + radius + 0.5);
        cv::Point lowerVertex = cv::Point(coordinates[0] + radius + 0.5, coordinates[1] - radius - 0.5);
        cv::rectangle(colorImage, upperVertex, lowerVertex, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
        cv::rectangle(frame, upperVertex, lowerVertex, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
    }
}