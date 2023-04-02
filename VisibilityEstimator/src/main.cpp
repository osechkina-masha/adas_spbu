#include <iostream>
#include <deque>
#include <filesystem>

#include "opencv2/highgui.hpp"

#include "edgeDetector.h"
#include "imageFilter.h"
#include "contourDetector.h"
#include "distanceEstimator.h"

/**
 * @brief Entry point. Sample of command line input: ./VisibilityEstimator video_name.mp4 1.3 500 328 194
 * @param argc is a number of arguments. There must be six arguments.
 * @param argv is an array with the following required arguments:
path to the video,
camera height in meters,
focal length of the camera in pixels,
X coordinate of vanishing point,
Y coordinate of vanishing point
*/
int main(int argc, char** argv)
{
    std::string path = argv[1];

    if (!std::filesystem::exists(path))
    {
        std::cerr << "Invalid path.";
        return 0;
    }

    double cameraHeight = std::strtod(argv[2], nullptr);
    int focalLength = std::strtol(argv[3], nullptr, 10);
    int vanishingPointX = std::strtol(argv[4], nullptr, 10);
    int vanishingPointY = std::strtol(argv[5], nullptr, 10);
    cv::Point vanishingPoint = cv::Point(vanishingPointX, vanishingPointY);

    cv::VideoCapture video(path);

    std::deque<cv::Mat> frameDeque;

    for (int i = 0; i < 4; i++)
    {
        cv::Mat frame;
        video >> frame;
        if (frame.empty())
        {
            break;
        }

        cv::Mat grayFrame;
        cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        frameDeque.push_back(grayFrame);
    }

    while (true)
    {
        cv::Mat averageImage = edge_detector::ImageFilter::selectMinimumForEachPixel(frameDeque);
        auto contours = edge_detector::ContourDetector::detectContours(averageImage);

        cv::Point farthestVisiblePoint = edge_detector::EdgeDetector::findFarthestVisiblePoint(contours, vanishingPoint);
        std::cout << distance_estimator::DistanceEstimator::calculateDistance(focalLength, farthestVisiblePoint.y, cameraHeight, vanishingPoint.y);
        std::cout << '\n';

        averageImage.release();
        contours.clear();

        auto dequeFirstElement = frameDeque.front();
        frameDeque.pop_front();
        dequeFirstElement.release();

        cv::Mat frame;
        video >> frame;
        if (frame.empty())
        {
            break;
        }

        cv::Mat grayFrame;
        cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        frameDeque.push_back(grayFrame);

        if (cv::waitKey(0) == 'q')
        {
            break;
        }
    }

    frameDeque.clear();
    video.release();

    return 0;
}