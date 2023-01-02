#include "imageSegmentation.h"
#include "detectorPolygons.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>


int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        return 1;
    }
    cv::VideoCapture video(argv[1]);
    if (!video.isOpened())
    {
        return 1;
    }
    while (video.isOpened())
    {
        cv::Mat frame;
        video >> frame;
        std::cout << 1;
        cv::Mat red = imageSegmentation::highlightRed(frame);
        cv::Mat white = imageSegmentation::highlightWhite(frame);
        cv::Mat blue = imageSegmentation::highlightBlue(frame);
        cv::Mat yellow = imageSegmentation::highlightYellow(frame);
        std::vector<cv::Vec3f> circlesRed = detectorPolygons::detectCircle(red);
        std::vector<cv::Vec3f> circlesBlue = detectorPolygons::detectCircle(blue);
        std::vector<cv::Vec3f> circlesYellow = detectorPolygons::detectCircle(yellow);
        std::vector<cv::Vec3f> circlesWhite = detectorPolygons::detectCircle(white);
        for (size_t i = 0; i < circlesRed.size(); i++)
        {
            cv::Vec3i coordinates = circlesRed[i];
            int radius = coordinates[2];
            cv::Point upperVertex = cv::Point(coordinates[0] - radius - 0.5, coordinates[1] + radius + 0.5);
            cv::Point lowerVertex = cv::Point(coordinates[0] + radius + 0.5, coordinates[1] - radius - 0.5);
            cv::rectangle(red, upperVertex, lowerVertex, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
            cv::rectangle(frame, upperVertex, lowerVertex, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
        }

        for (size_t i = 0; i < circlesBlue.size(); i++)
        {
            cv::Vec3i coordinates = circlesBlue[i];
            int radius = coordinates[2];
            cv::Point upperVertex = cv::Point(coordinates[0] - radius - 0.5, coordinates[1] + radius + 0.5);
            cv::Point lowerVertex = cv::Point(coordinates[0] + radius + 0.5, coordinates[1] - radius - 0.5);
            cv::rectangle(blue, upperVertex, lowerVertex, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
            cv::rectangle(frame, upperVertex, lowerVertex, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
        }

        for (size_t i = 0; i < circlesYellow.size(); i++)
        {
            cv::Vec3i coordinates = circlesYellow[i];
            int radius = coordinates[2];
            cv::Point upperVertex = cv::Point(coordinates[0] - radius - 0.5, coordinates[1] + radius + 0.5);
            cv::Point lowerVertex = cv::Point(coordinates[0] + radius + 0.5, coordinates[1] - radius - 0.5);
            cv::rectangle(yellow, upperVertex, lowerVertex, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
            cv::rectangle(frame, upperVertex, lowerVertex, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
        }

        for (size_t i = 0; i < circlesWhite.size(); i++)
        {
            cv::Vec3i coordinates = circlesWhite[i];
            int radius = coordinates[2];
            cv::Point upperVertex = cv::Point(coordinates[0] - radius - 0.5, coordinates[1] + radius + 0.5);
            cv::Point lowerVertex = cv::Point(coordinates[0] + radius + 0.5, coordinates[1] - radius - 0.5);
            cv::rectangle(white, upperVertex, lowerVertex, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
            cv::rectangle(frame, upperVertex, lowerVertex, cv::Scalar(255, 0, 0), 2, cv::LINE_8);
        }
        cv::imshow("red", red);
        cv::imshow("white", white);
        cv::imshow("blue", blue);
        cv::imshow("yellow", yellow);
        cv::imshow("detected traffic signs", frame);
        cv::waitKey(0);
    }
}