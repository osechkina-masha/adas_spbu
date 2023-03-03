#include "imageSegmentation.h"
#include "detectorPolygons.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>


int main(int argc, char* argv[])
{
  /*  if (argc != 2)
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
        cv::Mat red = ImageSegmentation::highlightRed(frame);
        cv::Mat white = ImageSegmentation::highlightWhite(frame);
        cv::Mat blue = ImageSegmentation::highlightBlue(frame);
        cv::Mat yellow = ImageSegmentation::highlightYellow(frame);
        std::vector<cv::Vec3f> circlesRed = DetectorPolygons::detectCircle(red);
        std::vector<cv::Vec3f> circlesBlue = DetectorPolygons::detectCircle(blue);
        std::vector<cv::Vec3f> circlesYellow = DetectorPolygons::detectCircle(yellow);
        std::vector<cv::Vec3f> circlesWhite = DetectorPolygons::detectCircle(white);
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
    */
    cv::Mat frame = cv::imread("../2022-11-19 151239.png");
   cv::imshow("yellow", frame);
    cv::Mat red = ImageSegmentation::highlightRed(frame);
    cv::Mat white = ImageSegmentation::highlightWhite(frame);
    cv::Mat blue = ImageSegmentation::highlightBlue(frame);
    cv::Mat yellow = ImageSegmentation::highlightYellow(frame);
    std::vector<cv::Vec3f> circlesRed = DetectorPolygons::detectCircle(red);
    std::vector<cv::Vec3f> circlesBlue = DetectorPolygons::detectCircle(blue);
    std::vector<cv::Vec3f> circlesYellow = DetectorPolygons::detectCircle(yellow);
    std::vector<cv::Vec3f> circlesWhite = DetectorPolygons::detectCircle(white);
    std::vector<cv::Vec4f> trianglesRed = DetectorPolygons::detectTriangle(red);
    std::vector<cv::Vec4f> trianglesBlue = DetectorPolygons::detectTriangle(blue);
    std::vector<cv::Vec4f> trianglesYellow = DetectorPolygons::detectTriangle(yellow);
    std::vector<cv::Vec4f> trianglesWhite = DetectorPolygons::detectTriangle(white);
    cv::Mat templateTriangle = cv::imread("t.png", cv::IMREAD_GRAYSCALE);
    int w = templateTriangle.cols;
    int h = templateTriangle.rows;
    for (std::vector<cv::Vec4f>::iterator iter = trianglesRed.begin(); iter != trianglesRed.end(); ++iter) {
        cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f((*iter)[0], (*iter)[1]),
            cv::Size2f(w * (*iter)[2], h * (*iter)[2]),
            (*iter)[3]);
        cv::Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++)
            line(red, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 6);
    }

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