#include "imageSegmentation.h"
#include "detectorPolygons.h"
#include "trafficSign.h"
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
        cv::Mat frame = cv::imread("2.jpg");
        cv::Mat red = ImageSegmentation::highlightRed(frame);
        cv::Mat white = ImageSegmentation::highlightColor(frame, cv::Scalar(0, 0, 150), cv::Scalar(360, 60, 255));
        cv::Mat blue = ImageSegmentation::highlightColor(frame, cv::Scalar(90, 50, 70), cv::Scalar(128, 255, 255));
        cv::Mat yellow = ImageSegmentation::highlightColor(frame, cv::Scalar(20, 100, 100), cv::Scalar(30, 255, 255));
        TrafficSign::showTrafficSigns(red, frame);
        TrafficSign::showTrafficSigns(white, frame);
        TrafficSign::showTrafficSigns(blue, frame);
        TrafficSign::showTrafficSigns(yellow, frame);
        cv::imshow("red", red);
        cv::imshow("white", white);
        cv::imshow("blue", blue);
        cv::imshow("yellow", yellow);
        cv::imshow("detected traffic signs", frame);
        cv::waitKey(0);
    }
}