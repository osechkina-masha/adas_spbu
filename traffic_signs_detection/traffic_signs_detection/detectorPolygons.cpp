#include <iostream>
#include "detectorPolygons.h"


std::vector<cv::Vec3f> detectorPolygons::detectCircle(const cv::Mat &frame)
{
    cv::Mat grayImage;
    cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
    medianBlur(grayImage, grayImage, 5);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(grayImage, circles, cv::HOUGH_GRADIENT, 1,
        grayImage.rows / 16,  
        100, 30, 10, 40); 
    return circles;
}