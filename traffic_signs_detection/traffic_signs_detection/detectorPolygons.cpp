#include <iostream>
#include "detectorPolygons.h"


std::vector<cv::Vec3f> DetectorPolygons::detectCircle(const cv::Mat &frame)
{
    cv::Mat grayImage;
    cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
  //  medianBlur(grayImage, grayImage, 5);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(grayImage, circles, cv::HOUGH_GRADIENT, 1,
        grayImage.rows / 16,  
        100, 30, 10, 40); 
    return circles;
}

std::vector<cv::Vec4f> DetectorPolygons::detectTriangle(const cv::Mat& frame)
{
    cv::Mat templateTriangle = cv::imread("t.png", cv::IMREAD_GRAYSCALE);
    cv::Mat grayImage;
    cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
    std::vector<cv::Vec4f> positionBallard;
    //  template width and height
    int w = templateTriangle.cols;
    int h = templateTriangle.rows;
    //  create ballard and set options
    cv::Ptr<cv::GeneralizedHoughBallard> ballard = cv::createGeneralizedHoughBallard();
    ballard->setMinDist(10);
    ballard->setLevels(360);
    ballard->setDp(2);
    ballard->setMaxBufferSize(1000);
    ballard->setVotesThreshold(40);
    ballard->setCannyLowThresh(30);
    ballard->setCannyHighThresh(110);
    ballard->setTemplate(templateTriangle);
    ballard->detect(grayImage, positionBallard);
    return positionBallard;
}