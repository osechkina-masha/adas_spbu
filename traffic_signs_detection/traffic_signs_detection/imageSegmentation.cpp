#include "imageSegmentation.h"
#include <iostream>

cv::Mat ImageSegmentation::highlightColor(const cv::Mat& originalImage, cv::Scalar lowerBoundary, cv::Scalar upperBoundary)
{
    cv::Mat colorImage;
    cv::Mat imageHSV;
    cv::cvtColor(originalImage, imageHSV, cv::COLOR_BGR2HSV);
    cv::inRange(imageHSV, lowerBoundary, upperBoundary, colorImage);
    cv::medianBlur(colorImage, colorImage, 3);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
        cv::Size(3, 3),
        cv::Point(1, 1));
    cv::Mat element1 = cv::getStructuringElement(cv::MORPH_ELLIPSE,
        cv::Size(7, 7),
        cv::Point(3, 3));
    erode(colorImage, colorImage, element);
    dilate(colorImage, colorImage, element1);
    cv::Mat newImage = cv::Mat::zeros(originalImage.size(), CV_8UC3);
    cv::bitwise_and(originalImage, originalImage, newImage, colorImage);
    cv::imshow("f3", newImage);
        cv::waitKey(0);
    return newImage;
}
cv::Mat ImageSegmentation::highlightRed(const cv::Mat &originalImage)
{
    cv::Mat imageHSV;
    cv::Mat redImage;
    cv::cvtColor(originalImage, imageHSV, cv::COLOR_BGR2HSV);
    cv::Mat redImage1, redImage2;
    cv::inRange(imageHSV, cv::Scalar(0, 100, 20), cv::Scalar(10, 255, 255), redImage1);
    cv::inRange(imageHSV, cv::Scalar(160, 100, 20), cv::Scalar(179, 255, 255), redImage2);
    redImage = redImage1 + redImage2;
    cv::medianBlur(redImage, redImage, 3);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
        cv::Size(3, 3),
        cv::Point(1, 1));
    cv::Mat element1 = cv::getStructuringElement(cv::MORPH_ELLIPSE,
        cv::Size(7, 7),
        cv::Point(3, 3));
    erode(redImage, redImage, element);
    dilate(redImage, redImage, element1);
    cv::Mat final = cv::Mat::zeros(originalImage.size(), CV_8UC3);
    cv::bitwise_and(originalImage, originalImage, final, redImage);
    cv::imshow("f3", final);
    cv::waitKey(0);
    return final;
}
