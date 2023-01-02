#include "imageFilter.h"

namespace edge_detector
{
    cv::Mat ImageFilter::applyCannyOperator(const cv::Mat& frame)
    {
        cv::Mat grayImage, blurGrayImage, resultImage;

        cv::GaussianBlur(frame, blurGrayImage, cv::Size(5, 5), 1.4, 1.4, 1);
        cv::Canny(blurGrayImage, resultImage, 20, 30);

        grayImage.release();
        blurGrayImage.release();

        return resultImage;
    }

    cv::Mat ImageFilter::selectMinimumForEachPixel(std::deque<cv::Mat> images)
    {
        cv::Mat resultImage = cv::Mat::zeros(images[0].size(), CV_8UC1);

        for (int row = 0; row < images[0].rows; row++)
        {
            for (int col = 0; col < images[0].cols; col++)
            {
                resultImage.at<uchar>(row, col) = getMinimumPixel(images, row, col);
            }
        }

        return resultImage;
    }

    int ImageFilter::getMinimumPixel(std::deque<cv::Mat> images, int row, int col)
    {
        uchar value = images[0].at<uchar>(row, col);

        for (int imageNumber = 1; imageNumber < images.size(); imageNumber++)
        {
            uchar newValue = images[imageNumber].at<uchar>(row, col);

            if (newValue < value)
            {
                value = newValue;
            }
        }

        return value;
    }
}