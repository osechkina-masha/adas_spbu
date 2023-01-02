#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "contourDetector.h"
#include "imageFilter.h"
#include "contourMerger.h"

namespace edge_detector
{
    std::vector<std::vector<cv::Point>> ContourDetector::detectContours(const cv::Mat& image)
    {
        std::vector<std::vector<cv::Point>> contours;

        cv::Mat cannyImage = ImageFilter::applyCannyOperator(image);
        findContours(cannyImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

        if (contours.empty())
        {
            return contours;
        }

        auto mergedContours = ContourMerger::connectContours(contours, image.cols, image.rows);

        cannyImage.release();
        contours.clear();

        return mergedContours;
    }
}