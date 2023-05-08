#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class ImageSegmentation
{
public:

    static cv::Mat highlightColor(const cv::Mat &originalImage, cv::Scalar lowerThreshold, cv::Scalar upperThreshold);

    // Red color segmentation
    static cv::Mat highlightRed(const cv::Mat &originalImage);
};
