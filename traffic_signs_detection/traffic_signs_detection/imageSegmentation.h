#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class imageSegmentation
{
public:

    // Red color segmentation
    static cv::Mat highlightRed(cv::Mat originalImage);

    // Yellow color segmentation
    static cv::Mat highlightYellow(cv::Mat originalImage);

    // Blue color segmentation
    static cv::Mat highlightBlue(cv::Mat originalImage);

    // White color segmentation
    static cv::Mat highlightWhite(cv::Mat originalImage);
};
