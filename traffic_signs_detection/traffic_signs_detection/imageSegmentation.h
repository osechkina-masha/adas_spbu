#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class ImageSegmentation
{
public:

    // Red color segmentation
    static cv::Mat highlightRed(const cv::Mat &originalImage);

    // Yellow color segmentation
    static cv::Mat highlightYellow(const cv::Mat &originalImage);

    // Blue color segmentation
    static cv::Mat highlightBlue(const cv::Mat &originalImage);

    // White color segmentation
    static cv::Mat highlightWhite(const cv::Mat &originalImage);
};
