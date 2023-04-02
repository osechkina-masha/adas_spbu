#pragma once

#include "opencv2/imgproc.hpp"

#include <deque>

namespace edge_detector
{
    class ImageFilter
    {
    public:
        /**
         * @brief Applies canny operator with set parameters.
         * @param frame is the frame to which Canny operator should be applied.
         * @return Result of applying Canny operator to the given frame.
        */
        static cv::Mat applyCannyOperator(const cv::Mat& frame);

        /**
         * @brief Makes an image with a minimum value between multiple frames for each pixel.
         * @param images is the deque of images to which algorithm should be applied.
         * @return Image with a minimum value between multiple frames for each pixel.
        */
        static cv::Mat selectMinimumForEachPixel(const std::deque<cv::Mat>& images);

    private:
        static int getMinimumPixel(std::deque<cv::Mat> images, int row, int col);
    };
}