#include <filesystem>

#include "inversePerspectiveMapping.h"

namespace distance_estimator
{
    cv::Mat InversePerspectiveMapping::inversePerspectiveMap(const cv::Mat& frame, cv::Point vanishingPoint, cv::Point farthestVisiblePoint)
    {
        cv::Mat warpedImage = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), frame.type());
        cv::Rect croppedRectangle = cv::Rect(0, farthestVisiblePoint.y, frame.cols, frame.rows - farthestVisiblePoint.y);
        cv::Mat croppedImage = frame(croppedRectangle);

        const float pitch = -atan(((float)vanishingPoint.y - (float)sizeY) / (float)focalLength);

        cv::Mat projectiveTransformationMatrix = getProjectiveTransfromationMatrix(croppedImage.size(), pitch);
        cv::warpPerspective(croppedImage, warpedImage, projectiveTransformationMatrix,
            cv::Size(warpedImage.cols, warpedImage.rows));

        return warpedImage;
    }

    cv::Mat InversePerspectiveMapping::getProjectiveTransfromationMatrix(cv::Size size, float pitch) const
    {
        cv::Mat transformationMatrix = (cv::Mat_<float>(3, 4) <<
            1, 0, 0, 0,
            0, cos(pitch), -sin(pitch), cameraHeight,
            0, sin(pitch), cos(pitch), 0);

        cv::Mat instrinsicMatrix = (cv::Mat_<float>(3, 3) <<
            -focalLength, 0, 0,
            0, -focalLength, 0,
            0, 0, 1);

        cv::Mat matrix = instrinsicMatrix * transformationMatrix;

        return matrix;
    }

    cv::Mat InversePerspectiveMapping::getMatrixWithoutSecondColumn(cv::Mat matrix)
    {
        return (cv::Mat_<float>(3, 3) <<
            matrix.at<float>(0, 0), matrix.at<float>(0, 2), matrix.at<float>(0, 3),
            matrix.at<float>(1, 0), matrix.at<float>(1, 2), matrix.at<float>(1, 3),
            matrix.at<float>(2, 0), matrix.at<float>(2, 2), matrix.at<float>(2, 3));
    }
}