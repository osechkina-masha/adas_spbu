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

        //cv::resize(warpedImage, warpedImage, cv::Size(frame.cols, frame.rows + 300));

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
        //for (int i = 0; i < 3; i++) {
        //    for (int j = 0; j < 3; j++) {
        //        auto g = matrix.at<float>(i, j);
        //    }
        //}

        //return getMatrixWithoutSecondColumn(matrix);
        return matrix;
        //cv::Mat rotationMatrix = (cv::Mat_<float>(3, 3) <<
        //    1, 0, 0,
        //    0, -sin(pitch), -cos(pitch),
        //    0, cos(pitch), -sin(pitch));

        //cv::Mat translationMatrix = (cv::Mat_<float>(3, 1) <<
        //    3.5f / 2.0f,
        //    0,
        //    1);

        //cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) <<
        //    focalLength, 0, size.width / 2,
        //    0, focalLength, size.height / 2,
        //    0, 0, 1);

        //cv::Mat transformationMatrix3D = (cv::Mat_<float>(3, 3) <<
        //    1, 0, -3.5f / 2.0f,
        //    0, -sin(pitch), cos(pitch) * cameraHeight,
        //    0, cos(pitch), sin(pitch) * cameraHeight);

        //cv::Mat scaleMatrix = (cv::Mat_<float>(3, 3) << )
        //cv::vconcat(rotationMatrix, translationMatrix, transformationMatrix3D);

        //cv::Mat translationMatrix = (cv::Mat_<float>(3, 3) <<
        //    0, 0, 0,
        //    0, 0, 0,
        //    0, 0, -cameraHeight / sin(pitch));

        //return cameraMatrix * transformationMatrix3D;
    }

    cv::Mat InversePerspectiveMapping::getMatrixWithoutSecondColumn(cv::Mat matrix)
    {
        return (cv::Mat_<float>(3, 3) <<
            matrix.at<float>(0, 0), matrix.at<float>(0, 2), matrix.at<float>(0, 3),
            matrix.at<float>(1, 0), matrix.at<float>(1, 2), matrix.at<float>(1, 3),
            matrix.at<float>(2, 0), matrix.at<float>(2, 2), matrix.at<float>(2, 3));
    }
}