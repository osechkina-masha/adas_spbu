#include <iostream>
#include "detectorPolygons.h"

double DetectorPolygons::angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1 * dx2 + dy1 * dy2)/sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

std::vector<cv::Rect> DetectorPolygons::detectShape(const cv::Mat &frame)
{
    cv::Mat grayImage;
    cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
	cv::Canny(grayImage, edges, 0, 50, 3);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::imshow("3", edges);
	std::vector<cv::Point> approx;
    
    std::vector<cv::Rect> rectangles;
	for (auto &contour : contours)
	{
		// Approximate contour with accuracy proportional
		// to the contour perimeter
		cv::approxPolyDP(cv::Mat(contour), approx, cv::arcLength(cv::Mat(contour), true) * 0.02, true);

		// Skip small or non-convex objects 
		if (std::fabs(cv::contourArea(contour)) < 100 || !cv::isContourConvex(approx))
			continue;

        cv::Rect rectangle = cv::boundingRect(contour);

		if (approx.size() == 3)
		{   
            rectangles.push_back(rectangle);
			//setLabel(dst, "TRI", contour);    // Triangles
		}
		else if (approx.size() >= 4 && approx.size() <= 6)
		{
			// Number of vertices of polygonal curve
			int vtc = approx.size();

			// Get the cosines of all corners
			std::vector<double> cos;
			for (int j = 2; j < vtc + 1; j++)
				cos.push_back(angle(approx[j%vtc], approx[j - 2], approx[j - 1]));

			// Sort ascending the cosine values
			std::sort(cos.begin(), cos.end());

			// Get the lowest and the highest cosine
			double mincos = cos.front();
			double maxcos = cos.back();

			// Use the degrees obtained above and the number of vertices
			// to determine the shape of the contour
			if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
				//setLabel(dst, "RECT", contours[i]);
                rectangles.push_back(rectangle);
			else if (vtc == 6 && mincos >= -0.55 && maxcos <= -0.45)
				//setLabel(dst, "HEXA", contours[i]);
                rectangles.push_back(rectangle);
		}
		else
		{
			// Detect and label circles
			double area = cv::contourArea(contour);
			int radius = rectangle.width / 2;

			if (std::abs(1 - ((double)rectangle.width / rectangle.height)) <= 0.2 &&
			    std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
				//setLabel(dst, "CIR", contours[i]);
                rectangles.push_back(rectangle);
		}
	}
    return rectangles;
}

