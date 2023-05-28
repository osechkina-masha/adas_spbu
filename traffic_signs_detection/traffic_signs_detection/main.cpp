//#include "imageSegmentation.h"
//#include "detectorPolygons.h"
#include "trafficSign.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cpprest/http_listener.h>
#include <cpprest/json.h>
#include <iostream>
#include "base64.h"

using namespace web;
using namespace web::http;
using namespace web::http::experimental::listener;
using namespace utility::conversions;
using namespace std;


#define TRACE(msg)            wcout << msg

DetectorTrafficSign detector = DetectorTrafficSign();

cv::Mat GetImageFromMemory(uchar *image, int length, int flag) {
    cv::Mat1b data(1, length, image);
    return imdecode(data, flag);
}

void handle_get(http_request request) {
    std::cout << "get" << std::endl;

    TRACE(L"\nhandle GET\n");

    request.extract_json().then([request](web::json::value body) {
        
        std::cout << "get" << std::endl;
        auto inputStream = to_utf8string(body.at(("image")).as_string());
        auto imageVector = base64_decode(inputStream);
        cv::Mat m = GetImageFromMemory(imageVector.data(), imageVector.size(), 1);
        cv::imshow("f3", m);
        cv::waitKey(0);
        json::value obj;
        std::vector<json::value> result;
        std::vector <std::vector<cv::Rect>> rectangles = detector.detectTrafficSigns(m);
        auto number = 0;
        for (auto &rect : rectangles)
    {
        for (auto &r : rect)
        {
            cv::rectangle(m, r, cv::Scalar(5, 6, 7), 3);
        }
    }
    cv::imshow("f3", m);
    cv::waitKey(0);
        for (auto &vectorRectangle : rectangles)
        {
            for (auto &rectangle : vectorRectangle)
            {   
                obj[number]["height"]= rectangle.height;
                obj[number]["x"] = rectangle.x;
                obj[number]["y"]= rectangle.y;
                obj[number]["width"]= rectangle.width;
                number++;
            }
        }
        
        request.reply(status_codes::OK, obj);
    });
}

int main(int argc, char* argv[])
{
   /*if (argc != 2)
    {
       return 1;
    }
    cv::VideoCapture video(argv[1]);
    if (!video.isOpened())
    {
        return 1;
    }
    while (video.isOpened())
    {
        cv::Mat frame;
        video >> frame;
      //  cv::Mat frame = cv::imread("2.jpg");
        cv::Mat red = ImageSegmentation::highlightRed(frame);
        cv::Mat white = ImageSegmentation::highlightColor(frame, cv::Scalar(0, 0, 150), cv::Scalar(360, 60, 255));
        cv::Mat blue = ImageSegmentation::highlightColor(frame, cv::Scalar(90, 50, 70), cv::Scalar(128, 255, 255));
        cv::Mat yellow = ImageSegmentation::highlightColor(frame, cv::Scalar(20, 100, 100), cv::Scalar(30, 255, 255));
        TrafficSign::showTrafficSigns(red, frame);
        TrafficSign::showTrafficSigns(white, frame);
        TrafficSign::showTrafficSigns(blue, frame);
        TrafficSign::showTrafficSigns(yellow, frame);
        cv::imshow("red", red);
        cv::imshow("white", white);
        cv::imshow("blue", blue);
        cv::imshow("yellow", yellow);
        cv::imshow("detected traffic signs", frame);
        cv::waitKey(0);
    }
    */
    std::cout << "Started main" << std::endl;
    http_listener listener("http://0.0.0.0:8083");
    listener.support(methods::GET, handle_get);
    try {
        listener
                .open()
                .then([&listener]() { TRACE(L"\nstarting to listen\n"); })
                .wait();

        while (true);
    }
    catch (exception const &e) {
        wcout << e.what() << endl;
    }

    return 0;
}