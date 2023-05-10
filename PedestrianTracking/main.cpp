#include "trackers/MyTracker.h"
#include <cpprest/http_listener.h>
#include <cpprest/json.h>

using namespace web;
using namespace web::http;
using namespace web::http::experimental::listener;
using namespace utility::conversions;

#include "base64.h"
#include <iostream>
#include <set>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

#define TRACE(msg)            wcout << msg

MyTracker tracker = MyTracker();


cv::Mat GetImageFromMemory(uchar *image, int length, int flag) {
    cv::Mat1b data(1, length, image);
    return imdecode(data, flag);
}


void handle_get(http_request request) {
    std::cout << "get" << std::endl;

    TRACE(L"\nhandle GET\n");

    request.extract_json().then([request](web::json::value body) {

        auto inputStream = to_utf8string(body.at(("image")).as_string());
        auto imageVector = base64_decode(inputStream);
        cv::Mat m = GetImageFromMemory(imageVector.data(), imageVector.size(), 1);
        cv::cvtColor(m, m, cv::COLOR_BGRA2RGB);

        cv::Rect2d ped = tracker.update(m);
        json::value obj;

        obj[U("x")] = U(int(ped.x));
        obj[U("y")] = U(int(ped.y));
        obj[U("height")] = U(int(ped.size().height));
        obj[U("width")] = U(int(ped.size().width));

        request.reply(status_codes::OK, obj);
    });
}


void handle_post(http_request request) {
    std::cout << "post" << std::endl;
    TRACE("\nhandle POST\n");
    request.extract_json().then([request](web::json::value body) {

        auto inputStream = to_utf8string(body.at(("image")).as_string());
        auto imageVector = base64_decode(inputStream);
        cv::Mat m = GetImageFromMemory(imageVector.data(), imageVector.size(), 1);
        cv::cvtColor(m, m, cv::COLOR_BGRA2RGB);

        double x = body.at(U("x")).as_double();
        double y = body.at(U("y")).as_double();
        double x1 = body.at(U("x1")).as_double();
        double y1 = body.at(U("y1")).as_double();
        cv::Rect_<double> ped = cv::Rect2d({x, y}, cv::Size(int(x1 - x), int(y1 - y)));
        tracker.init(m, ped);
        string s = "{ }";

        json::value answer = json::value::parse(U(s));
        request.reply(status_codes::OK, answer);
    });
}


void test_tracker() {
    cv::Rect_<double> ped = cv::Rect2d({344, 300}, cv::Size(82, 72));
    auto frame1 = cv::imread("frame1.jpg", 1);
    std::cout << frame1.channels() << std::endl;
    auto frame2 = cv::imread("frame2.jpg", 1);
    tracker.init(frame1, ped);
    tracker.update(frame2);
}

int main() {
    std::cout << "Started main" << std::endl;

//    test_tracker();

    http_listener listener("http://0.0.0.0:8083");
    listener.support(methods::POST, handle_post);
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