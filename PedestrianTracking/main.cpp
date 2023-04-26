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
        cv::Mat m = GetImageFromMemory(imageVector.data(), imageVector.size(), 0);
        cv::Rect2d ped = tracker.update(m);
        json::value obj;
        obj[U("x")] = U(ped.x);
        obj[U("y")] = U(ped.y);
        obj[U("height")] = U(ped.size().height);
        obj[U("width")] = U(ped.size().width);
        request.reply(status_codes::OK, obj);
    });
}


void handle_post(http_request request) {
    std::cout << "post" << std::endl;
    TRACE("\nhandle POST\n");
    request.extract_json().then([request](web::json::value body) {

        auto inputStream = to_utf8string(body.at(("image")).as_string());
        auto imageVector = base64_decode(inputStream);
        cv::Mat m = GetImageFromMemory(imageVector.data(), imageVector.size(), 0);
        double x = body.at(U("x")).as_double();
        double y = body.at(U("y")).as_double();
        int height = body.at(U("height")).as_integer();
        int width = body.at(U("width")).as_integer();
        cv::Rect_<double> ped = cv::Rect2d({x, y}, cv::Size(width, height));
        tracker.init(m, ped);
        string s = "{ \"length\" : " + to_string(m.rows) + "}";
        json::value answer = json::value::parse(U(s));
        request.reply(status_codes::OK, answer);
    });
}


int main() {
    std::cout << "Started main" << std::endl;

    http_listener listener("http://0.0.0.0:2390");
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