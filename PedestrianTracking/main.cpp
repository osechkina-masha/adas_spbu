//#include <opencv2/imgcodecs.hpp>
#include "trackers/MyTracker.h"

//#include "UI/mainwindow.h"
//#include <iostream>
//#include <QApplication>
//
//using namespace cv;
//using namespace std;
//
//int main(int argc, char *argv[]) {
//    QApplication a(argc, argv);
//    MainWindow w;
//
//    if (argc != 2) {
//        cout << "Incorrect args" << endl;
//        return 0;
//    }
//    w.show();
//    if (w.setPathToVideo(argv[1])) {
//        w.showFirstImage();
//    } else {
//        cerr << "Unable to open file!" << endl;
//    }
//    return a.exec();
//}
int main() {
    MyTracker tr = MyTracker();
//    cv::Mat m = cv::imread("img.png", cv::IMREAD_COLOR);
//    cv::Rect_<double> ped = cv::Rect2d({10.0, 10.0}, cv::Size(20, 20));
//    tr.init(m, ped);
//    tr.update(m);
    return 0;
}