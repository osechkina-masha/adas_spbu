cmake_minimum_required(VERSION 3.23)
project(VisibilityEstimator)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(VisibilityEstimator
        src/main.cpp
        src/contourDetector.cpp
        src/contourDetector.h
        src/contourDrawer.cpp
        src/contourDrawer.h
        src/contourMerger.cpp
        src/contourMerger.h
        src/contourOrdering.cpp
        src/contourOrdering.h
        src/distanceEstimator.cpp
        src/distanceEstimator.h
        src/edgeDetector.cpp
        src/edgeDetector.h
        src/imageFilter.cpp
        src/imageFilter.h
        src/inversePerspectiveMapping.cpp
        src/inversePerspectiveMapping.h)

set(OpenCV_LIBS
        opencv_highgui
        opencv_imgproc
        opencv_imgcodecs)

target_link_libraries(VisibilityEstimator ${OpenCV_LIBS})