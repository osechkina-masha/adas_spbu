cmake_minimum_required (VERSION 3.8)

project(traffic_signs_detection)

find_package( OpenCV REQUIRED )
find_package(cpprestsdk REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable (traffic_signs_detection "main.cpp" "imageSegmentation.h" "imageSegmentation.cpp" "detectorPolygons.cpp" "detectorPolygons.h" "trafficSign.h" "trafficSign.cpp" "base64.cpp" "base64.h")

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} PRIVATE cpprestsdk::cpprest)
