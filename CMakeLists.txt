cmake_minimum_required (VERSION 3.8)

project(adas_spbu)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory(traffic_signs_detection/traffic_signs_detection )
add_executable (${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
