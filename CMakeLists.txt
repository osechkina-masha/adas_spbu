cmake_minimum_required(VERSION 2.8)

project(adas_spbu)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(PedestrianTracking)