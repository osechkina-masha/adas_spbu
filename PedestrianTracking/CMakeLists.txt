cmake_minimum_required(VERSION 3.5)

project(Tracking LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#comment 2 lines below if you don't want to use legacy
add_compile_definitions(USE_LEGACY)
set(USE_LEGACY_CMAKE true)

SET(SOURCE_FILES main.cpp
        trackers/MyTracker.cpp
        trackers/MyTracker.h
        trackers/Tracker.cpp
        trackers/Tracker.h
        trackers/OpticalFlowTracker.cpp
        trackers/OpticalFlowTracker.h
        trackers/KCFTracker.cpp
        trackers/KCFTracker.h
        trackers/CSRTTracker.cpp
        trackers/CSRTTracker.h
        trackers/GOTURNTracker.cpp
        trackers/GOTURNTracker.h
        trackers/consts.h trackers/constsForOpticalFlow.h trackers/TrackerPattern.cpp trackers/TrackerPattern.h)

SET(LEGACY_SOURCE_FILES trackers/TLDTracker.cpp trackers/TLDTracker.h)

if (USE_LEGACY_CMAKE)
    add_executable(Tracking
            ${SOURCE_FILES}
            ${LEGACY_SOURCE_FILES})
else ()
    add_executable(Tracking
            ${SOURCE_FILES})
endif ()


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(Tracking PRIVATE ${OpenCV_LIBS})

add_library(TrackingLib SHARED ${SOURCE_FILES} ${LEGACY_SOURCE_FILES})
target_link_libraries(TrackingLib PRIVATE ${OpenCV_LIBS})
