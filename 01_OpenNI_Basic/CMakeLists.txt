cmake_minimum_required(VERSION 2.8)

set( CMAKE_BUILD_TYPE release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

project(OpenNI_Basic)

set(OPEN_NI_ROOT "/home/liubin/OpenNI2" CACHE FILEPATH "Root directory of OpenNI2")

find_path(OpenNI_INCLUDE_DIR OpenNI.h PATH "${OPEN_NI_ROOT}/Include")

find_library(OpenNI_LIBRARY OpenNI2 PATH "${OPEN_NI_ROOT}/Bin/x64-Release/")

find_package(OpenCV REQUIRED)

include_directories(${OpenNI_INCLUDE_DIR})

add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OpenNI_LIBRARY})
