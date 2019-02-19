cmake_minimum_required(VERSION 2.8)

set( CMAKE_BUILD_TYPE release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

project(Skeleton_Tracking)

set(OPEN_NI_ROOT "/home/liubin/OpenNI2" CACHE FILEPATH "Root directory of OpenNI2")
set(NITE_ROOT "/home/liubin/NiTE-Linux-x64-2.2" CACHE FILEPATH "Root directory of NITE2")

find_path(OpenNI_INCLUDE_DIR OpenNI.h PATH "${OPEN_NI_ROOT}/Include")
find_library(OpenNI_LIBRARY OpenNI2 PATH "${OPEN_NI_ROOT}/Bin/x64-Release/")

find_path(NITE_INCLUDE_DIR NiTE.h PATH "${NITE_ROOT}/Include")
find_library(NITE_LIBRARY NiTE2 PATH "${NITE_ROOT}/Samples/Bin/x64-Release/")

find_package(OpenCV REQUIRED)
#find_package(PCL 1.7 REQUIRED)
 
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

include_directories(${OpenNI_INCLUDE_DIR})
include_directories(${NITE_INCLUDE_DIR})

add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OpenNI_LIBRARY} ${NITE_LIBRARY})
