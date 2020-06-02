cmake_minimum_required(VERSION 3.15)
project(Depth_Sorting_final)

set(OpenCV_DIR D:/opencv/cmake-build)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(Depth_Sorting_final main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Depth_Sorting_final ${OpenCV_LIBS})