# CMakeList.txt : CMake project for cifar10_infer, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("cifar10_infer")

# Add source to this project's executable.
add_executable (cifar10_infer "cifar10_infer.cpp")

include_directories("include" "/usr/include/opencv4")

#add_library(pqvm SHARED IMPORTED) 

target_link_directories(cifar10_infer PUBLIC "/usr/local/lib")
target_link_libraries(cifar10_infer opencv_core opencv_imgcodecs pqvm)

# TODO: Add tests and install targets if needed.
