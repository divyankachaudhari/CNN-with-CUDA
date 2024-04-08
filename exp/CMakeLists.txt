cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(lenet_cpp)

find_package(Torch REQUIRED)

add_executable(lenet_cpp cnn.cpp)
target_link_libraries(lenet_cpp "${TORCH_LIBRARIES}")
set_property(TARGET lenet_cpp PROPERTY CXX_STANDARD 14)
