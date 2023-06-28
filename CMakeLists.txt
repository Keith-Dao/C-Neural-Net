cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_testing()

# Project
project(NeuralNetwork VERSION 1.0)

# Files
add_executable(${PROJECT_NAME} main.cpp)

# Source files
add_subdirectory(src)
target_include_directories(${PROJECT_NAME} PRIVATE src)
target_link_libraries(${PROJECT_NAME} src)

# Testing
add_subdirectory(test)