cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Project
project(NerualNetwork VERSION 1.0)

# Files
add_executable(${PROJECT_NAME} main.cpp)

# Source files
add_subdirectory(src)
target_include_directories(${PROJECT_NAME} PRIVATE src)
target_link_libraries(${PROJECT_NAME} src)

# Eigen
find_package(Eigen3 REQUIRED NO_MODULE)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

# JSON
find_package(nlohmann_json 3.2.0 REQUIRED)
target_link_libraries(${PROJECT_NAME} nlohmann_json::nlohmann_json)

# YAML
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES})

# Matplot++
find_package(Matplot++ REQUIRED)
target_link_libraries(${PROJECT_NAME} Matplot++::matplot)

# Indicators
find_package(indicators REQUIRED)
target_link_libraries(${PROJECT_NAME} ${INDICATORS_LIBRARIES})