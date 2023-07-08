cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-O2)

# Project
project(NeuralNetwork VERSION 1.0)

# Files
add_executable(${PROJECT_NAME} main.cpp)

# Source files
add_subdirectory(src)
target_link_libraries(${PROJECT_NAME} nn)

# Testing
SET(BUILD_TESTS OFF CACHE BOOL "Include tests during build")
if (${BUILD_TESTS}) 
    enable_testing()
    add_subdirectory(test)
endif()

# Include what you use
SET(USE_IWYU OFF CACHE BOOL "Use include-what-you-use")
if (${USE_IWYU})
    message("Building with incude-what-you-use.")
    find_program(iwyu_path NAMES include-what-you-use iwyu REQUIRED)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path})
endif()
