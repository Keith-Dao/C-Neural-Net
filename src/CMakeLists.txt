project(src VERSION 1.0)

file(GLOB_RECURSE SRC_FILES *.cpp)
file(GLOB_RECURSE HEADER_FILES *.hpp)
add_library(${PROJECT_NAME} STATIC ${SRC_FILES} ${HEADER_FILES})

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