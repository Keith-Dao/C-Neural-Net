project(nn VERSION 1.0)

add_library(${PROJECT_NAME} OBJECT 
    metrics.cpp
    utils/string.cpp
    utils/cli.cpp
    utils/matrix.cpp
    utils/path.cpp
    utils/indicators.cpp
    utils/math.cpp
    utils/image.cpp
    linear.cpp
    exceptions/eigen.cpp
    exceptions/json.cpp
    exceptions/load.cpp
    exceptions/differentiable.cpp
    exceptions/metrics.cpp
    exceptions/utils.cpp
    exceptions/activation_functions.cpp
    exceptions/loss.cpp
    exceptions/image_loader.cpp
    exceptions/model.cpp
    activation_functions.cpp
    image_loader.cpp
    cross_entropy_loss.cpp
    model.cpp
    linear.hpp
    activation_functions.hpp
    cross_entropy_loss.hpp
    utils/matrix.hpp
    utils/exceptions.hpp
    utils/indicator.hpp
    utils/string.hpp
    utils/cli.hpp
    utils/math.hpp
    utils/path.hpp
    utils/image.hpp
    metrics.hpp
    exceptions/activation_functions.hpp
    exceptions/utils.hpp
    exceptions/differentiable.hpp
    exceptions/loss.hpp
    exceptions/metrics.hpp
    exceptions/eigen.hpp
    exceptions/load.hpp
    exceptions/image_loader.hpp
    exceptions/json.hpp
    exceptions/model.hpp
    image_loader.hpp
    model.hpp
)

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

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Tabulate
find_package(tabulate REQUIRED)
target_link_libraries(${PROJECT_NAME} ${TABULATE_LIBRARIES})

# Termcolor
find_package(termcolor REQUIRED)
target_link_libraries(${PROJECT_NAME} ${TERMCOLOR_LIBRARIES})

# GNU Readline
find_path(READLINE_HEADER_PATH readline)
include_directories(${READLINE_HEADER_PATH}/readline)
find_library(READLINE_LIB_PATH readline)
link_directories(${READLINE_LIB_PATH})
target_link_libraries(${PROJECT_NAME} readline)

# ncurses
set(CURSES_NEED_NCURSES TRUE)
find_package(Curses REQUIRED)
include_directories(${CURSES_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${CURSES_LIBRARIES})
