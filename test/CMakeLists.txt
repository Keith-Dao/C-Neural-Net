# Source files
include_directories(${CMAKE_SOURCE_DIR}/src)

# Gtest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Eigen
find_package(Eigen3 REQUIRED NO_MODULE)

# JSON
find_package(nlohmann_json 3.2.0 REQUIRED)

# Tests
file(GLOB_RECURSE TEST_FILES test_*.cpp *.hpp)
add_executable(tests ${TEST_FILES})
target_link_libraries(
    tests 
    src
    GTest::gtest_main
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

include(GoogleTest)
gtest_discover_tests(tests)
