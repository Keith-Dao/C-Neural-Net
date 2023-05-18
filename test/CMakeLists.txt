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

# Tests
file(GLOB_RECURSE TEST_FILES test_*.cpp)
add_executable(tests ${TEST_FILES})
target_link_libraries(
    tests 
    src
    GTest::gtest_main
    Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(tests)
