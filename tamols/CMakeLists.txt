cmake_minimum_required(VERSION 3.16)
project(tamols)

# Find Drake
find_package(drake CONFIG REQUIRED)

# C++ configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Create core library
add_library(tamols
    src/tamols_state.cc
    src/constraints.cc
    src/helpers.cc
)

# Setup include directories
target_include_directories(tamols PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link Drake
target_link_libraries(tamols PUBLIC drake::drake)

# Add examples
add_executable(simple_example examples/simple_example.cc)
target_link_libraries(simple_example PRIVATE tamols)

# Add tests
enable_testing()
add_executable(tamols_test tests/tamols_test.cc)
target_link_libraries(tamols_test PRIVATE tamols)
add_test(NAME tamols_test COMMAND tamols_test)