# Kokkos minimally requires 3.16 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.16)

# Project can mix languages - must have C++ support
# Kokkos flags are only applied to C++ files
project(CNN_Kokkos CXX)

# We build kokkos as a subdirectory of our project
add_subdirectory(/uufs/chpc.utah.edu/common/home/u1193107/cs6235/kokkos kokkos)
#find_package(Kokkos REQUIRED)

add_executable(cnn_kokkos cnn_kokkos.cpp)

# This is the only line required to set up all compiler/linker flags
target_link_libraries(cnn_kokkos Kokkos::kokkos)

# Adds a test for the executable
enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND example 10)
