cmake_minimum_required(VERSION 3.10)
project(PointGridVoxelizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O2 -march=native")

# Аналогичное вашей цели
add_executable(voxelize
    src/main.cpp
    src/utils/Voxelizer.cpp
)

# Новая цель, которая читает .npz и пишет output.ply
add_executable(voxelize_bin
    src/main_bin.cpp
    src/utils/Voxelizer.cpp
)
target_include_directories(voxelize_bin PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
