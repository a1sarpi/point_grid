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

# ----- Тест Tensor3D + Conv3D -----
add_executable(test_tensor_conv
    tests/test_tensor_conv.cpp
    src/net/Tensor3D.cpp
    src/layers/Conv3D.cpp
)

# Пусть компилятор ищет заголовки сразу в src/
target_include_directories(test_tensor_conv PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)


# ----- Тест для MaxPool3D ----- 
add_executable(test_maxpool3d
    tests/test_maxpool3d.cpp
    src/net/Tensor3D.cpp
    src/layers/MaxPool3D.cpp
)
target_include_directories(test_maxpool3d PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# ----- Тест для ReLU3D -----
add_executable(test_relu3d
    tests/test_relu3d.cpp
    src/net/Tensor3D.cpp
    src/layers/ReLU3D.cpp
)
target_include_directories(test_relu3d PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# ----- Тест для BatchNorm3D -----
add_executable(test_batchnorm3d
    tests/test_batchnorm3d.cpp
    src/net/Tensor3D.cpp
    src/layers/BatchNorm3D.cpp
)
target_include_directories(test_batchnorm3d PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# ----- Тест для FullyConnected -----
add_executable(test_fullyconnected
    tests/test_fullyconnected.cpp
    src/layers/FullyConnected.cpp
)
target_include_directories(test_fullyconnected PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# ----- Тест для SoftmaxCrossEntropy -----
add_executable(test_sm_ce
    tests/test_sm_ce.cpp
    src/layers/SoftmaxCrossEntropy.cpp
)
target_include_directories(test_sm_ce PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

