cmake_minimum_required(VERSION 3.10)
project(PointGrid LANGUAGES CXX)

# --- Настройки компилятора ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Оптимизации (можно убрать -march=native, если портируете на другие машины)
add_compile_options(-O2 -march=native)

# --- Общие include-пути ---
include_directories(${CMAKE_SOURCE_DIR}/src)

# --- Voxelizer: .txt → .ply ---
add_executable(voxelize
    src/main.cpp
    src/utils/Voxelizer.cpp
)

# --- Voxelizer из .npz → .ply (если нужен) ---
add_executable(voxelize_bin
    src/main_bin.cpp
    src/utils/Voxelizer.cpp
)

add_executable(preprocess
    src/utils/Preprocessor.cpp
    src/utils/Voxelizer.cpp
)
target_include_directories(preprocess PRIVATE ${CMAKE_SOURCE_DIR}/src)

# --- Статическая библиотека с 3D-CNN (Tensor3D + все слои + оптимизатор) ---
add_library(pointgrid_network STATIC
    src/net/Tensor3D.cpp
    src/network/Network.cpp
    src/layers/Conv3D.cpp
    src/layers/BatchNorm3D.cpp
    src/layers/ReLU3D.cpp
    src/layers/MaxPool3D.cpp
    src/layers/FullyConnected.cpp
    src/layers/SoftmaxCrossEntropy.cpp
    src/optim/SGD.cpp
)

# Если в дальнейшем понадобятся C++-утилиты для train/infer/test,
# просто создайте соответствующие .cpp и линковку с этой библиотекой:
#
# add_executable(my_train src/train.cpp)
# target_link_libraries(my_train PRIVATE pointgrid_network)
#
# и т.п.

# --- Для удобства: единый include для всех таргетов ---
foreach(tgt IN ITEMS voxelize voxelize_bin pointgrid_network)
    target_include_directories(${tgt} PRIVATE ${CMAKE_SOURCE_DIR}/src)
endforeach()
