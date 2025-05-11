#include "DataLoader.h"
#include <dirent.h>       // для обхода директорий
#include <fstream>
#include <sstream>
#include <algorithm>
#include <array>          // для std::array
#include <stdexcept>
#include <cstring>        // для strcmp

DataLoader::DataLoader(const std::string& data_dir,
                       int batch_size,
                       float val_ratio)
    : batch_size_(batch_size),
      val_ratio_(val_ratio)
{
    loadFileLists(data_dir);
    if (voxel_files_.empty())
        throw std::runtime_error("DataLoader: нет PLY-файлов в " + data_dir);
    split_index_ = static_cast<size_t>(voxel_files_.size() * (1.0f - val_ratio_));
}

void DataLoader::loadFileLists(const std::string& data_dir) {
    DIR* dir = opendir(data_dir.c_str());
    if (!dir) throw std::runtime_error("Не удалось открыть директорию: " + data_dir);

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (std::strcmp(entry->d_name, ".") == 0 ||
            std::strcmp(entry->d_name, "..") == 0)
            continue;

        std::string fname = entry->d_name;
        if (fname.size() > 4 && fname.substr(fname.size() - 4) == ".ply") {
            std::string stem = fname.substr(0, fname.size() - 4);
            voxel_files_.push_back(data_dir + "/" + fname);
            cls_label_files_.push_back(data_dir + "/" + stem + "_cls.txt");
            seg_label_files_.push_back(data_dir + "/" + stem + "_seg.txt");
        }
    }
    closedir(dir);

    std::sort(voxel_files_.begin(),     voxel_files_.end());
    std::sort(cls_label_files_.begin(), cls_label_files_.end());
    std::sort(seg_label_files_.begin(), seg_label_files_.end());
}

std::pair<std::vector<Tensor3D>, std::vector<int>>
DataLoader::nextBatch(bool train) {
    auto& files  = voxel_files_;
    auto& labs   = cls_label_files_;
    size_t start = train ? 0            : split_index_;
    size_t end   = train ? split_index_ : files.size();
    size_t& pos  = train ? train_pos_   : val_pos_;

    std::vector<Tensor3D> batchX;
    std::vector<int>       batchY;
    batchX.reserve(batch_size_);
    batchY.reserve(batch_size_);

    for (int i = 0; i < batch_size_; ++i) {
        if (pos >= end) pos = start;
        batchX.push_back(loadVoxelMask(files[pos]));
        auto lbls = loadLabels(labs[pos]);
        if (lbls.empty())
            throw std::runtime_error("Пустой файл меток: " + labs[pos]);
        batchY.push_back(lbls[0]);
        ++pos;
    }
    return {batchX, batchY};
}

std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>
DataLoader::nextSegBatch(bool train) {
    auto& files  = voxel_files_;
    auto& segs   = seg_label_files_;
    size_t start = train ? 0            : split_index_;
    size_t end   = train ? split_index_ : files.size();
    size_t& pos  = train ? train_pos_   : val_pos_;

    std::vector<Tensor3D> batchX;
    std::vector<Tensor3D> batchY;
    batchX.reserve(batch_size_);
    batchY.reserve(batch_size_);

    for (int i = 0; i < batch_size_; ++i) {
        if (pos >= end) pos = start;
        batchX.push_back(loadVoxelMask(files[pos]));
        batchY.push_back(loadSegMask(segs[pos]));
        ++pos;
    }
    return {batchX, batchY};
}

void DataLoader::reset() {
    train_pos_ = 0;
    val_pos_   = 0;
}

Tensor3D DataLoader::loadVoxelMask(const std::string& ply_path) {
    std::ifstream in(ply_path);
    if (!in) throw std::runtime_error("Не удалось открыть PLY: " + ply_path);

    std::string line;
    size_t vertex_count = 0;

    while (std::getline(in, line)) {
        if (line.rfind("element vertex", 0) == 0) {
            std::istringstream iss(line);
            std::string elem, vert;
            iss >> elem >> vert >> vertex_count;
        }
        if (line == "end_header") break;
    }

    // Объявляем points с пробелом между '>' для совместимости
    std::vector<std::array<float,3> > points(vertex_count);
    for (size_t i = 0; i < vertex_count; ++i) {
        in >> points[i][0] >> points[i][1] >> points[i][2];
    }

    const int D = 32, H = 32, W = 32;
    Tensor3D mask(D, H, W, 1);
    mask.fill(0.0f);

    for (auto& p3 : points) {
        int x = static_cast<int>(p3[0]);
        int y = static_cast<int>(p3[1]);
        int z = static_cast<int>(p3[2]);
        if (0 <= x && x < D &&
            0 <= y && y < H &&
            0 <= z && z < W) {
            mask(x, y, z, 0) = 1.0f;
        }
    }
    return mask;
}

std::vector<int> DataLoader::loadLabels(const std::string& labels_path) {
    std::ifstream in(labels_path);
    if (!in) throw std::runtime_error("Не удалось открыть labels: " + labels_path);
    std::vector<int> labels;
    int v;
    while (in >> v) labels.push_back(v);
    return labels;
}

Tensor3D DataLoader::loadSegMask(const std::string& seg_path) {
    std::ifstream in(seg_path);
    if (!in) throw std::runtime_error("Не удалось открыть seg: " + seg_path);
    const int D = 32, H = 32, W = 32;
    Tensor3D mask(D, H, W, 1);
    for (int d = 0; d < D; ++d) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int lbl;
                in >> lbl;
                mask(d, h, w, 0) = static_cast<float>(lbl);
            }
        }
    }
    return mask;
}
