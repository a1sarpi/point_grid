#pragma once

#include "net/Tensor3D.h"
#include <string>
#include <vector>
#include <utility>

class DataLoader {
public:
    DataLoader(const std::string& data_dir,
               int batch_size,
               float val_ratio = 0.2f);

    std::pair<std::vector<Tensor3D>, std::vector<int>>
    nextBatch(bool train = true);

    std::pair<std::vector<Tensor3D>, std::vector<Tensor3D>>
    nextSegBatch(bool train = true);

    void reset();

    // Новые методы
    /// Общее число примеров в тренировочной части
    size_t getNumTrainSamples() const { return split_index_; }
    /// Общее число примеров в валидационной части
    size_t getNumValSamples()   const { return voxel_files_.size() - split_index_; }
    /// Размер батча
    int    getBatchSize()       const { return batch_size_; }

private:
    std::vector<std::string> voxel_files_;
    std::vector<std::string> cls_label_files_;
    std::vector<std::string> seg_label_files_;

    int    batch_size_;
    float  val_ratio_;
    size_t split_index_;

    size_t train_pos_ = 0;
    size_t val_pos_   = 0;

    void loadFileLists(const std::string& data_dir);

    Tensor3D loadVoxelMask(const std::string& ply_path);
    std::vector<int> loadLabels(const std::string& labels_path);
    Tensor3D loadSegMask(const std::string& seg_path);
};
