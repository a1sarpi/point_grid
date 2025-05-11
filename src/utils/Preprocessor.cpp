#include "Voxelizer.h"
#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: preprocess <input.txt> <output.ply>\n";
        return 1;
    }
    std::string in_txt  = argv[1];
    std::string out_ply = argv[2];

    auto points = Voxelizer::loadPointCloud(in_txt);
    if (points.empty()) {
        std::cerr << "Error loading " << in_txt << "\n";
        return 1;
    }

    std::ofstream out(out_ply);
    out << "ply\nformat ascii 1.0\n"
           "element vertex " << points.size() << "\n"
           "property float x\nproperty float y\nproperty float z\n"
           "end_header\n";
    for (auto& p : points) {
        out << p.x << " " << p.y << " " << p.z << "\n";
    }
    return 0;
}
