#include "utils/Voxelizer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <regex>
#include <map>
#include <array>
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

// Загружаем color map из JSON через простой парсер на regex
// Ищем строки вида  "12": [0.85, 0.325, 0.098]
std::map<int, std::array<uint8_t,3>> loadColorMap(const std::string& path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("Cannot open color map " + path);
    std::map<int,std::array<uint8_t,3>> m;
    // Паттерн: "(digits)": [float, float, float]
    const std::string pat =
        "\"(\\d+)\"\\s*:\\s*\\[\\s*([0-9]*\\.?[0-9]+)\\s*,\\s*"
                           "([0-9]*\\.?[0-9]+)\\s*,\\s*"
                           "([0-9]*\\.?[0-9]+)\\s*\\]";
    std::regex re(pat);
    std::string line;
    std::smatch sm;
    while(std::getline(in, line)) {
        if(std::regex_search(line, sm, re)) {
            int idx = std::stoi(sm[1].str());
            float rf = std::stof(sm[2].str());
            float gf = std::stof(sm[3].str());
            float bf = std::stof(sm[4].str());
            m[idx] = {
                static_cast<uint8_t>(std::round(rf * 255.0f)),
                static_cast<uint8_t>(std::round(gf * 255.0f)),
                static_cast<uint8_t>(std::round(bf * 255.0f))
            };
        }
    }
    return m;
}

// Пишем PLY: сначала все исходные точки (с цветом по cmap),
// затем центры занятых вокселей (в красном)
void writePLY(const std::string& out,
              const std::vector<PointNL>& pts,
              const std::vector<VoxelFeat>& vox,
              const std::map<int,std::array<uint8_t,3>>& cmap) {
    int N = VOX_N;
    // Подготовим массив вершин: x,y,z,r,g,b
    std::vector<std::array<float,6>> verts;
    verts.reserve(pts.size() + N*N*N);

    // 1) исходные точки
    for(const auto& p : pts) {
        uint8_t r=128, g=128, b=128;
        auto it = cmap.find(p.label);
        if(it != cmap.end()) {
            r = it->second[0];
            g = it->second[1];
            b = it->second[2];
        }
        verts.push_back({ p.x, p.y, p.z,
                          static_cast<float>(r),
                          static_cast<float>(g),
                          static_cast<float>(b) });
    }

    // 2) центры занятых вокселей (красные точки)
    // найдём границы и размер ячейки
    float minx=pts[0].x, maxx=pts[0].x,
          miny=pts[0].y, maxy=pts[0].y,
          minz=pts[0].z, maxz=pts[0].z;
    for(const auto& p : pts) {
        minx = std::min(minx, p.x); maxx = std::max(maxx, p.x);
        miny = std::min(miny, p.y); maxy = std::max(maxy, p.y);
        minz = std::min(minz, p.z); maxz = std::max(maxz, p.z);
    }
    float dx = (maxx - minx) / N;
    float dy = (maxy - miny) / N;
    float dz = (maxz - minz) / N;

    for(int idx = 0; idx < N*N*N; ++idx) {
        if(vox[idx][3 + 3*VOX_K] > 0.5f) {
            int i = idx / (N*N);
            int j = (idx / N) % N;
            int k = idx % N;
            float cx = minx + (i + 0.5f) * dx;
            float cy = miny + (j + 0.5f) * dy;
            float cz = minz + (k + 0.5f) * dz;
            verts.push_back({ cx, cy, cz,
                              255.0f, 0.0f, 0.0f });
        }
    }

    // Запись PLY
    std::ofstream f(out);
    if(!f) throw std::runtime_error("Cannot open output file " + out);
    f << "ply\nformat ascii 1.0\n";
    f << "element vertex " << verts.size() << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    f << "end_header\n";
    for(const auto& v : verts) {
        f << v[0] <<" "<< v[1] <<" "<< v[2] <<" "
          << int(v[3]) <<" "<< int(v[4]) <<" "<< int(v[5]) << "\n";
    }
}

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage: voxelize <pointcloud.txt>\n";
        return 1;
    }
    // 1) загрузка и вокселизация
    auto pts = Voxelizer::loadPointCloud(argv[1]);
    auto vox = Voxelizer::voxelize(pts);
    std::cout << "Points: " << pts.size()
              << ", Voxels: " << vox.size() << "\n";

    // 2) цветовая карта
    auto cmap = loadColorMap("part_color_mapping.json");

    // 3) экспорт PLY
    writePLY("output.ply", pts, vox, cmap);
    std::cout << "Saved output.ply – откройте его в любом PLY-viewer (MeshLab, etc.)\n";
    return 0;
}
