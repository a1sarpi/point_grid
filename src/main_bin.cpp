#include "utils/Voxelizer.h"
#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <array>
#include <map>
#include <cmath>
#include <regex>

#include <regex>
#include <fstream>
#include <string>
#include <map>
#include <array>
#include <cmath>
#include <cstdint>

static std::map<int,std::array<uint8_t,3>> loadColorMap(const std::string& path){
    std::ifstream in(path);
    std::map<int,std::array<uint8_t,3>> out;
    // Обратите внимание: маркер raw защищает двойные кавычки и слэши внутри шаблона
    std::regex re(R"raw("(\d+)"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\])raw");
    std::string line;
    std::smatch m;
    while(std::getline(in,line)){
        if(std::regex_search(line,m,re)){
            int idx = std::stoi(m[1].str());
            float rf = std::stof(m[2].str()),
                  gf = std::stof(m[3].str()),
                  bf = std::stof(m[4].str());
            out[idx] = {
                uint8_t(std::round(rf*255)),
                uint8_t(std::round(gf*255)),
                uint8_t(std::round(bf*255))
            };
        }
    }
    return out;
}


// Запись PLY: точки (с цветом по метке) + центры вокселей
static void writePLY(const std::string& fn,
    const std::vector<PointNL>& pts,
    const std::vector<VoxelFeat>& vox)
{
    auto cmap = loadColorMap("part_color_mapping.json");
    int N = VOX_N;

    // 1) собираем вершины
    std::vector<std::array<float,6>> verts;
    verts.reserve(pts.size());
    for(auto& p: pts){
        auto it = cmap.find(p.label);
        uint8_t r=128,g=128,b=128;
        if(it!=cmap.end()){
            r = it->second[0];
            g = it->second[1];
            b = it->second[2];
        }
        verts.push_back({p.x,p.y,p.z,float(r),float(g),float(b)});
    }
    // 2) находим центры занятых вокселей
    //    (забираем те, у которых vox[i][3+3*VOX_K] > 0.5)
    //    и добавляем их красным цветом
    float minx=pts[0].x, maxx=pts[0].x,
          miny=pts[0].y, maxy=pts[0].y,
          minz=pts[0].z, maxz=pts[0].z;
    for(auto& p: pts){
        minx = std::min(minx,p.x); maxx = std::max(maxx,p.x);
        miny = std::min(miny,p.y); maxy = std::max(maxy,p.y);
        minz = std::min(minz,p.z); maxz = std::max(maxz,p.z);
    }
    float dx=(maxx-minx)/N, dy=(maxy-miny)/N, dz=(maxz-minz)/N;
    for(int idx=0; idx < N*N*N; ++idx){
        if(vox[idx][3 + 3*VOX_K] > 0.5f){
            int i = idx/(N*N), j = (idx/N)%N, k = idx%N;
            float cx = minx + (i+0.5f)*dx;
            float cy = miny + (j+0.5f)*dy;
            float cz = minz + (k+0.5f)*dz;
            verts.push_back({cx,cy,cz,255.0f,0.0f,0.0f});
        }
    }

    // 3) пишем PLY
    std::ofstream f(fn);
    f<<"ply\nformat ascii 1.0\n";
    f<<"element vertex "<<verts.size()<<"\n";
    f<<"property float x\nproperty float y\nproperty float z\n";
    f<<"property uchar red\nproperty uchar green\nproperty uchar blue\n";
    f<<"end_header\n";
    for(auto& v:verts){
        f<<v[0]<<" "<<v[1]<<" "<<v[2]<<" "
         <<int(v[3])<<" "<<int(v[4])<<" "<<int(v[5])<<"\n";
    }
    std::cout<<"Wrote "<<fn<<"\n";
}

int main(int argc,char** argv){
    if(argc!=2){
        std::cerr<<"Usage: voxelize_bin <cloud.bin>\n";
        return 1;
    }
    // 1) читаем бинарник
    std::ifstream in(argv[1], std::ios::binary);
    uint32_t M;
    in.read((char*)&M, sizeof(M));
    std::vector<PointNL> pts;
    pts.reserve(M);
    for(uint32_t i=0;i<M;++i){
        PointNL p;
        in.read((char*)&p.x, sizeof(float));
        in.read((char*)&p.y, sizeof(float));
        in.read((char*)&p.z, sizeof(float));
        p.nx = p.ny = p.nz = 0.0f;
        int32_t lbl;
        in.read((char*)&lbl, sizeof(lbl));
        p.label = lbl;
        pts.push_back(p);
    }
    // 2) вокселизация
    auto vox = Voxelizer::voxelize(pts);
    // 3) запись PLY
    writePLY("output.ply", pts, vox);
    return 0;
}
