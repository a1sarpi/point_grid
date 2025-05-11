import json
import os

def colorize_ply(ply_in, labels_txt, color_json, ply_out):
    # 1) Загрузить JSON-маппинг
    with open(color_json) as f:
        cmap = json.load(f)  # ключи как str, значения [r,g,b]

    # 2) Считать метки
    with open(labels_txt) as f:
        labels = [int(line.strip()) for line in f]

    # 3) Считать PLY
    with open(ply_in) as f:
        lines = f.readlines()
    # Найти конец header
    hdr_end = 0
    for i, l in enumerate(lines):
        if l.strip() == "end_header":
            hdr_end = i
            break
    header = lines[:hdr_end+1]
    verts  = lines[hdr_end+1:]

    # 4) Записать новый PLY с цветами
    with open(ply_out, "w") as f:
        # Добавляем свойства цвета в header
        for l in header:
            if l.startswith("element vertex"):
                # вставляем color properties сразу после count
                f.write(l)
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            else:
                f.write(l)
        # Потом все вершины с цветом
        for lbl, vert in zip(labels, verts):
            xyz = vert.strip().split()[:3]
            rgb = cmap.get(str(lbl), [128,128,128])
            f.write(f"{xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ply_in",     required=True)
    p.add_argument("--labels_txt", required=True)
    p.add_argument("--color_json", required=True)
    p.add_argument("--ply_out",    required=True)
    args = p.parse_args()
    colorize_ply(args.ply_in, args.labels_txt, args.color_json, args.ply_out)
