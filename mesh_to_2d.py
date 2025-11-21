#!/usr/bin/env python3
"""
CPU 渲染 GLB 文件 - 每个模型多视角
使用 trimesh 自带渲染生成训练用彩色 2D 图像
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        
import argparse
import pathlib
import trimesh
from PIL import Image
from tqdm import tqdm
import numpy as np

def render_glb_multi_views(glb_path, output_dir, num_views=6, resolution=(512,512)):
    """
    渲染单个 GLB 文件多个视角
    """
    glb_path = pathlib.Path(glb_path)
    model_uid = glb_path.stem
    model_output_dir = pathlib.Path(output_dir) / model_uid
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        mesh = trimesh.load(str(glb_path), force='mesh')
        if mesh.is_empty:
            return False

        # 计算旋转角度
        angles = np.linspace(0, 360, num_views, endpoint=False)

        for i, angle in enumerate(angles):
            scene = mesh.scene()
            # 旋转网格
            scene.camera_transform = trimesh.transformations.rotation_matrix(
                np.radians(angle), [0,1,0], mesh.centroid
            )
            # 渲染图片 bytes
            png = scene.save_image(resolution=resolution, visible=False, flags={'offscreen': True})
            # 转成 PIL 保存
            img = Image.open(trimesh.util.wrap_as_stream(png))
            img.save(model_output_dir / f"{model_uid}_view_{i:02d}.png")
        return True
    except Exception as e:
        print(f"[WARNING] Failed {glb_path.name}: {e}")
        return False

def find_all_glb_files(input_dir):
    return sorted(pathlib.Path(input_dir).rglob("*.glb"))

def batch_render(input_dir, output_dir, num_views=6, resolution=(512,512)):
    input_dir = pathlib.Path(input_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    glb_files = find_all_glb_files(input_dir)
    print(f"[INFO] Found {len(glb_files)} GLB files in {input_dir}")
    
    successful = 0
    failed = 0

    for glb_file in tqdm(glb_files, desc="Rendering GLB files"):
        if render_glb_multi_views(glb_file, output_dir, num_views, resolution):
            successful += 1
        else:
            failed += 1

    print(f"[INFO] Rendering complete: {successful} success, {failed} failed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="GLB 文件目录")
    parser.add_argument("--output-dir", default="./rendered_views", help="输出目录")
    parser.add_argument("--num-views", type=int, default=6, help="每个模型视角数")
    parser.add_argument("--resolution", type=int, nargs=2, default=[512,512], help="分辨率 WxH")
    args = parser.parse_args()

    batch_render(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_views=args.num_views,
        resolution=tuple(args.resolution)
    )

if __name__ == "__main__":
    main()
