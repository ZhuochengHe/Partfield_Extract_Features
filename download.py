#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
from datasets import load_dataset
import objaverse
import os
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Batch 参数
    parser.add_argument("--batch", type=int, required=True,
                        help="Batch ID starting from 0")
    parser.add_argument("--batch_size", type=int, default=300,
                        help="Number of GLBs per batch")

    # 根路径
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing cache & batch folders.")

    return parser.parse_args()


def main():
    args = parse_args()

    BATCH_ID = args.batch
    BATCH_SIZE = args.batch_size
    DATA_DIR = args.data_dir

    # 固定 cache，不随 batch 变化
    CACHE_DIR = os.path.join(DATA_DIR, "objaverse_cache")

    # 每个 batch 拥有自己的 glb 目录
    DOWNLOAD_DIR = os.path.join(DATA_DIR, f"objaverse_glbs/batch_{BATCH_ID}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print(f"🌟 Batch = {BATCH_ID}, size = {BATCH_SIZE}")
    print(f"📁 Base Dir   = {DATA_DIR}")
    print(f"📁 Cache Dir  = {CACHE_DIR} (shared)")
    print(f"📁 Batch Out  = {DOWNLOAD_DIR}")

    # 设置 objaverse 缓存路径
    objaverse.BASE_PATH = DATA_DIR
    objaverse._VERSIONED_PATH = CACHE_DIR

    # 加载注释
    print("加载 Objaverse++ 高质量列表...")
    ds = load_dataset("cindyxl/ObjaversePlusPlus", split="train")

    # 高质量 ID
    hq_ids = [row["UID"] for row in ds if int(row["score"]) == 3]
    total = len(hq_ids)
    print(f"✨ High+Superior 共 {total} 个")

    # 分批
    start = BATCH_ID * BATCH_SIZE
    end = min(start + BATCH_SIZE, total)

    if start >= total:
        print(f"❌ Batch {BATCH_ID} 超出范围 (max index = {total})")
        return

    uids_to_download = hq_ids[start:end]
    print(f"➡️ 当前批次 UID 范围: {start} ~ {end-1} (共 {len(uids_to_download)} 个)")

    # 下载 glb
    processes = multiprocessing.cpu_count()
    print(f"🚀 开始下载，使用 {processes} 个进程…")
    objects = objaverse.load_objects(
        uids=uids_to_download,
        download_processes=processes,
    )

    # 保存 glb
    saved = 0
    for uid, path in objects.items():
        if os.path.isdir(path):
            files = [f for f in os.listdir(path) if f.endswith(".glb")]
            if not files:
                print(f"⚠️ 无 glb：{uid}")
                continue
            src = os.path.join(path, files[0])
        else:
            src = path

        dest = os.path.join(DOWNLOAD_DIR, os.path.basename(src))
        shutil.copy(src, dest)
        saved += 1

    print(f"🎉 完成！本批次下载 {saved} 个 GLB → {DOWNLOAD_DIR}")


if __name__ == "__main__":
    main()
