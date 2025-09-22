#!/bin/bash
SOURCE_DIR="/data/jizheng/pose/train"  # 包含zip文件的目录
TARGET_DIR="/data/jizheng/PAB/pose/train"  # 解压后的目标目录

# 确保目标目录存在
mkdir -p "$TARGET_DIR"

# 遍历所有zip文件并解压到目标目录
for zipfile in "$SOURCE_DIR"/*.zip; do
    unzip -o "$zipfile" -d "$TARGET_DIR"
    echo "解压完成：$zipfile"
done
