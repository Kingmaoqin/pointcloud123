#!/usr/bin/env bash
# 由仓库重建迁移包。产物为 迁移包_pointcloud123/ 与同名 .tar.gz（均已 gitignore）。
#
# 设计意图：迁移包里的 src/scripts/tests 是仓库内容的副本，不进版本库，避免仓库里
# 出现两份代码。三份指导文件的正本在 docs/ 与 迁移包说明.md，此处只做拷贝。
set -euo pipefail
cd "$(dirname "$0")/.."
B=迁移包_pointcloud123
rm -rf "$B" "$B.tar.gz"

mkdir -p "$B/01_算法代码" "$B/03_数据与实验/结果摘要/图" \
         "$B/03_数据与实验/结果摘要/缺陷记录_勿引用"

# ---- 01 算法代码 ----
cp -r src scripts tests configs "$B/01_算法代码/"
cp environment.yml requirements.txt pyproject.toml "$B/01_算法代码/"
find "$B/01_算法代码" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
cp docs/算法指导.md "$B/01_算法代码/"

# ---- 02 专利定稿 ----
P1="$B/02_专利定稿/专利一_点云缺口补扫视点规划"
mkdir -p "$P1/A_初稿_早版本" "$P1/B_修改稿_定型版本" "$P1/C_审查与代码核实"
cp "发明(初稿)：建筑点云缺口补扫视点规划方法、装置、设备及存储介质(1).docx" "$P1/A_初稿_早版本/"
cp "权利要求1-2（基于建筑点云缺口检测的补充扫描视点规划方法）.docx" "$P1/A_初稿_早版本/"
cp 初稿审查/修改稿/*.docx 初稿审查/修改稿/*.md "$P1/B_修改稿_定型版本/"
cp 初稿审查/*.docx 初稿审查/*.md "$P1/C_审查与代码核实/"

P2="$B/02_专利定稿/专利二_设计模型与在线遮挡发现"
mkdir -p "$P2/A_送审稿" "$P2/B_规划环境先验解耦_最终交付"
cp -r 送审_补充扫描规划专利/* "$P2/A_送审稿/"
cp -r 交付_20260811_规划环境先验解耦/* "$P2/B_规划环境先验解耦_最终交付/"

# ---- 03 数据与实验 ----
cp docs/数据获取指南.md docs/实验使用方法.md "$B/03_数据与实验/"
for e in exp2 exp3b exp5 exp6 exp7 exp8; do
  [ -f "results/$e/summary.md" ] && { mkdir -p "$B/03_数据与实验/结果摘要/$e"
    cp "results/$e/summary.md" "$B/03_数据与实验/结果摘要/$e/"; }
done
cp results/exp7/figs/*.png "$B/03_数据与实验/结果摘要/图/" 2>/dev/null || true
cp results/archive/exp7_prefix_stood/summary.md \
   "$B/03_数据与实验/结果摘要/缺陷记录_勿引用/" 2>/dev/null || true
# 全部原始 run.json：17 MB → 约 1.3 MB
tar czf "$B/03_数据与实验/完整实验原始记录_results.tar.gz" results

cp 迁移包说明.md "$B/README_先读这个.md"
tar czf "$B.tar.gz" "$B"
echo "完成：$(du -sh "$B" | cut -f1) 目录 / $(du -h "$B.tar.gz" | cut -f1) 压缩包"
