# 数据资源说明

本仓库**不包含**第三方原始数据集，只包含代码、实验结果与文档。原因有二：体积超出
GitHub 的合理范围；以及第三方数据的再分发授权不在本项目掌握之内。

---

## 一、仓库内已包含的数据（可直接使用）

| 内容 | 位置 | 体积 | 说明 |
|---|---|---|---|
| **全部实验结果** | `results/` | 约 17 MB | E2 / E3b / E5 / E6 / E7 / E8 的逐次运行记录（`run.json`）、汇总表与统计、曲线图。每个 `run.json` 内记录产出它的 `git_commit` |
| 合成变电站场景 | 由代码生成 | — | `src/patent_gap/simulation/scene_gen.py`，给定 seed 完全可复现，无需下载任何数据 |
| 参数与配置 | `configs/` | 64 KB | 各实验的配置 |

**合成场景的全部实验（E2/E5/E7/E8）不需要任何外部数据即可完整复现。**

---

## 二、需要另行获取的数据（真实建筑 CRAS）

只有 E6（真实建筑实施例）依赖它。

| 文件 | 体积 | 用途 |
|---|---|---|
| `craslabannotated.zip` | 4.0 GB | 标注点云 |
| `craslabbim.ifc` | 65 MB | 建筑 IFC 模型 |

### 获取方式

```bash
bash scripts/download_cras.sh          # 下载到 data/raw/
python -m patent_gap.cli inspect-data --config configs/data/cras.yaml
python scripts/preprocess_cras.py      # 生成 data/processed/ 下的派生缓存
```

`.gitignore` 已排除 `data/raw/*.zip`、`data/raw/*.ifc` 与 `data/processed/`，
下载后不会被误提交。

---

## 三、复现顺序

```bash
conda env create -f environment.yml && conda activate patent_gap    # 或 pip install -r requirements.txt
pytest tests/ -q                                                     # 应为 128 passed

# 主实验（不需要外部数据）
python scripts/run_e7_prior.py --out results/exp7                    # 规划环境先验强度消融，240 次运行
python scripts/run_e8_frontier.py --out results/exp8                 # frontier 探索基线对照，144 次运行
python scripts/plot_e7.py                                            # 出图
python scripts/check_delivery_numbers.py                             # 核对文档数字与原始结果是否一致

# 真实建筑实施例（需先获取 CRAS）
python scripts/run_e6_real_cras.py
```

> **版本守卫**：断点续跑时若发现盘上结果产自不同 commit，脚本会直接 `SystemExit`
> 拒绝，以免拼出版本混合的结果集。要重跑请先清空对应输出目录。
