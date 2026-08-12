# 06 Oracle Leakage 审计

**Oracle leakage** 在此指：规划器读到了真实现场**不可能提供**给它的信息。判据不是
「读没读完整模型」，而是「作业当天，这条信息拿不拿得到」。

证据等级：**[A]** 已在代码中核实 ｜ **[B]** 由实现推出 ｜ **[C]** 设计取舍 ｜
**[D]** 未确认。

---

## 一、规划器读几何的全部位置

逐一列出，不靠印象。`grep` 范围为 `src/` 与 `scripts/`。

| 位置 | 读什么 | 属规划还是评测 | 判定 |
|---|---|---|---|
| `planning_map(world)` | M_plan 占据侧（M_occ） | 规划 | ✅ 观测驱动 |
| `score_candidates_v2(..., oracle=plan_oracle, ...)` | M_plan 遮挡侧（M_vis） | 规划 | ✅ 内容为该先验档给出的几何（M_ref 目标表面 ＋ 地面 ＋ 该档保留的环境构件）＋ 已发现几何；不含该档未给出的任何几何 |
| `_make_candidates` → `world.patches` | 目标分块形心与法向 | 规划 | ✅ M_ref 本职；已剔除非目标环境构件 |
| `world.sampler` | 目标表面采样点 | 规划 | ✅ 实测 `tri_to_patch>=0` 仅覆盖目标构件（环境/临时构件 0/14） |
| `UnmodeledOccluders(base=world.ref_tris)` | 基准网格 | 规划 | ✅ 基准为**该先验档实际给出的几何**（M_ref 目标表面 ＋ 地面 ＋ 该档保留的环境构件），即 M_vis 的初始内容；不含任何该档未给出的几何。详见 `09` 核查一 |
| `set_select` / `route` / `gap.scoring` / `registration.predictor` | — | 规划 | ✅ 完全不读几何 |
| `station_sample_masks` → `world.oracle` | 竣工实景全几何 | **评测/仿真** | ✅ Simulator Truth，即"传感器实际看到什么" |
| `build_ground_truth` → `world.grid` / `world.oracle` | 竣工实景 | **评测** | ✅ 真值；每个（场景，种子）算一次后各先验档共用 |
| `vis_audit_at` → `world.oracle` | 竣工实景 | **测量** | ✅ 在选站**之后**计算，不进入任何决策 |
| `default_init_stations` → `world.grid` | 竣工实景可通行图 | 规划前 | ⚠️ **残留项**，见第三节 |

---

## 二、已发现并修复的泄漏

### 2.1 可通行空间来自完整模型 —— 最强的一处

`build_trav_grid(scene)` 遍历完整模型全部构件生成障碍与膨胀，**与任何观测无关**，
且 `world.grid` 在整个闭环中从不更新。规划器在执行第一站之前就知道现场每个障碍的
平面占位。**[A]**

**修复**：改用 `PlanningEnvGrid`（三态、观测驱动），并在 S8 增加占据侧回灌。

### 2.2 可见性对完整模型求交

`plan_oracle` 在既有全部正式实验中直接复用评测用的 `oracle`（`bim_tri_mask()` 恒为
全真），即规划器预测可见性与仿真器生成"实际看到什么"用的是同一份几何。**[A]**

**修复**：`plan_oracle` 初始内容改为 `prior_tri_mask(env_prior_frac)`，P0 时仅剩
M_ref 目标表面与地面。

### 2.3 规划器绕开了设计模型里查不到的临时占位物

`build_trav_grid` 遍历 `scene.components` 时**不看 `in_bim`**。竣工态临时占位物
（施工车辆、料堆、脚手架）因此进入规划器的障碍图 —— 它绕开了一批它不可能知道位置的
东西。实测 M_mid seed0 `n_temp=6`：规划器认为可走 23937 格，实景只有 23356 格。**[A]**

**修复**：`build_trav_grid(bim_only=True)` 生成 `SimWorld.plan_grid` 供规划器使用；
评测真值与初始站仍用竣工实景图（参考站集要 physically 可站）。`n_temp=0` 时二者是
同一对象。

**对既有实验的影响**：E2 / E5 / E6 均取 `n_temp=0`，该泄漏不激活，结果逐比特不受
影响 **[A]**。**E3b 的 `n_temp=6` 那一批带着这个泄漏** —— 它对 `B10_full` 与
`B11_disc` 同等生效（二者共用同一张图），故 **B11−B10 的对照不因此失效，但两条线的
绝对水平都偏乐观**。E3b 结果保留在盘上未作改动。**[B]**

### 2.4 目标集与 M_ref 口径不一致（物理上讲不通的先验轴）

`build_scene_patches` 把 `building` / `fence` / `clutter` 也当成待扫目标，而
`prior_tri_mask` 又把它们当作"非目标环境"从 M_plan⁽⁰⁾ 移除。于是 P0 下规划器被要求
扫一栋厂房，却拿不到那栋厂房的几何 —— 它在求交时会以为自己能透视过去。**[A]**

**修复**：非目标环境构件不再进入目标分块表（占合成变电站分块数的 5.7–8.0%）。不变量
「凡 M_ref 声明为目标的表面，M_vis 的初始内容必须含其几何」由测试锁定。

**代价**：目标集本身改变，**E7/E8 与 E2/E5/E6 的绝对指标不可跨实验直接比较**；各
实验内部的对照不受影响。`run.json` 的 `git_commit` 守卫会拦住把两种口径拼在一起的
续跑。既有结果全部保留在盘上。

### 2.5 （非泄漏，但曾使 P0 看上去完全失效）M_plan 更新被缓存吞掉

`TravGrid._compute_free` 缓存 `self._free` 且只在其为 `None` 时重算；
`PlanningEnvGrid` 改写了被包装栅格的 `_obstacle` 却未让该缓存失效，于是**每一轮新观测
对可通行空间的修正都被丢弃**，自由空间冻结在初始状态。表现为 P0 的 awc 0.043，看着
像"弱先验下方法不work"。**[A]** 修复后同条件 awc 0.725。

> 这一条记在这里是因为它的表现形态最危险：一个安静的低分，看上去像是**结论**，
> 实际上是**缺陷**。若当时直接把 0.043 写进报告，得到的会是一个完全错误的结论。

---

## 三、残留项（保留，并说明理由）

### 3.1 初始站集由竣工实景可通行图挑选

`default_init_stations(world)` 读 `world.grid`。**[A]**

**为什么保留**：初始扫描在本方法介入之前**已经发生**，正是它产生了待补的缺口。初始
站位是作业输入（现场记录的实际架站位置），不是本算法规划的结果。**[C]**

**为什么不影响 E7 的结论**：各先验档使用**同一组**初始站（每个场景/种子算一次后传给
全部档位），因此它不构成档位之间的混淆因素。**[A]**

**诚实的边界**：在 P0 下，"第一站站在哪"这件事本身仍不是由观测推出的。本方法不主张
解决"完全无信息时如何架第一站"——那是初始扫描作业的问题，不在本发明范围内。

### 3.2 评测侧全部使用竣工实景几何

真值站集、`C_gt`、缺口判定、逐站可见掩码、可见性审计均读 `world.oracle` /
`world.grid`。这是 Simulator Truth，**必须**用完整几何，否则"预测错了没有"无从测量。
它们不进入任何规划决策：`vis_audit_at` 在选站之后调用，其返回值只写进 `history`。
**[A]**

### 3.3 带电体禁入区作为先验写入 M_plan

`add_safety_prior` 在无任何观测时即可写入禁入区。**[C]**

**理由**：这是作业安全规程而非现场几何，与是否已经扫描无关；且它**只增加禁入、不增加
可通行**，方向上只会让规划器更保守，不会让它多知道哪里能走。

---

## 四、测试覆盖

| 不变量 | 测试 |
|---|---|
| P100 与既有实现逐比特相同 | `test_full_prior_is_bit_identical_to_previous_behaviour` |
| 目标表面在各档都在 M_plan⁰ 中 | `test_every_target_surface_is_present_in_the_planning_model` |
| 非目标环境不是待扫目标，但仍参与遮挡 | `test_non_target_environment_is_not_a_scan_target` |
| S1 不随环境先验强度变化 | `test_s1_is_unaffected_by_how_much_environment_the_model_describes` |
| P0 下遮挡模型不含未观测环境几何 | `test_weak_prior_planner_cannot_see_unobserved_environment` |
| 未知不得默认可通行 | `test_unknown_is_not_free_by_default` |
| 新观测确实并入 M_plan | `test_new_observation_grows_the_planning_map` |
| 在线发现的几何不成为新目标 | `test_discovered_geometry_never_becomes_a_scan_target` |
| 规划器不绕开模型里查不到的东西 | `test_planner_traversability_excludes_geometry_absent_from_the_model` |
| **探索基线只读当前 M_plan** | `test_frontier_baseline_reads_only_the_observed_map`（把 `world.grid`/`plan_grid` 换成一读就抛异常的哨兵） |
| 审计开关默认关闭 | `test_vis_audit_is_off_by_default_and_records_when_on` |
| 平台站过的位置不被膨胀抹掉 | `test_a_position_the_platform_occupied_stays_traversable` |
| 但已确认占据的格不被站位圆盘强行放开 | `test_a_stood_cell_confirmed_occupied_is_not_forced_free` |
| **已作为先验的环境构件不被重复发现** | `test_known_prior_environment_is_not_rediscovered` |
| 发现累积幂等 | `test_discovery_accumulation_is_idempotent` |
| 二维 carving 的已知近似边界与两条自限机制 | `test_free_carving_is_a_two_dimensional_approximation` |

---

## 五、结论

1. 改造前存在**两处**真实的过强先验（可通行空间、可见性求交），外加**一处**更隐蔽的
   泄漏（规划器绕开设计模型中不存在的临时占位物）。**[A]**
2. 三处均已修复；修复对 `n_temp=0` 的既有实验逐比特无影响，既有结果全部保留。**[A]**
3. 残留一处**有意保留**的完整模型读取（初始站集），已说明理由，且不构成 E7 各档之间
   的混淆因素。**[A]**
4. 评测侧使用完整几何是必须的，且与规划侧完全分离。**[A]**

---

## 六、代码核查后的复审（本轮 Phase A 之后）

`09_剩余代码逻辑核查_FINAL.md` 完成两项代码事实核查后，重新确认 P0 条件下规划决策**之前**
不读取以下任何一项 **[A]**：

| 不得读取 | 核查结论 |
|---|---|
| 完整环境障碍图 | ✅ `planning_map()` 返回 `world.env`；`world.grid`/`plan_grid` 在闭环内不被规划路径读取，已由哨兵测试锁定 |
| 完整环境求交网格 | ✅ 评分一律走 `plan_oracle`，其内容由 `ref_tris` 决定 |
| 临时占位物位置 | ✅ `plan_grid` 按 `bim_only=True` 建立；`temp_obstacle` 的 `in_bim=False`，不进入 |
| 真值可见性 | ✅ `vis_audit_at` 在**选站之后**调用，返回值只写进 `history` |
| 真值自由空间 | ✅ 同上；`build_ground_truth` 每（场景，种子）算一次后各档共用，不进入决策 |

允许读取且已核实其用途 **[A]**：M_ref 目标几何、当前观测、安全规程；以及 simulator
truth —— 仅用于产生传感器观测、事后评价、以及选站后的审计。

**残留项（3.1 初始站位）继续保留**，不假装解决"完全无信息情况下第一站去哪"。
