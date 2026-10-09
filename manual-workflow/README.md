# 手动混合任务与温度反馈实验

本目录是本次梳理新增的操作示例；不修改已有启动器、配置或优化算法。支持 2 个及更多已有 plain `.traceg` 输入。温度反馈是候选频率估计，需要实际复跑确认，不是完整 DVFS 控制器。

新增的矩阵反解见 [MATRIX_INVERSE.md](MATRIX_INVERSE.md)：直接加载已导出的 G/B/L，支持立方功耗假设、背景项和多频率拟合；已完成预测功耗重新输入 HotSpot 的验证。下面的单点比例与两点割线仍保留为简单近似。

实际功耗频率公式的源码依据见 [ACCELWATTCH_FREQUENCY.md](ACCELWATTCH_FREQUENCY.md)。立方律不是当前固定电压 AccelWattch 的实际实现关系。

2026-09-29 完成了 Core=ICNT=L2、DRAM 固定 850 MHz 的八频率真实仿真，见 [实测 P–f 报告](../manual-results/power-frequency-20260929/REPORT.md)。本组三任务/固定 SM 映射的平均输出在 800–4000 MHz 内用二次模型比严格线性更准确；不是所有任务通用定律。

## 支持范围与限制

- 原项目已有 stream 隔离、多 kernel 并发、数量或具体 SM 映射、统一四时钟域设置、逐 SM 功耗输出、HotSpot 稳态分析。
- 原 `scripts/gen_mix_trace.py` 直接支持两路；本目录 `merge_traces.py` 支持任意多路，按输入顺序指定 stream 1、2、3……。
- 混合 trace 是把独立采集的任务轨迹交替提交。不同 stream 允许重叠，但 memcpy、启动等待及任务长度可能导致并非全程同时运行；不同任务地址空间不在本示例中重定位，若地址范围重叠，会影响共享缓存干扰结果。
- 数量格式 `1:43,2:9,3:28` 表示连续区间：stream 1 为 SM 0–42，stream 2 为 43–51，stream 3 为 52–79；它不是 Alg2 的交错布局。该示例保持 80 个物理 SM，与当前 floorplan 匹配。
- Core、ICNT、L2 同频调整，DRAM 固定 850 MHz，电压固定；不支持每个任务各自指定频率。
- 原 `lyhong_power_report.log` 已经是 83 列 HotSpot power trace：SM_0…SM_79、L2_cache、HBM2_0、HBM2_1，单位 W，**无需从标准总功耗报告拆分**。
- 该 trace 未包含 MCP、NOCP、CONSTP、STATICP 等非 SM 项；`lyhong_SM_power_report.log` 有相应总量。当前热模型映射只覆盖导出的功耗子集。得到完整芯片温度需要为缺失项建立有物理依据的空间功耗映射，并避免重复计入 SM 功耗。本示例不猜测其放置。
- HotSpot 不指定 `-o` 即仅算稳态；本版本对各功耗行求算术平均后计算稳态。等长采样可直接平均；末尾不足采样区间的处理可能影响短任务平均值。本示例设 100 核心周期采样以降低短任务粗粒度误差。
- 稳态结果表示持续施加平均负载达到热平衡，不能用来描述一次几十微秒任务的真实瞬态峰温。
- `temp.steady` 当前 `grid_map_mode=center` 为模块中心温度。建议用 `temp.grid.steady` 活性硅层全部网格的最大值作为芯片温度阈值指标；默认 64×64。该指标包括 SM、L2、HBM 的对应区域。若阈值只针对 SM，需要进一步采用 SM 区域掩码。
- `run_round.sh`、完整新一轮 GPU 仿真未在本次验证中运行；已验证三路合并、小文件流编号、脚本语法和历史真实功耗输入的 HotSpot 运行。

## 1. 准备

以下命令在同一个 bash 会话中执行。当前已有可执行文件；源码变更后再编译。

```bash
cd /home/lyhong/Downloads/Accel-Hotsim
export EXP_ROOT="$PWD"
export CUDA_INSTALL_PATH=/home/lyhong/cuda-12.8
export OPENCL_REMOTE_GPU_HOST="${OPENCL_REMOTE_GPU_HOST:-}"
export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
source "$EXP_ROOT/gpu-simulator/setup_environment.sh"
# 如需要重新编译：
# make -j -C "$EXP_ROOT/gpu-simulator"
```

当前 `Lyhong_Percore_sim=1`。原 `hotspot-link/hotspot` 可执行文件已存在，且其层文件使用当前项目的绝对 floorplan 路径。

## 2. 构建三任务混合输入

输出必须是新目录；重跑时换目录名，避免覆盖原混合 trace。

```bash
export MIX_TRACE="$EXP_ROOT/manual-results/mix3/traces"
python3 "$EXP_ROOT/manual-workflow/merge_traces.py" \
  --input "backprop=$EXP_ROOT/hw_run/rodinia_2.0-ft/9.1/backprop-rodinia-2.0-ft/4096___data_result_4096_txt/traces" \
  --input "bfs=$EXP_ROOT/hw_run/rodinia_2.0-ft/9.1/bfs-rodinia-2.0-ft/__data_graph4096_txt___data_graph4096_result_txt/traces" \
  --input "heartwall=$EXP_ROOT/hw_run/rodinia_2.0-ft/9.1/heartwall-rodinia-2.0-ft/__data_test_avi_1___data_result_1_txt/traces" \
  --out "$MIX_TRACE"
```

stream 1=backprop，2=bfs，3=heartwall。两任务只需去掉一项并重新选择输出目录；更多任务继续添加 `--input`。

也可直接使用项目已有的三任务输入：

```bash
export MIX_TRACE="$EXP_ROOT/hw_run/rodinia_2.0-ft/9.1/backprop-bfs-heartwall-rodinia-2.0-ft/backprop_bfs_heartwall/traces"
```

## 3. 第一轮性能、功耗与稳态温度

```bash
export SM_SPEC='1:43,2:9,3:28'
export FREQ1=1132
export ROUND1="$EXP_ROOT/manual-results/round1"
bash "$EXP_ROOT/manual-workflow/run_round.sh" \
  "$MIX_TRACE" "$FREQ1" "$SM_SPEC" "$ROUND1"
```

SM 数量可自行更改，但本示例要求正整数、总和为 80，并覆盖所有 trace stream。输出目录必须不存在。脚本直接调用 Accel-Sim，无需在 benchmark YAML 中注册，也绕过启动器的自动 Alg1/Alg2 分配。

脚本生成 `gpgpusim.config`、`trace.config`、`experiment.config` 和 `stream_sm_alloc.txt`。最后载入的 `experiment.config` 指定实验参数：

```text
-gpgpu_clock_domains 1132:1132:1132:850.0
-gpgpu_n_clusters 80
-gpgpu_n_cores_per_cluster 1
-gpgpu_concurrent_kernel_sm 1
-gpgpu_concurrent_stream_sm_config_file ./stream_sm_alloc.txt
-power_simulation_enabled 1
-power_simulation_mode 0
-accelwattch_xml_file ./accelwattch_sass_sim.xml
-dvfs_enabled 0
-gpgpu_runtime_stat 100:0
```

底层仿真命令（先在该轮目录中准备上述配置和 XML、ICNT 文件）：

```bash
cd "$ROUND1"
"$EXP_ROOT/gpu-simulator/bin/release/accel-sim.out" \
  -config ./gpgpusim.config -config ./trace.config -config ./experiment.config \
  -trace "$MIX_TRACE/kernelslist.g" > sim.log 2>&1
```

## 4. 查看输出

```bash
cat "$ROUND1/performance.json"
tail -30 "$ROUND1/accelwattch_power_report.log"
tail -30 "$ROUND1/lyhong_SM_power_report.log"
cat "$ROUND1/mapped_average_power.csv"
```

| 输出 | 内容 |
|---|---|
| `sim.log` | 频率、累计指令数/周期数、IPC、缓存和访存统计、错误信息 |
| `performance.json` | 最后累计统计、执行时间 μs、热映射平均 W、采样行数 |
| `accelwattch_power_report.log` | 原 AccelWattch 报告，沿用其 kernel/区间统计口径 |
| `lyhong_power_report.log`、`power.ptrace` | 同一份逐 SM/L2/HBM 功耗样本 |
| `lyhong_SM_power_report.log` | 各组件功耗及 SM、非 SM、GPU 总功耗记录 |
| `mapped_average_power.csv` | 各热模型输入块的平均功率，非完整 GPU 总功率 |
| `temp.steady` | 按模块输出的稳态 K，当前为中心映射 |
| `temp.grid.steady` | 多层网格稳态 K，前 4096 行为活性硅层 |
| `hotspot.log`、`model_extract/` | 热分析日志及该 HotSpot 版本自动导出的模型矩阵 |

全程总时间 `gpu_tot_sim_cycle / MHz` 得到 μs。`gpgpu_simulation_time` 是主机运行时间。各任务 IPC 需要按其 stream 的执行窗口和指令计数定义，不能直接把全局 `gpu_tot_ipc` 当成各任务 IPC。

## 5. 单独运行稳态热分析及绘图

run_round 已执行这一操作。如只想重算热模型，可在对应轮次目录执行：

```bash
cd "$ROUND1"
cp lyhong_power_report.log power.ptrace
"$EXP_ROOT/hotspot-link/hotspot" \
  -c "$EXP_ROOT/hotspot-link/GV100-hotspot.config" \
  -p "$ROUND1/power.ptrace" \
  -grid_layer_file "$EXP_ROOT/hotspot-link/flp/gv100.lcf" \
  -model_type grid \
  -steady_file "$ROUND1/temp.steady" \
  -grid_steady_file "$ROUND1/temp.grid.steady" > hotspot.log 2>&1

python3 "$EXP_ROOT/hotspot-link/plot_gv100_grid_heatmap.py" \
  --grid-steady "$ROUND1/temp.grid.steady" \
  --flp "$EXP_ROOT/hotspot-link/flp/gv100_new.flp" \
  --rows 64 --cols 64 --layer 0 --out "$ROUND1/temperature.png"
```

稳态分析无需按新频率调整 `sampling_intvl`；它使用平均功耗求热平衡。瞬态分析则需要匹配采样时间。

## 6. 第一轮温度反馈计算候选频率

固定电压下通常写作 `P(f)=P_fixed+K V² f`，热模型固定时 `T-Ta=R P`（多节点为矩阵映射）。只有忽略固定项、且全部输入功率按同一比例随频率变化时，才可使用：

`f_candidate = f_current * (T_limit - T_ambient)/(T_peak - T_ambient)`。

本项目有固定 DRAM、空闲功耗和变化的利用率，因此这个单点公式只用于首轮试探；不能保证下一轮刚好到达阈值，也不能由一个温度和功率点唯一确定固定功耗与动态系数。

当前环境温度配置为 318.15 K = 45°C。以下 80°C 是可修改的示例阈值，不是项目已有规定。

```bash
export THRESHOLD_C=80
export FREQ2="$(python3 "$EXP_ROOT/manual-workflow/suggest_frequency.py" \
  --steady "$ROUND1/temp.grid.steady" --grid \
  --freq "$FREQ1" --threshold-c "$THRESHOLD_C" --ambient-c 45 \
  --min-mhz 300 --max-mhz 4000)"
printf 'Next candidate frequency: %s MHz\n' "$FREQ2"
```

诊断输出到 stderr，stdout 仅输出数值供下一步使用。频率上下限是实验搜索范围，不是硬件额定范围；若估计被截断，可能无法在该范围到达阈值。

## 7. 返回仿真输入，执行第二轮

```bash
export ROUND2="$EXP_ROOT/manual-results/round2"
bash "$EXP_ROOT/manual-workflow/run_round.sh" \
  "$MIX_TRACE" "$FREQ2" "$SM_SPEC" "$ROUND2"
cat "$ROUND2/performance.json"
```

使用相同任务、SM 分配、floorplan、环境温度，仅调整前三个时钟域频率。每轮都重新生成性能、功耗及热分析结果。

## 8. 用两轮实测温度改进下一次估计

在局部线性近似下，`slope=(T2-T1)/(f2-f1)`，然后 `f3=f2+(T_limit-T2)/slope`。它吸收了两个点之间的固定背景影响，但遇到访存瓶颈变化、峰温位置变化或非单调情况仍需重新验证。

```bash
export FREQ3="$(python3 "$EXP_ROOT/manual-workflow/suggest_frequency.py" \
  --steady "$ROUND2/temp.grid.steady" --grid --freq "$FREQ2" \
  --previous-steady "$ROUND1/temp.grid.steady" --previous-freq "$FREQ1" \
  --threshold-c "$THRESHOLD_C" --ambient-c 45 --min-mhz 300 --max-mhz 4000)"
bash "$EXP_ROOT/manual-workflow/run_round.sh" \
  "$MIX_TRACE" "$FREQ3" "$SM_SPEC" "$EXP_ROOT/manual-results/round3"
```

达到容差（例如 `abs(T_peak-T_limit)<=0.5°C`）后停止。如果目标是最高可用频率，选择满足 `T_peak<=T_limit` 的频率。若已找到一低一高分别低于/高于阈值，采用区间二分，每次实际复跑比持续外推更稳妥；只有测得单调性且阈值落在区间内时才适用。

## 扩展与已有工具

- 原启动器支持手动数量：`run_simulations.py -B rodinia_2.0-ft-backprop-bfs-heartwall -C QV100-Lyhong_SIM-1132MHZ -T ./hw_run/rodinia_2.0-ft/9.1 -N manual1 --stream-sm-count '1:43,2:9,3:28'`。显式传该参数跳过自动 Alg1/Alg2。注意原启动器还会做 trace 特征分析并写共享输出，结果目录依 CUDA 版本确定。
- 如果要使用交错放置，可用 Alg2 生成 `sm_map.csv` 后将列表写成 `1:0|3|...`。单一 SM 列表目前有歧义，因此本示例只用数量格式；改变具体物理位置会改变温度，所有轮次应固定位置。
- 没有真实 GPU 也可运行已有 trace 的仿真；获取新的真实 trace 才需要对应 CUDA 应用和 GPU。
- 逐任务独立调频、温度驱动自动 DVFS、完整非 SM 热功耗映射、运行时任务迁移均需要额外实现。
