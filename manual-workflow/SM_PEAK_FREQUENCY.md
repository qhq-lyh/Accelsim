# 每个 SM 的峰值功耗拟合

运行目录为项目根目录。功耗输入仅使用 `lyhong_power_report.log`：取 `SM_0` 到 `SM_79` 的列最大值，忽略 L2/HBM 列。

```bash
python3 manual-workflow/explore_sm_peak_frequency.py \
  --out-dir manual-results/sm-peak-frequency-20260929 \
  --frequencies 800,1132,1400,1698,2200,2800,3396,4000 \
  --benchmarks all --jobs 3

python3 manual-workflow/fit_sm_peak_frequency.py \
  --data-dir manual-results/sm-peak-frequency-20260929 \
  --mixed-dir manual-results/power-frequency-20260929
```

第一条命令用已存在的 11 个单 benchmark traces 采样，所有 80 个 SM 均可用于该负载。可通过 `--benchmarks backprop,bfs,hotspot` 限定负载。每个仿真保存到 `负载/频率MHz/`，包含实际配置、`sim.log`、原始功耗日志和正常完成标记 `complete.json`。已完成的仿真支持重用；若需要改变输入配置，使用新的输出目录，避免混用实验。共同配置保存在 `inputs/`，SHA256 和固定参数保存在 `manifest.json`。

频率仅更改每次仿真的配置副本：Core=ICNT=L2=f，DRAM=850 MHz。关闭 DVFS。采样间隔统一为 500 个核心周期。基准配置保持原样。

第二条命令逐负载、逐 SM 拟合，且计算各频率、各 SM 在所有已采样负载中的最大峰值包络。可省略 `--mixed-dir` 仅使用新采样的单 benchmark。混合负载数据作为独立条件，其 SM 分配是 43/9/28；它不代表其他混合分配。

输出：

- `sm_peak_samples.csv`：单 benchmark 的原始峰值，含 SM 编号、频率、W、达到峰值的行号（从 0 开始）、采样数、性能信息及日志路径。
- `sm_peak_all_conditions.csv`：加入混合负载及跨负载包络后的拟合输入。
- `sm_peak_envelope.csv`：每个频率、每个 SM 的跨负载最大值，包含获胜负载与原始日志路径。
- `fits/models.csv`、`fits/models.json`：每个 SM 的模型类型、系数和验证误差，公式的频率单位为 GHz，功耗单位为 W。
- `fits/candidate_validation.csv`：线性、二次、三次及带偏置 f³ 候选模型的验证结果。
- `fits/负载名.png`：横坐标 MHz，纵坐标 W，颜色为 SM 编号，点为观测，线为拟合。
- `fits/all_sm_benchmarks.png`：80 个小图，每图一个 SM，颜色区分负载。
- `fits/REPORT.md`：实验口径、误差汇总与限制。

1400 和 2800 MHz 留出验证。其余六点通过留一交叉验证选择模型，误差相差不超过日志量化相关的 0.002 W 时取参数更少的模型，最终系数使用全部八点重新拟合。没有预设所有 SM 都满足同一指数。留出误差较大的曲线不能当作准确的频率反推公式，应在拟议频率附近重新仿真验证。

这里的峰值是日志采样窗口功耗的最大值，不是瞬时功耗。固定周期数的窗口在不同频率下覆盖不同实际时间（800 MHz 为 0.625 μs，4000 MHz 为 0.125 μs）；末尾未满一个窗口的活动可能未输出。这会使短 kernel 的峰值出现明显变化。不同 SM 的峰值也未必同时发生，跨负载包络还可能来自不同 benchmark；只能表示已采样负载范围的功耗包络场景，不能直接当作真实稳态功耗快照。

拟合系数是这些 traces、调度与采样条件下的经验系数，不直接等同于物理电容等参数。功耗日志中的 SM 列未分摊 GPU 全局静态功耗。
