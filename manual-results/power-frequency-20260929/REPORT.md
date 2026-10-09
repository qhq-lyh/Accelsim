# 实测频率—功耗关系

实验仅修改每轮 gpgpusim.config 中 gpgpu_clock_domains 的前三项，三者保持相等；DRAM 固定 850 MHz。原基础配置未改动。

固定 backprop+bfs+heartwall trace、80 个 SM、原有物理映射（43/9/28 个）、XML、电压设置和 500 核心周期功耗采样。

P 是已采样窗口的算术平均 W；f 以下使用 GHz，即配置中的 MHz/1000。各窗口时间相同；末尾不足 500 周期未计入该平均。该模型不是所有任务/SM 分配通用公式。

| f (MHz) | GPU total (W) | SM/L2/HBM mapped (W) | Cycles | Unsampled tail (cycles) |
|---:|---:|---:|---:|---:|
| 800 | 61.993797 | 25.014010 | 49173 | 173 |
| 1132 | 64.070495 | 26.940293 | 49650 | 150 |
| 1400 | 65.797720 | 28.562070 | 50090 | 90 |
| 1698 | 67.717296 | 30.406317 | 50627 | 127 |
| 2200 | 70.892446 | 33.334087 | 51678 | 178 |
| 2800 | 74.477587 | 36.624333 | 52825 | 325 |
| 3396 | 77.783951 | 39.634654 | 53993 | 493 |
| 4000 | 81.012845 | 42.596645 | 55315 | 315 |

1400、2800 MHz 在初始模型拟合时排除，用于独立频率点验证。另对训练点进行留一验证。若简单模型的验证最大相对误差在最佳模型的 0.05 个百分点内，优先简单模型。选型后使用全部频率点重拟合最终系数。

## Total_GPU_Power

选中模型：**quadratic**。

`P = 56.396615886 + (7.081332217) f + (-0.230452836) f²`

最终全部样本最大偏差 0.079605 W，最大相对偏差 0.1284%。

| Candidate | Held-out max error (%) | Held-out RMSE (W) | Training LOO max error (%) |
|---|---:|---:|---:|
| affine | 0.4697 | 0.247495 | 0.7587 |
| quadratic | 0.1043 | 0.072267 | 0.2682 |
| saturating | 0.1202 | 0.081480 | 0.3090 |
| offset_cubic | 4.5512 | 2.396929 | 8.2437 |

## mapped_power_w

选中模型：**quadratic**。

`P = 19.714984887 + (6.721746765) f + (-0.249920534) f²`

最终全部样本最大偏差 0.081577 W，最大相对偏差 0.3261%。

| Candidate | Held-out max error (%) | Held-out RMSE (W) | Training LOO max error (%) |
|---|---:|---:|---:|
| affine | 0.9767 | 0.252952 | 1.6554 |
| quadratic | 0.2958 | 0.072196 | 0.6852 |
| saturating | 0.3257 | 0.083081 | 0.8360 |
| offset_cubic | 8.6468 | 2.239303 | 16.5225 |

## 解释与适用范围

源码仍是组件能量/采样时间；升频会改变每周期活动数、等待、活跃 SM 与静态功率选择，所以并不保证严格恒定 q、b。拟合模型对本实验 800–4000 MHz 范围内的输出作经验近似，不用于范围外保证温度或功耗。

不能把仅 SM/L2/HBM 的 mapped 功率当成完整 GPU total；后者还包含 MCP、NoC、CONSTP、STATICP 等。

CONSTP 为 XML 常量 32.3252 W；STATICP 的平均值随活动状态变化。IDLE_COREP 分项汇总日志存在遗漏累加而显示 0 的问题，但 Total_SM_Power 和逐 SM 功耗包含空闲项。本实验没有修改源代码，也没有用该错误分项拟合。

矩阵温度反解需要逐块功耗，而不是只将 GPU total 标量放入 B。block_quadratic_models.csv 同时提供 83 个热映射块的二次系数（f 使用 GHz）；它们仍只适用于本实验范围，应检查逐块拟合误差并用候选频率复跑验证。

复现：`python3 manual-workflow/explore_power_frequency.py --out-dir <新目录> --jobs 2`，然后 `python3 manual-workflow/fit_power_frequency.py --root <结果目录>`。保存的 inputs/ 与 manifest.json 给出了配置、trace、二进制校验信息。
