# 2800 MHz 的个别 SM 峰值与活跃因子

口径：Core/ICNT/L2 为 2800 MHz，DRAM 为 850 MHz，`-dvfs_enabled 0`，每 500 个核心周期输出一行。两个原始运行分别是单独 backprop 和 backprop+bfs+heartwall 混合负载。采样行号从 0 开始。

## 尖峰分项

为定位分项，临时输出了模型内部 `sample_Per_cmp_pwr[sm][component]`，在相同输入下重跑两次。重跑的 `lyhong_power_report.log` 和 `lyhong_SM_power_report.log` 与原始日志逐字节一致。诊断分项已经整理成 [anomaly_components.csv](anomaly_components.csv)；各分项相加与原始 SM 列的最大差异小于 0.0005 W，即原始 SM 列保留三位小数的舍入误差。临时诊断代码已从源文件移除。

|运行|采样行|SM|SM 合计 W|L1 D-cache 分项 DCP W|DCP 占比|其余分项合计 W|全 GPU 的 DCP W|DCP 均分的活跃 SM 数|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|backprop|29|16|1.020|0.493|48.3%|0.527|38.912|79|
|backprop|30|16|9.100|8.669|95.3%|0.431|8.669|1|
|mixed|32|78|1.298|0.614|47.3%|0.684|26.402|43|
|mixed|33|78|7.447|7.331|98.4%|0.116|7.331|1|

“活跃 SM 数”由该行 `lyhong_SM_power_report.log` 的汇总 DCP 除以目标 SM 的 DCP 得到，分别约为 79、1、43、1，和源码中把全局 L1 访问数除以采样时活跃 SM 数的算法吻合。尖峰行里 **整个 GPU 的 DCP 其实下降**，目标 SM 的寄存器、流水线和计算单元等其余功耗也没有同步上升。因此 9.100 W 和 7.447 W 主要是 L1 功耗归属集中造成的局部伪峰，不能解释成这两个 SM 突然有极高的真实切换活跃因子。混合实验的 SM78 属于 backprop 的 43 个 SM；两个尖峰都在 backprop 即将结束时。

源头位于 `gpu-simulator/gpgpu-sim/src/gpgpu-sim/power_interface.cc`：`set_Per_l1cache_power` 收到的是整个 GPU 的 L1 hit/miss 数和当前周期活跃 SM 向量。`gpu-simulator/gpgpu-sim/src/accelwattch/gpgpu_sim_wrapper.cc` 的这个函数以 `active_cnt` 平均分配全局计数；因此 500 周期窗口尾部只剩一个活跃 SM 时，先前窗口内所有 SM 的 L1 活动都算到该 SM。其他以相同方式按当前活跃 SM 数平摊的缓存分项也应排查。根本修复方向是在统计阶段保留每 SM 的 L1 访问计数，并用同一采样窗口的每 SM 计数建模，避免用窗口末尾的活跃 SM 集合分摊整段窗口活动。

## 能否得到活跃因子 a

现有报告 **没有直接输出标量 a**。模型输入实际上是各采样窗口的指令、寄存器、缓存、执行单元访问计数和流水线占空比；代码内部有 `sample_Per_perf_counters`，但当前 `lyhong_power_report.log` 只输出各 SM 功耗合计，`lyhong_SM_power_report.log` 只输出跨 SM 的分项汇总。`gpgpu_inst_stats.txt` 也不是完整的逐 SM、逐窗口活跃因子日志。虽然有 `-power_trace_enabled` 配置项，本项目当前 `gpgpu_sim_wrapper.cc` 初始化时把 `g_power_trace_enabled` 强制设为 false，且该 trace 使用的 `sample_perf_counters` 是汇总值，不是完整逐 SM 活动向量。

在固定电压下可以从每 SM、每窗口的**动态分项**估计 `aC`，例如 `C_eff,sw = P_dyn/(f V²)`；同一 SM、同一频率和电压下两个 benchmark 的 `P_dyn` 比值可作相对活动强度指标。若只用当前合计日志，至少排除完全空闲窗口的 `IDLE_COREP`；不要对活跃窗口机械减去 0.283 W，因为 `IDLE_COREP` 在活跃窗口本来就是 0。还须排除或修复上述 L1 归属伪峰。这个比值不是物理上唯一的 a：不同模块有各自的有效电容、每次访问能耗和活动计数，不同 benchmark 的模块构成也不同；单凭总功耗、频率和电压无法把 a 与 C 唯一分开。

`sim.log` 的 `Lyhong_print: sum_active_sm` 也不是 a。这里 `lyhong_active_sm` 是调用时当前周期的活跃 SM 指示向量，再除以 500 周期采样间隔；例如 80 个 SM 都活跃时显示 0.160，而不是“活跃因子 16%”。它只能帮助判定采样时有多少 SM 被标为活跃，不能代替整个窗口的访问计数或翻转率。

更可靠的下一步是在每个采样窗口直接导出逐 SM 的 `REG_RD/WR`、`FP/DP/INT/TENSOR`、L1 hit/miss、shared-memory 访问次数和 pipeline duty cycle，并按模块计算访问次数/500 周期，再用 AccelWattch 的每次访问能量和实际模型电压分别计算加权 `a_r C_r`。对峰值拟合或稳态热输入，应先用真实逐 SM L1 计数消除局部伪峰，并采用同一时刻的完整功耗向量。
