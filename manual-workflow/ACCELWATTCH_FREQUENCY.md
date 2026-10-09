# 当前 AccelWattch 的频率—功耗关系

本说明以当前项目 SASS SIM、逐 SM 路径为依据。先前 `f^3` 反解保留为用户假设演示，不代表当前实现的频率公式；矩阵脚本默认指数已改为 1，但线性模型仍有固定活动率等前提。

后续八频率实验已完成，见 [实测报告](../manual-results/power-frequency-20260929/REPORT.md)。在固定三任务和 SM 映射下，800–4000 MHz 的平均功耗呈轻微下凹曲线，二次经验模型的验证误差低于线性模型；源码关系与经验拟合应区分。

## 源码路径

1. `src/gpgpu-sim/power_interface.cc:46` 将 `config.get_core_freq()/1000000` 作为 MHz 传给功耗初始化。
2. `src/accelwattch/gpgpu_sim_wrapper.cc:210–213` 设置采样周期 S、`target_core_clockrate`、核心 `clock_rate`。
3. `src/accelwattch/processor.cc:545–546` 逐 SM 设置 `executionTime = S/(clock_rate*1e6)`。
4. `src/accelwattch/core.cc:3419–3422` 等位置计算事件能量：单次读/写能量乘对应事件数。流水线能量另依据 duty cycle 和周期数构造。
5. `src/accelwattch/gpgpu_sim_wrapper.cc:1421–1457` 将逐 SM 组件采样能量除以 executionTime，得到 W；L2/MC/NoC/DRAM 也在同一采样时间尺度下换算。
6. `src/accelwattch/gpgpu_sim_wrapper.cc:618–624` 设置逐 SM 空闲功率为“空闲标志 × XML idle_core_power”，没有显式乘频率。
7. `src/accelwattch/gpgpu_sim_wrapper.cc:1108–1239` 用执行类别、活跃 lane 和活跃核心比例选择/计算 STATICP；CONSTP 从 XML 常量获取。它们不能直接当作所有频率下数值不变的常数，因为活动状态可能变化。

所有源码路径相对于 `gpu-simulator/gpgpu-sim`。能量系数来自 McPAT/CACTI 的组件模型及 XML 缩放，不是仅由频率得出的通用常数。改变频率还可能影响组件初始化中的时间/延迟参数，须检查系数是否保持一致。

## 从实现得到的关系

f 使用 Hz，S 为采样核心周期数，第 i 个 SM 的第 c 个组件采样能量为 E_ic：

`Delta_t = S/f`

`P_ic = E_ic / Delta_t = f E_ic/S`。

对于能量按事件线性计数的部分：

`E_ic = sum_r e_icr * N_ir`，因此

`P_ic(f) = f * sum_r e_icr(f) * [N_ir(f)/S]`。

加上流水线、空闲及其他实际项，可概括为：

`p(f) = f q(f) + b(f)`。

q(f) 包含每周期事件量、duty cycle、lane 活跃情况、事件能量系数及校准因子；b(f) 包含按源码单独计算的空闲/常量/静态等功率。两者均可能随频率和运行活动状态变化。

因此实现没有唯一的全局 `P=a*f` 或 `P=a*f^3` 公式。只有固定电压、事件能量系数不变、每周期活动率和背景项保持不变时，才可局部近似 `p(f)=b+q*f`。

全程平均功率应按各采样实际时间加权。当前 steady HotSpot 对输入样本简单平均，等时间样本时一致；不满采样区间等边界处理需另外核对。

## 电压路径

开启 `dvfs_enabled` 时，wrapper 对相应组件应用动态功率 `Vratio^2`、IDLE_COREP/STATICP 应用 `Vratio`；但当前 SASS SIM 的 `mcpat_cycle` 直接调用 `set_model_voltage(1)`，没有 V(f) 自动关系。逐 SM 导出路径还需分别检查电压缩放实际应用到的数组，不能仅看到总组件数组有缩放就认为每个导出 SM 已一致缩放。

所以不能从当前配置推出 `V proportional to f`，也不能据此推出立方功耗律。当前手动示例 `dvfs_enabled=0`，采用固定电压设置。

## 如何结合 G/B 反推

若先冻结 q、b，得到解析估计：

`u=solve(G,B*b)`，`v=solve(G,B*q)`，

`T_j(f)=Ta+u_j+v_j*f`，

`f_candidate=min_j[(Tlimit-Ta-u_j)/v_j]`，仅考虑 v_j>0，并检查背景是否已经超过阈值。

这里 q 使用 W/Hz；矩阵脚本使用归一化 `p0+k*(f/fref)`，因此返回式额外乘 fref。当前功耗 trace 只提供各块合计，不能精确拆出 q/b。多频率 `--fit-offset --exponent 1` 可作局部校准，但它拟合的是输出，不是直接导出 AccelWattch 内部系数。

若要求与完整仿真一致，必须使用：

`F(f)=max_j[Ta + solve(G,B*p_AccelWattch(f))_j]-Tlimit`。

每个候选 f 重新运行性能仿真得到活动统计，并通过 AccelWattch 获得 p(f)，再用已验证热矩阵求温度。找到阈值两侧频率、检查局部单调后，以二分/割线求 F(f)=0。G/B 只需加载、分解一次，搜索中不必每次重新调用 HotSpot。若没有区间跨越阈值，则该频率范围不可达。

这是对完整实现的数值反求；单点功耗不能唯一识别未来频率下的活动率，因此不能仅靠一个已有温度点给出保证正确的闭式频率。

## 当前建议

- 之前约 4717.95 MHz 的 80°C 结果仅属于纯立方假设下的热模型解，不作为当前 AccelWattch 调频结论。
- 用已有三频率数据作线性+背景拟合，可作为下一次实际运行的初值；不应将其在远离样本的高频外推称为准确结果。
- 若目标是精确闭环，选择允许的频率范围和温度阈值，然后用实际 AccelWattch 输出求根。
- 当前热功耗映射未覆盖全部 GPU 功率，此限制同样适用于源码关系和频率反解。
