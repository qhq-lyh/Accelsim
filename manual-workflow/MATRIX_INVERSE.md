# 用已导出的热矩阵反求频率

**修正：实际频率关系应以当前 AccelWattch 源码为准，见 [ACCELWATTCH_FREQUENCY.md](ACCELWATTCH_FREQUENCY.md)。下文立方结果是显式假设的演示；当前脚本默认指数为 1，只有冻结活动率与能量系数时才对应源码的线性近似。**

## 已核对的矩阵

当前 `hotspot-link/model_extract` 中：

- `Amatrixcolptr/rowind/nzval` 是 CSC 格式的热导矩阵 **G**，16396×16396，单位 W/K；这里的 Amatrix 不是动态状态矩阵 `-C^-1 G`。
- `Bmatrix` 为 16396×83 的功耗注入矩阵，83 列按有功耗层的 floorplan 单元顺序排列；当前为 SM_0…SM_79、L2_cache、HBM2_0、HBM2_1。
- `Lmatrix` 文件实际存储 L 的转置，为 16396×86；读取后 L 为 86×16396，提供按面积权重映射的块观测。
- B/L 文件最后一行是矩阵维度，不是非零元；索引从 0 开始。
- 当前没有 `Cmatrix`，因为导出的是稳态模型。稳态反解不需要热容 C。
- 64×64 网格中，前 4096 个节点为活性硅层。阈值默认施加在这一层所有节点的最大温度，而不是模块中心温度。

来源核对：本机 `/home/lyhong/Downloads/hotspot_with_model_extract/temperature_grid.c` 的 `dumpBL` 和 `model_extract/model_extract.py` 与当前导出文件格式一致。对当前真实功耗的矩阵重建与 HotSpot 校验也一致。

## 方程和反解

令 x 为相对环境的温升，p 为块功率（W）：

`C dx/dt + G x = B p`。

稳态 `G x = B p`，因此 `x = solve(G, B p)`。观测块温度为 `Tobs = Ta + L x`，但最大网格温度直接使用活性层 x，避免平均/中心映射漏掉热点。G 不显式求逆；稀疏 LU 分解后解方程。

用户的 `P=a C_e f^3` 中，C_e 是等效电学电容，不是热矩阵 C。可以合并系数 k=a C_e。为避免 Hz^3 的巨大数字，用参考频率 fref 归一化：

`p(f) = p0 + k (f/fref)^n`，当前默认 n=1；下文立方假设示例显式选择 n=3，k 单位 W。

单点纯立方假设时，p0=0、k=pref；每个块的 a*C_e 系数可用 `pref/fref^3` 得到，但 a 和 C_e 无法仅由该数据分别识别。

背景项存在时：

`u = solve(G, B p0)`，`v = solve(G, B k)`，

`Tj(f) = Ta + uj + vj (f/fref)^n`。

所有活性层节点满足 `Tj <= Tlimit`，允许的最大统一频率为：

`f* = fref * [min_j ((Tlimit-Ta-uj)/vj)]^(1/n)`，只考虑 vj>0。

若背景项已经超过阈值，单靠降频无解；若全部 vj=0，不能从温度反推频率。包含背景项时，限制频率的节点可能与参考频率下最热的节点不同，因此脚本检查所有网格节点。

纯立方、统一空间缩放时，该式退化为：

`f* = fref * [(Tlimit-Ta)/(Tmax_ref-Ta)]^(1/3)`。

## 运行命令

依赖 NumPy、SciPy，本机已安装。示例沿用 80°C 阈值和 45°C 环境温度；阈值应按实验需求修改。

```bash
cd /home/lyhong/Downloads/Accel-Hotsim
export EXP_ROOT="$PWD"
export P3396="$EXP_ROOT/sim_run_12.8-3.3G-compare/backprop-bfs-heartwall-rodinia-2.0-ft/backprop_bfs_heartwall/QV100-Lyhong_SIM/lyhong_power_report.log"

python3 manual-workflow/infer_frequency_matrix.py \
  --matrices hotspot-link/model_extract \
  --sample "3396=$P3396" \
  --exponent 3 --threshold-c 80 --ambient-c 45 \
  --out-dir manual-results/my-matrix-inverse
```

输出目录必须不存在。已有计算结果在 `manual-results/matrix-inverse/cubic-single/`，可以直接查看，无需重复执行：

```bash
cat manual-results/matrix-inverse/cubic-single/result.json
cat manual-results/matrix-inverse/cubic-single/frequency_mhz.txt
```

如限定 4000 MHz，再加 `--max-mhz 4000`。脚本同时报告原始解、限幅后的频率与温度、阈值是否可达，避免把限幅结果误称为达到阈值。

输出文件：

- `result.json`：矩阵维度、频率、最高温度、限制节点、样本拟合误差及可达性。
- `frequency_mhz.txt`：候选频率，供下一轮输入。
- `power_coefficients.csv`：每块的固定功率、参考频率缩放系数及预测功率。
- `predicted_power.ptrace`：该候选频率功耗模型的预测输入；可单独交给 HotSpot 校验热矩阵反解。
- `predicted_active_grid.csv`：活性硅层的预测摄氏温度。
- `predicted_L_observations_c.csv`：L 的面积权重观测，和 HotSpot 当前 `center` 映射口径不同。

## 两种验证必须区分

1. **热矩阵验证**：把 `predicted_power.ptrace` 输入 HotSpot，检查是否得到目标温度。它验证 G/B 反解及功耗映射。
2. **功耗模型验证**：把候选频率输入 Accel-Sim，重新获得实际功耗，再输入 HotSpot。它验证所假设的 `p(f)` 是否对应当前性能/功耗仿真。

已经完成第 1 种验证。纯立方单点假设下，3396 MHz 参考结果对应最高 58.05299°C，反解 80°C 得到 **4717.95345 MHz**；预测功耗再运行 HotSpot 得到 **80.00°C**。

**这不等于已证明 Accel-Sim 在 4717.95345 MHz 会输出该功耗。** 当前源码 `get_core_freq()` 返回 unsigned Hz，超过约 4294.967 MHz 不适合直接传给现有功耗初始化接口；`run_round.sh` 为此限制到 4294 MHz。另外，80°C 在已测频率范围外，任何估计都是外推。

若限频 4000 MHz，纯立方模型预测最高 **66.32981°C**，80°C 在该范围不可达。60°C 的纯立方示例解位于范围内，已有结果在 `cubic-single-60`：

```bash
export FREQ_NEXT="$(cat "$EXP_ROOT/manual-results/matrix-inverse/cubic-single-60/frequency_mhz.txt")"
# 准备 CUDA 环境，定义 MIX_TRACE 和 SM_SPEC，见 README.md 后再执行：
bash "$EXP_ROOT/manual-workflow/run_round.sh" \
  "$MIX_TRACE" "$FREQ_NEXT" "$SM_SPEC" \
  "$EXP_ROOT/manual-results/matrix-verified-round"
```

本次没有执行新一轮 Accel-Sim。真正闭环仍需这一轮实际功耗校验。

## 多频率拟合与当前模型的适用性

可重复提供 `--sample MHz=ptrace`，用每个块的平均功耗拟合系数。

`--fit-offset` 使用非负最小二乘拟合 p0>=0、k>=0，需要至少两个不同频率。默认零背景时，拟合纯幂次系数。`--scale sm` 只让 SM 功耗随频率变化，L2/HBM 保持最高频率样本中的值；这仍是用户指定的近似，不能推断真实哪些功耗固定。

```bash
export P1132="$EXP_ROOT/sim_run_12.8-1.1G-compare/backprop-bfs-heartwall-rodinia-2.0-ft/backprop_bfs_heartwall/QV100-Lyhong_SIM/lyhong_power_report.log"
export P1698="$EXP_ROOT/sim_run_12.8-1.6G-compare/backprop-bfs-heartwall-rodinia-2.0-ft/backprop_bfs_heartwall/QV100-Lyhong_SIM/lyhong_power_report.log"

python3 manual-workflow/infer_frequency_matrix.py \
  --sample "1132=$P1132" --sample "1698=$P1698" --sample "3396=$P3396" \
  --exponent 3 --fit-offset --threshold-c 80 \
  --out-dir manual-results/my-cubic-fit
```

已有三组热映射功率为 26.9403、30.4063、39.6347 W，频率比 1:1.5:3，而纯立方应为功率比 1:3.375:27；实际仅 1:1.129:1.471。因此当前固定电压仿真不支持把全部功耗视为纯立方。

三组逐块拟合后，立方+背景模型的功耗向量相对误差约 6.5%、6.1%、0.4%；线性+背景模型约 0.7%、0.8%、0.1%。线性模型在已测区间更符合数据，但这也不保证远距离外推准确。`--exponent 1 --fit-offset` 可以使用线性模型反求。

`f^3` 通常来自动态功耗 `alpha*C_e*V^2*f` 加上近似 `V∝f`；当前 SASS SIM 路径没有这种自动电压-频率联动。若要研究真实 DVFS 的立方关系，还需提供 V(f)、活动变化、静态/漏电功耗及模型校准。

上述功率与温度仅对应已导出的 SM/L2/HBM 热功耗范围，未包含全部 GPU 功耗。导热参数、floorplan、环境或功耗列发生变化时，需要重新导出/核对矩阵。自然对流或温度相关漏电导致模型非线性时，固定 G 的解析式不再是完整解。
