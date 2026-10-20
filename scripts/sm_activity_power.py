#!/usr/bin/env python3
# 作用：算法3、功耗模型拟合与预测共用的活动特征定义及每核心周期活动计算。
# 输入：包含 window_cycles、counter_*、frequency_hz、idle_fraction 的 DataFrame。
# 输出：activity_rates 返回事件/周期与 pipeline duty；features 返回完整拟合特征。
# 执行单元计数已含 XML 标定，不重复标定；派生 cache miss 保留原模型的有符号值。
import numpy as np

COUNTERS = ['TOT_INST','FP_INT','IC_H','IC_M','DC_RH','DC_RM','DC_WH','DC_WM',
            'TC_H','TC_M','CC_H','CC_M','SHRD_ACC','REG_RD','REG_WR','NON_REG_OPs',
            'INT_ACC','FPU_ACC','DPU_ACC','INT_MUL24_ACC','INT_MUL32_ACC','INT_MUL_ACC',
            'INT_DIV_ACC','FP_MUL_ACC','FP_DIV_ACC','FP_SQRT_ACC','FP_LG_ACC','FP_SIN_ACC',
            'FP_EXP_ACC','DP_MUL_ACC','DP_DIV_ACC','TENSOR_ACC','TEX_ACC']
DIRECT = COUNTERS[:17]
GROUPS = {'FPU_WEIGHTED': COUNTERS[17:19], 'SFU_WEIGHTED': COUNTERS[19:]}
FEATURES = DIRECT + list(GROUPS) + ['PIPE_A','idle_fraction']


def activity_rates(frame):
    cycles = frame.window_cycles.to_numpy(dtype=float)
    rates = frame[['counter_'+c for c in DIRECT]].to_numpy(dtype=float)/cycles[:,None]
    grouped = [frame[['counter_'+c for c in members]].sum(axis=1).to_numpy(dtype=float)/cycles
               for members in GROUPS.values()]
    return np.column_stack([rates,*grouped,frame.counter_PIPE_A.to_numpy(dtype=float)])


def features(frame):
    rate = activity_rates(frame)
    return np.column_stack([rate*(frame.frequency_hz.to_numpy()/1e9)[:,None],
                            frame.idle_fraction.to_numpy()])
