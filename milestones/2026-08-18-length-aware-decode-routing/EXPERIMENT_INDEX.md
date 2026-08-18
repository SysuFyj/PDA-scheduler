# 实验与结果索引

## 实验 A：持续长请求 Burst

- 设计：`experiments/1p3d_burst_20260817/EXPERIMENT_PLAN.md`
- 执行：`experiments/1p3d_burst_20260817/run_burst_experiment.py`
- 分析：`experiments/1p3d_burst_20260817/analyze_results.py`
- 最终报告：`outputs/1p3d_burst_20260817/FINAL_EXPERIMENT_REPORT.md`
- 有效性审计：`outputs/1p3d_burst_20260817/VALIDITY_AUDIT.md`
- 正式汇总：`outputs/1p3d_burst_20260817/formal-summary.json`
- 容量扫描：`outputs/1p3d_burst_20260817/peak-scan-summary.json`
- 表格：`outputs/1p3d_burst_20260817/tables/`
- 图：`outputs/1p3d_burst_20260817/figures/`

## 实验 B：异构长度微突发

- 设计：`experiments/1p3d_microburst_20260817/EXPERIMENT_PLAN.md`
- 执行：`experiments/1p3d_microburst_20260817/run_microburst_experiment.py`
- 请求级抢占观测：`experiments/1p3d_microburst_20260817/vllm_preemption_instrumentation.py`
- 分析：`experiments/1p3d_microburst_20260817/analyze_results.py`
- 最终报告：`outputs/1p3d_microburst_20260817/FINAL_EXPERIMENT_REPORT.md`
- 有效性审计：`outputs/1p3d_microburst_20260817/VALIDITY_AUDIT.md`
- 正式汇总：`outputs/1p3d_microburst_20260817/formal-summary.json`
- 表格：`outputs/1p3d_microburst_20260817/tables/`
- 图：`outputs/1p3d_microburst_20260817/figures/`

## 未纳入阶段结论

- `outputs/1p3d_burst_20260817/excluded_32k_burst/`：RR 长时间不能排空的探索性极端配置。
- `outputs/1p3d_burst_20260817/excluded_16k_burst/`：仍过于极端的探索配置。
- `outputs/1p3d_microburst_20260817_excluded_attempt_path_match/`：服务启动前的路径匹配错误，没有测量请求。
- 其他 `experiments/` 目录和本地 `outputs/` 不自动视为本阶段有效证据。
