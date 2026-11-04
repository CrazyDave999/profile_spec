原生优化 engine，RTX 5090 / Qwen3-8B BF16 / BS=1 / width=7 / max_depth=15。
budget 包含 root；已测64/128/256，没有16/32的 GPU 结果。
verify GPU 时间按 launch correlation 归因；host NVTX 包含异步提交，不是 GPU 执行长度。
完整轮次从第一次 draft_forward 到生成后 cudaDeviceSynchronize 完成，再除以轮数，排除 prefill。
短 trace 每组2条、cap256；benchmark 最多32条、cap512；不可混合两组数据计算精确速度比。
此实现将 verify 行数固定到预算对应 graph bucket；降低实际节点数而不改变 bucket 未必降低主要 GEMM 成本。
只分析已有结果，没有运行新的 GPU benchmark。
