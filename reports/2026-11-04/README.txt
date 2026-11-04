这些数据来自已经完成的 RTX 5090 / Qwen3-8B BF16 batch_reference trace。
所有 BS 使用相同 eager SDPA / dense padded KV 实现，无 CUDA Graph；不是 vLLM 或原生优化 engine。
BS=1/2/4/8/16 分别采集最多 2/4/8/16/32 条输入，即最多两批；AIME25 只有30条。
输出上限128，tree budget=128、depth=15。输入子集和 padding 随 BS 改变，不能当成严格固定输入的 BS 消融。
完成的序列会退出；mean_active_batch 是正式采集中真实的活跃 batch 均值，已剔除 warmup 记录。
host_ms_per_call 是 CPU NVTX 范围长度，包含 CUDA API 和等待；不是纯 CPU 计算时间。
gpu_kernel_ms_per_call 是通过 launch correlation 归因到该 NVTX 的 GPU kernel 时长之和，包含 NVTX pop 后执行的 kernel。
CPU/GPU 时间可能重叠，不能相加；memcpy 单列。sync_api_ms 仅统计显式命名的 CUDA Synchronize API，
不包括其他可能阻塞的 API，也不能将 host-sync 当作纯 CPU 时间。
每轮表排除 prefill；prefill 按每条输入顺序执行，不能拿其总时长比较 BS 扩展。
kernel 类别是按名称划分的 GEMM/GEMV、attention 和其余 kernel，未测 FLOPs 或 DRAM bytes，不能据此确认 compute bound。
profiler 有开销；headline 吞吐应查看 reports/2026-11-03/benchmark_results.csv 的独立运行。
查看 nsys：打开 NVTX、CUDA API、GPU kernel 行，定位 JetSpec/verify_forward 和随后的 accept；
accept 的长 cudaStreamSynchronize 常在等待先前异步提交的 verify，并不代表 accept 自身 GPU 工作很重。
