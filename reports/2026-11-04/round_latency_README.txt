完整轮次均值 = decode wall time / 正式采集轮数。
每个请求的 decode window 从首个 AR/kv_pack 或 JetSpec/draft_prepare 开始，到该请求返回后紧随的 cudaDeviceSynchronize 完成。
因此包含各阶段、阶段间空隙及最后一轮的 GPU 尾部，不包含 prefill，不把 CPU NVTX 与 GPU kernel 时长相加。
另外保留相邻轮次起点间隔（排除每个请求的最后一轮），供交叉核对。
这些是 instrumented batch_reference trace，cap=128，所有 BS 都没有 CUDA Graph；不是 headline benchmark。
BS=1只采两条输入，BS=16采32条；AIME25只有30条。序列提前结束后退出，padding 和输入子集随 BS 改变。
trace 的 raw accept length 可能含输出截断前的候选 block；effective acceptance 则按实际输出的 decode tokens / 活跃 row-rounds 定义。
benchmark_cap512_* 字段来自另外的无 profiler 运行，不能与 trace latency 混合重建速度比。
gpu_kernel_busy_fraction 是 decode window 内设备 kernel 区间并集 / wall time，不包含纯 memcpy 活跃时间。
它只表示时间上的 kernel duty fraction，不是 SM/Tensor 饱和度，也不能证明 compute bound。
