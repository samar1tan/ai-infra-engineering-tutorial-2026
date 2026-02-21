# Interview Preparation

Seven-round interview preparation for top AI infrastructure positions.

## Interview Matrix Overview

| Round | Focus Area | Course Mapping | Duration |
|-------|------------|----------------|----------|
| 1 | Low-Level Systems Coding (C++/CUDA & Concurrency) | SYS-601 | 60 min |
| 2 | GPU Architecture and Kernel Design | SYS-601, SYS-604 | 60 min |
| 3 | Distributed Training Architecture | SYS-602 | 60 min |
| 4 | High-Performance Networking (Collectives & RDMA) | SYS-603 | 60 min |
| 5 | High-Throughput Inference System Design | SYS-604 | 60 min |
| 6 | Resilience, Scheduling, and Orchestration | SYS-605 | 60 min |
| 7 | Scalable AI Mega-Cluster Design (Capstone) | All | 90 min |

## Directory Structure

```
interview-prep/
├── round-1-systems-coding/    # Lock-free structures, atomics, CUDA
├── round-2-gpu-kernel/        # FlashAttention, operator fusion
├── round-3-distributed-training/  # 3D parallelism, memory formulas
├── round-4-networking/        # NCCL, RDMA, Ring/Tree AllReduce
├── round-5-inference/         # PagedAttention, continuous batching
├── round-6-resilience/        # Gang scheduling, checkpointing
└── round-7-capstone/          # Full system design
```

## Key Competencies

### Round 1: Low-Level Systems Coding
- [ ] 无锁并发数据结构 (Lock-free Data Structures)
- [ ] 原子操作与内存顺序 (`std::memory_order_acquire`)
- [ ] 零拷贝 (Zero-copy) 实现
- [ ] CUDA 共享内存优化的矩阵乘法

### Round 2: GPU Architecture and Kernel Design
- [ ] FlashAttention 分块计算原理白板推演
- [ ] Warp 级并行与内存交互调度
- [ ] 算子融合 (Operator Fusion) 策略
- [ ] FP8 量化下的指令流利用率

### Round 3: Distributed Training Architecture
- [ ] DP/TP/PP 多维度组合设计
- [ ] 通信流量数学推演
- [ ] 显存占用公式 (Weights + Gradients + Optimizer States)
- [ ] DualPipe 计算-通信重叠

### Round 4: High-Performance Networking
- [ ] Ring AllReduce vs Tree AllReduce 数学权衡
- [ ] RDMA 内核旁路 (Kernel Bypass) 必然性
- [ ] ECMP 哈希流碰撞与长尾延迟
- [ ] PFC 死锁现象与规避

### Round 5: High-Throughput Inference
- [ ] KV Cache 动态膨胀问题
- [ ] PagedAttention 与 OS 页表对比
- [ ] 前缀缓存 (Prefix Caching) 实现
- [ ] TTFT vs ITL 平衡 (SLA 约束)

### Round 6: Resilience & Scheduling
- [ ] Volcano Gang Scheduling 防死锁
- [ ] 多层级 Checkpointing 低损耗高频快照
- [ ] 冗余流水线模板平滑自愈
- [ ] 万卡 MTBF 分析

### Round 7: Mega-Cluster Design (Capstone)
- [ ] 物理数据中心组网拓扑选择
- [ ] 并行策略对节点局部性约束
- [ ] 海量数据并行文件系统 I/O 预估
- [ ] MFU (Model FLOPS Utilization) 计算

## Practice Resources

Each round directory should contain:
1. `questions.md` - Common interview questions
2. `solutions/` - Your prepared answers and code
3. `whiteboard/` - Diagrams and formulas for whiteboard sessions
4. `mock-interviews/` - Self-assessment recordings
