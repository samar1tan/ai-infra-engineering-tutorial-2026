# SYS-603: High-Performance AI Networking and Collectives

**Priority:** 3 (High) | **Duration:** Months 4-7 (parallel with SYS-602) | **Dependencies:** SYS-601

## Overview

在动辄调动数万张 GPU 的现代集群中，网络系统不再是简单的外围数据传输组件，而是成为了整个分布式 AI 巨型计算机的"内部总线"。在高达数千 Gbps 的吞吐量要求面前，传统 TCP/IP 协议栈显得极其笨重。RDMA 与深层定制的集合通信库是高级系统工程师必须攻克的深水区。

### Core Engineering Domains
- RDMA protocols (RoCEv2, InfiniBand)
- NCCL algorithms (Ring, Tree, etc.)
- Network topology awareness
- Congestion control (PFC, ECMP)

## Learning Objectives

- [ ] 彻底理解集合通信原语 (AllReduce, AllGather, Reduce-Scatter, Broadcast)
- [ ] 掌握 Ring AllReduce 与 Tree AllReduce 的数学权衡
- [ ] 深入研究 NCCL 拓扑探测与通道构建机制
- [ ] 理解 RDMA 内核旁路与零拷贝原理
- [ ] 分析 PFC 死锁与 ECMP 哈希碰撞问题

## Source Code Study

### Primary Repository: `NVIDIA/nccl`

#### Key Directories
```
src/
├── collectives/        # 集合通信实现
│   ├── all_reduce.cc   # AllReduce 核心逻辑
│   ├── all_gather.cc
│   └── reduce_scatter.cc
├── transport/          # 传输层抽象
│   ├── net.cc          # 网络传输
│   └── p2p.cc          # 点对点通信
├── graph/              # 拓扑图构建
│   ├── topo.cc         # 拓扑探测
│   └── search.cc       # 最优路径搜索
└── include/
    └── nccl.h          # 公开 API
```

### Study Goals
1. 追踪拓扑探测阶段的 Ring/Double-Tree 构建
2. 理解 Chunk 切分与 Pipeline 重叠机制
3. 分析通道分配与带宽利用率优化

## Required Reading

### Academic Papers
- [ ] RDMA over Commodity Ethernet at Scale (Guo et al., 2016)
- [ ] OmniReduce: Efficient Sparse Collective Communication (Fei et al., 2021)
- [ ] SwitchML: Hardware-Accelerated Distributed Machine Learning (Sapio et al., 2021)
- [ ] Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols (2025)
- [ ] NCCLX: Scalable, High-Performance Collective Communication for 100k+ GPUs (Zeng et al., 2025)

### Engineering Blogs & Tutorials
- [ ] [Unpacking NCCL: A Deep Dive into Multi-GPU Communication](https://medium.com/@nitin966/unpacking-nccl-a-deep-dive-into-multi-gpu-communication-2b667e77d96d)
- [ ] [Understanding NCCL Tuning to Accelerate GPU-to-GPU Communication](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/)
- [ ] [Fast Multi-GPU collectives with NCCL](https://developer.nvidia.com/blog/)
- [ ] [NCCL Deep Dive: Cross Data Center Communication and Network Topology Awareness](https://developer.nvidia.com/blog/nccl-deep-dive-cross-data-center-communication-and-network-topology-awareness/)
- [ ] [RDMA over Ethernet for Distributed AI Training at Meta Scale](https://cs.stanford.edu/~keithw/sigcomm2024/sigcomm24-final246-acmpaginated.pdf)
- [ ] [Zettascale OSU NCCL Benchmark on H100 AI Workloads](https://blogs.oracle.com/cloud-infrastructure/zettascale-osu-nccl-benchmark-h100-ai-workloads)
- [ ] [Understanding RoCEv2: A Beginner's Guide to RDMA over Converged Ethernet](https://medium.com/)
- [ ] [The Battle of AI Networking: Ethernet vs InfiniBand](https://counterpointresearch.com/en/insights/smart-networking-key-for-optimum-roi-in-ai-data-centers)

## Key Algorithms

### Ring AllReduce
```
Communication Volume: 2 * (N-1) / N * data_size
Latency: 2 * (N-1) * α + 2 * (N-1) / N * data_size / β
Where: α = latency per message, β = bandwidth
```

### Tree AllReduce  
```
Communication Volume: 2 * data_size * log(N)
Latency: 2 * log(N) * α + 2 * data_size / β
Better for small messages, worse bandwidth utilization for large
```

### Double Binary Tree (NCCL default for large messages)
```
Combines benefits of both approaches
Uses two complementary trees for full bandwidth utilization
```

## Network Topology Concepts

### Fat-Tree Topology
```
        ┌─────────────────────────────────────┐
        │          Core Switches              │
        └─────────────────────────────────────┘
                    │           │
        ┌───────────┴───┐   ┌───┴───────────┐
        │  Aggregation  │   │  Aggregation  │
        └───────────────┘   └───────────────┘
            │       │           │       │
        ┌───┴───┐ ┌─┴───┐   ┌───┴───┐ ┌─┴───┐
        │ ToR 1 │ │ToR 2│   │ ToR 3 │ │ToR 4│
        └───────┘ └─────┘   └───────┘ └─────┘
          │││       │││       │││       │││
         GPUs      GPUs      GPUs      GPUs
```

### Intra-Node: NVLink/NVSwitch
```
8x H100 with NVSwitch:
- Full bisection bandwidth: 900 GB/s per GPU
- Any-to-any communication without congestion
```

## Hands-on Labs

### Lab 1: NCCL Benchmarking
```
labs/lab01-nccl-benchmark/
├── run_allreduce_bench.sh
├── run_allgather_bench.sh
├── analyze_bandwidth.py
└── topology_visualization.py
```

### Lab 2: Communication Patterns
```
labs/lab02-comm-patterns/
├── ring_allreduce_sim.py
├── tree_allreduce_sim.py
├── compare_algorithms.py
└── latency_bandwidth_tradeoff.py
```

### Lab 3: RDMA Exploration
```
labs/lab03-rdma/
├── rdma_basics.c
├── ibv_pingpong.c
├── roce_vs_ib_comparison.md
└── pfc_analysis.py
```

## Interview Mapping

This course directly maps to:
- **Round 4:** High-Performance Networking (Collectives & RDMA)

### Sample Interview Questions
1. 解释 Ring AllReduce 与 Tree AllReduce 的数学权衡
2. ECMP 哈希流碰撞如何导致长尾延迟？如何缓解？
3. PFC 如何引发死锁？如何规避？
4. 为什么现代 AI 基建拥抱 RDMA 实现内核旁路？

## Progress Tracking

| Week | Topic | Status |
|------|-------|--------|
| 1-2 | Collective Communication Primitives | ⬜ |
| 3-4 | NCCL Architecture Deep Dive | ⬜ |
| 5-6 | Ring & Tree Algorithms | ⬜ |
| 7-8 | RDMA Fundamentals | ⬜ |
| 9-10 | RoCEv2 vs InfiniBand | ⬜ |
| 11-12 | Congestion Control (PFC, ECMP) | ⬜ |
| 13-14 | Topology-Aware Scheduling | ⬜ |
| 15-16 | NCCLX & Future Directions | ⬜ |
