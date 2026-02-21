# SYS-605: Large-Scale Cluster Scheduling and Fault Tolerance

**Priority:** 5 (Medium) | **Duration:** Months 11-12 | **Dependencies:** SYS-602, SYS-603

## Overview

当 AI 计算集群跨入万卡乃至十万卡级别时，单一组件的性能优化红利将被系统级的可靠性 (Reliability) 瓶颈彻底吞噬。在十万卡级别的超长周期训练作业中，硬件故障几乎每天都会发生。具备自愈能力的容错调度 (Fault Tolerance) 架构构成了 AI 基础设施的最后一道防线。

### Core Engineering Domains
- Gang scheduling on Kubernetes
- High-frequency checkpointing
- Automated failure recovery
- Elastic training systems

## Learning Objectives

- [ ] 理解 Kubernetes 与 Volcano 的群组调度机制
- [ ] 掌握高频 Checkpoint 的存储 I/O 优化策略
- [ ] 学习基于分布式内存的快速故障恢复 (Gemini)
- [ ] 探索弹性流水线模板的热切换机制 (Oobleck)
- [ ] 设计万卡级别集群的全局容错架构

## Source Code Study

### Primary Repository: `volcano-sh/volcano`

#### Key Components
```
volcano/
├── pkg/scheduler/
│   ├── api/           # 调度器 API
│   ├── plugins/       # 调度插件
│   │   ├── gang/      # Gang Scheduling 实现
│   │   └── drf/       # Dominant Resource Fairness
│   └── framework/     # 调度框架
├── pkg/controllers/
│   ├── job/           # Job 控制器
│   └── queue/         # Queue 控制器
└── cmd/
    └── scheduler/     # 调度器入口
```

### Study Goals
1. 理解 "All-or-Nothing" 调度保障的实现
2. 分析 Gang Scheduling 如何防止死锁
3. 探索拓扑感知的任务放置策略

## Required Reading

### Academic Papers
- [ ] Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates (Jang et al., 2023)
- [ ] ByteCheckpoint: An Industrial-Grade Checkpointing System for Large-Scale LFM Training (Wan et al., 2025)
- [ ] Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints (Zhuang et al., 2023)
- [ ] Reliability of AI Supercomputer Clusters (Kokolis et al., Meta FAIR, 2024)
- [ ] Deadline-Aware Flow Scheduling for AI Clusters with Heterogeneous Latency Requirements (2024)

### Engineering Blogs & Tutorials
- [ ] [Fault-tolerant training: How we build reliable clusters](https://nebius.com/blog/posts/how-we-build-reliable-clusters)
- [ ] [Storage Requirements for AI Clusters: The Hidden Cost of Checkpointing](https://www.cudocompute.com/blog/storage-requirements-for-ai-clusters)
- [ ] [Slurm for ML](https://www.run.house/blog/slurm-for-ml)
- [ ] [Volcano: Collision Between Containers and Batch Computing](https://www.cncf.io/blog/2021/02/26/volcano-collision-between-containers-and-batch-computing/)
- [ ] [Uber's Journey to Ray on Kubernetes](https://www.uber.com/blog/ubers-journey-to-ray-on-kubernetes-ray-setup/)
- [ ] [Ray vs Kubernetes for AI training scheduling comparison](https://cloud.google.com/blog/products/containers-kubernetes/ray-on-gke-new-features-for-ai-scheduling-and-scaling)
- [ ] [Understanding Slurm for AI/ML Workloads](https://www.whitefiber.com/blog/understanding-slurm-for-ai-ml-workloads)

## Key Concepts

### Gang Scheduling
```
┌─────────────────────────────────────────────────────────────┐
│                    Gang Scheduling                          │
├─────────────────────────────────────────────────────────────┤
│  All-or-Nothing: Either all pods scheduled or none          │
│                                                             │
│  Without Gang Scheduling:                                   │
│  - Partial allocation → deadlock risk                       │
│  - Resource fragmentation                                   │
│  - Wasted GPU-hours waiting                                 │
│                                                             │
│  With Gang Scheduling:                                      │
│  - Atomic allocation guarantees                             │
│  - No partial failures                                      │
│  - Efficient resource utilization                           │
└─────────────────────────────────────────────────────────────┘
```

### Checkpointing Strategies
```
┌─────────────────────────────────────────────────────────────┐
│              Checkpoint Storage Hierarchy                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 1: GPU HBM (fastest, volatile)                       │
│      ↓                                                      │
│  Level 2: CPU DRAM (Gemini approach)                        │
│      ↓    - In-memory checkpoints                           │
│      ↓    - Fast recovery (seconds)                         │
│      ↓                                                      │
│  Level 3: NVMe SSD (local)                                  │
│      ↓                                                      │
│  Level 4: Parallel FS (Lustre, GPFS)                        │
│      ↓    - Durable but slow                                │
│      ↓    - I/O bottleneck at scale                         │
│                                                             │
│  ByteCheckpoint: Tiered async writes                        │
└─────────────────────────────────────────────────────────────┘
```

### Failure Recovery Approaches
```
Traditional: Stop → Restore from Checkpoint → Resume
  - High recovery time (minutes to hours)
  - Global synchronization barrier

Oobleck: Pipeline Template Switching
  - Pre-computed redundant pipeline configurations
  - Hot-swap failed nodes
  - No global rollback needed
  - Recovery in seconds
```

## Failure Taxonomy (Meta FAIR Study)

| Failure Type | MTBF (10K GPUs) | Recovery Time |
|--------------|-----------------|---------------|
| GPU Memory Error | ~12 hours | Full restart |
| NIC Failure | ~24 hours | Node replacement |
| Node Crash | ~48 hours | Checkpoint restore |
| Network Partition | Rare | Topology reroute |

## Hands-on Labs

### Lab 1: Volcano Setup
```
labs/lab01-volcano/
├── install_volcano.sh
├── gang_scheduling_demo.yaml
├── queue_management.py
└── topology_aware_placement.yaml
```

### Lab 2: Checkpointing Strategies
```
labs/lab02-checkpointing/
├── pytorch_checkpoint.py
├── async_checkpoint.py
├── distributed_checkpoint.py
└── io_benchmark.py
```

### Lab 3: Fault Injection
```
labs/lab03-fault-injection/
├── chaos_engineering.py
├── failure_simulation.py
├── recovery_time_analysis.py
└── pipeline_recovery_demo.py
```

### Lab 4: KubeRay Integration
```
labs/lab04-kuberay/
├── ray_cluster_setup.yaml
├── elastic_training.py
├── autoscaling_config.yaml
└── monitoring_dashboard.py
```

## Architecture Patterns

### Cloud-Native AI Training Stack
```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│            (PyTorch, Megatron-LM, DeepSpeed)                │
├─────────────────────────────────────────────────────────────┤
│                    Orchestration Layer                      │
│              (Ray, Volcano, KubeRay)                        │
├─────────────────────────────────────────────────────────────┤
│                    Kubernetes Layer                         │
│           (Pod scheduling, Resource quotas)                 │
├─────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                     │
│        (GPU nodes, RDMA network, Storage)                   │
└─────────────────────────────────────────────────────────────┘
```

## Interview Mapping

This course directly maps to:
- **Round 6:** Resilience, Scheduling, and Orchestration

### Sample Interview Questions
1. 如何通过 Volcano 实施 Gang Scheduling 防范死锁？
2. 设计多层级 Checkpointing 系统以低损耗实现高频快照
3. 如何在不触发全局回滚的前提下实现平滑自愈？
4. 万卡级别集群的 MTBF 分析与容错架构设计

## Progress Tracking

| Week | Topic | Status |
|------|-------|--------|
| 1-2 | Kubernetes & Volcano Fundamentals | ⬜ |
| 3-4 | Gang Scheduling Deep Dive | ⬜ |
| 5-6 | Checkpointing Strategies | ⬜ |
| 7-8 | In-Memory Checkpointing (Gemini) | ⬜ |
| 9-10 | Elastic Training (Oobleck) | ⬜ |
| 11-12 | Capstone: Full Stack Design | ⬜ |
