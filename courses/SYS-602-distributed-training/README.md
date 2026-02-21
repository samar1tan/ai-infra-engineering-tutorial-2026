# SYS-602: Distributed Training and Hybrid Parallelism

**Priority:** 2 (Critical) | **Duration:** Months 4-7 | **Dependencies:** SYS-601

## Overview

由于单卡显存容量的物理极限，大语言模型必须被科学地拆解并分布到成百上千张 GPU 上。本课程致力于研究如何利用多维度的混合并行策略 (Hybrid Parallelism)，在跨节点通信开销与单卡计算效率之间寻找最优的纳什均衡。

### Core Engineering Domains
- 3D/4D Parallelism (DP, TP, PP, EP)
- MoE routing strategies
- Memory optimization (ZeRO Stage 1/2/3)
- Pipeline scheduling (1F1B, DualPipe)

## Learning Objectives

- [ ] 手推各种并行策略的显存占用和通信量公式
- [ ] 精读 Megatron-LM 的 tensor_parallel 和 pipeline_parallel 源码
- [ ] 理解 ZeRO Stage 1/2/3 的分片策略与 AllGather 通信开销
- [ ] 掌握 1F1B 调度器与流水线气泡压缩原理
- [ ] 深入理解 DeepSeek DualPipe 的双向流水线并行调度

## Source Code Study

### Primary Repository: `NVIDIA/Megatron-LM`

#### Key Directories
```
megatron/core/
├── tensor_parallel/    # 张量并行核心实现
│   ├── layers.py       # ColumnParallelLinear, RowParallelLinear
│   └── mappings.py     # 通信原语封装
├── pipeline_parallel/  # 流水线并行核心实现
│   ├── schedules.py    # 1F1B, Interleaved 1F1B
│   └── p2p_communication.py
└── distributed/        # 分布式优化器
```

### Study Goals
1. 理解列切分与行切分的组合艺术
2. 追踪 Forward/Backward 中通信原语的注入时机
3. 分析流水线气泡的数学模型

## Required Reading

### Academic Papers
- [ ] Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (Shoeybi et al., 2019)
- [ ] ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)
- [ ] Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning (Zheng et al., 2022)
- [ ] Distributed Hybrid Parallelism for Large Language Models: Comparative Study and System Design Guide (Amer et al., 2026)
- [ ] DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)

### Engineering Blogs & Tutorials
- [ ] [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/)
- [ ] [Megatron-LM: How Model Parallelism is Pushing Language Models to New Heights](https://huggingface.co/docs/accelerate/en/usage_guides/megatron_lm)
- [ ] [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [ ] [Training 175B Parameter Language Models at 1000 GPU scale with Alpa and Ray](https://developer.nvidia.com/blog/)
- [ ] [Megatron Bridge Documentation & Parallelisms Guide](https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/parallelisms.html)
- [ ] [DeepSeek-V3 Technical Report Break Down: DualPipe & FP8](https://medium.com/@345490675/day-4-of-deepseeks-open-source-week-from-dualpipe-to-eplb-ed90f2f81d55)
- [ ] [How Meta Optimized Llama 3 Pretraining](https://engineering.fb.com/)

## Key Formulas to Master

### Memory Footprint Analysis
```
Model States = Parameters + Gradients + Optimizer States
             = Φ + Φ + 2Φ (for Adam: m, v)
             = 4Φ per GPU (without ZeRO)

ZeRO Stage 1: Optimizer states sharded → 2Φ + 2Φ/N
ZeRO Stage 2: + Gradients sharded → 2Φ/N + 2Φ/N  
ZeRO Stage 3: + Parameters sharded → 4Φ/N
```

### Communication Volume
```
Data Parallel AllReduce: 2 * model_size * (N-1) / N
Tensor Parallel AllReduce: 2 * activation_size * (N-1) / N
Pipeline P2P: activation_size * micro_batches
```

## Hands-on Labs

### Lab 1: Parallelism Simulation
```
labs/lab01-parallelism-sim/
├── dp_simulation.py
├── tp_simulation.py
├── pp_simulation.py
└── memory_calculator.py
```

### Lab 2: Megatron-LM Experiments
```
labs/lab02-megatron/
├── setup_environment.sh
├── run_tp_experiment.py
├── run_pp_experiment.py
└── analyze_timeline.py
```

### Lab 3: ZeRO Deep Dive
```
labs/lab03-zero/
├── zero_stage1.py
├── zero_stage2.py
├── zero_stage3.py
└── communication_analysis.py
```

## Interview Mapping

This course directly maps to:
- **Round 3:** Distributed Training Architecture

### Sample Interview Questions
1. 设计方案训练 1 万亿参数语言模型，集群 1024 张 H100
2. 计算 TP=8, PP=4, DP=32 下每次 Forward/Backward 的通信量
3. 解释 DualPipe 如何实现计算-通信重叠

## Progress Tracking

| Week | Topic | Status |
|------|-------|--------|
| 1-2 | Data Parallelism & DDP | ⬜ |
| 3-4 | ZeRO Optimization | ⬜ |
| 5-6 | Tensor Parallelism | ⬜ |
| 7-8 | Pipeline Parallelism | ⬜ |
| 9-10 | Hybrid Parallelism Design | ⬜ |
| 11-12 | MoE & Expert Parallelism | ⬜ |
| 13-14 | DualPipe & Advanced Topics | ⬜ |
| 15-16 | Capstone Project | ⬜ |
