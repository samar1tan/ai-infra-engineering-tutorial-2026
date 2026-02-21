# SYS-604: High-Throughput LLM Inference Systems

**Priority:** 4 (High) | **Duration:** Months 8-10 | **Dependencies:** SYS-601, SYS-602

## Overview

如果说训练系统决定了人工智能大模型的智商下限，那么推理引擎的工程架构则直接决定了 AI 企业商业化变现的成本上限。大语言模型基于自回归生成特性，使其长期处于极度的内存带宽受限 (Memory Bandwidth Bound) 状态。如何在毫秒级延迟内实现显存利用率的极致压榨，是本课程的核心系统命题。

### Core Engineering Domains
- KV Cache management
- Continuous batching
- PagedAttention mechanism
- Speculative decoding

## Learning Objectives

- [ ] 深入理解 KV Cache 动态膨胀问题及其解决方案
- [ ] 精读 vLLM 的 scheduler.py 和 block_manager.py 源码
- [ ] 掌握 PagedAttention 与操作系统虚拟内存的异同
- [ ] 理解连续批处理 (Continuous Batching) 的调度逻辑
- [ ] 探索投机解码 (Speculative Decoding) 的加速原理

## Source Code Study

### Primary Repository: `vllm-project/vllm`

#### Key Files
```
vllm/
├── core/
│   ├── scheduler.py        # 请求调度核心逻辑
│   ├── block_manager.py    # KV Cache 块管理
│   └── block_manager_v2.py # 优化版块管理
├── engine/
│   ├── llm_engine.py       # 主引擎入口
│   └── async_llm_engine.py # 异步引擎
├── attention/
│   └── backends/
│       └── flash_attn.py   # FlashAttention 后端
└── csrc/attention/
    └── attention_kernels.cu # PagedAttention CUDA 实现
```

### Study Goals
1. 追踪一次推理请求的完整生命周期
2. 理解动态批处理如何打破静态限制
3. 分析 PagedAttention Kernel 如何查询非连续内存页

## Required Reading

### Academic Papers
- [ ] Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023)
- [ ] FlashInfer: Customizable and Efficient Attention Engine for LLM Serving (Ye et al., 2024)
- [ ] Fast Speculative Decoding for vLLM (Snowflake AI Research, 2024)
- [ ] Achieving Platform Portability for vLLM by using Triton Autotuning (IBM Research, 2024)
- [ ] DeepSeekMoE: Towards Ultimate Expert Specialization (Dai et al., 2024)

### Engineering Blogs & Tutorials
- [ ] [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- [ ] [Explaining the source code behind the vLLM fast inference engine](https://medium.com/@crclq2018/explaining-the-source-code-behind-the-vllm-fast-inference-engine-91429f54d1f7)
- [ ] [Code Review: Deep Dive into vLLM's Architecture](https://zerohertz.github.io/vllm-openai-1/)
- [ ] [The Architecture Behind vLLM: How PagedAttention Improves Memory Utilization](https://medium.com/@mandeep0405/the-architecture-behind-vllm-how-pagedattention-improves-memory-utilization-2f9b25272110)
- [ ] [How Prompt Caching Works in vLLM](https://docs.vllm.ai/)
- [ ] [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [ ] [Why vLLM is the best choice for AI inference today](https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today)

## Key Concepts

### PagedAttention vs OS Virtual Memory
```
┌─────────────────────────────────────────────────────────────┐
│                    OS Virtual Memory                        │
├─────────────────────────────────────────────────────────────┤
│  Virtual Address Space → Page Table → Physical Pages        │
│  - Fixed page size (4KB)                                    │
│  - Demand paging, Copy-on-Write                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    PagedAttention                           │
├─────────────────────────────────────────────────────────────┤
│  Logical KV Blocks → Block Table → Physical GPU Memory      │
│  - Block size = num_heads * head_dim * num_tokens           │
│  - Dynamic allocation per request                           │
│  - Memory sharing across requests (prefix caching)          │
└─────────────────────────────────────────────────────────────┘
```

### Latency Metrics
```
TTFT (Time To First Token): Prefill 阶段延迟
ITL (Inter-Token Latency): Decode 阶段每个 token 延迟
E2E Latency = TTFT + (output_length - 1) * ITL
```

### Memory Analysis
```
KV Cache Size per Token = 2 * num_layers * num_heads * head_dim * dtype_size
Example (Llama-70B, FP16):
  = 2 * 80 * 64 * 128 * 2 = 2.62 MB per token
  For 4K context: ~10 GB KV Cache per request!
```

## Hands-on Labs

### Lab 1: vLLM Setup & Benchmarking
```
labs/lab01-vllm-setup/
├── install_vllm.sh
├── benchmark_throughput.py
├── benchmark_latency.py
└── analyze_metrics.py
```

### Lab 2: PagedAttention Deep Dive
```
labs/lab02-paged-attention/
├── kv_cache_analysis.py
├── block_allocation_trace.py
├── memory_fragmentation.py
└── prefix_caching_demo.py
```

### Lab 3: Continuous Batching
```
labs/lab03-continuous-batching/
├── static_vs_dynamic_batch.py
├── scheduler_simulation.py
├── sla_optimization.py
└── ttft_itl_tradeoff.py
```

### Lab 4: Speculative Decoding
```
labs/lab04-speculative-decoding/
├── draft_model_setup.py
├── speculation_analysis.py
├── acceptance_rate.py
└── latency_improvement.py
```

## Interview Mapping

This course directly maps to:
- **Round 2:** GPU Architecture and Kernel Design (PagedAttention kernel)
- **Round 5:** High-Throughput Inference System Design

### Sample Interview Questions
1. 从零设计兼容 OpenAI API 的生产级大模型服务系统
2. 推演 PagedAttention 如何管理非连续物理显存块
3. 如何在 SLA 约束下平衡 TTFT 和 ITL？
4. 实现跨请求的前缀缓存 (Prefix Caching)

## Progress Tracking

| Week | Topic | Status |
|------|-------|--------|
| 1-2 | LLM Inference Fundamentals | ⬜ |
| 3-4 | KV Cache & Memory Analysis | ⬜ |
| 5-6 | PagedAttention Deep Dive | ⬜ |
| 7-8 | vLLM Architecture Study | ⬜ |
| 9-10 | Continuous Batching | ⬜ |
| 11-12 | Speculative Decoding | ⬜ |
