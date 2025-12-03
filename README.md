# CacheDesigner: Graph-Guided Cache-to-Cache Communication for Multi-Agent LLMs

CacheDesigner combines **GDesigner's** graph-based topology learning with **LatentMAS's** KV-cache communication for efficient multi-agent collaboration.

## Architecture

- **Backbone**: GDesigner (graph topology + GCN optimization)
- **Cache Layer**: LatentMAS-style KV-cache fusion
- **Key Innovation**: Task-adaptive graph structure + cache-level communication

## Structure

```
G-cache/
├── GDesigner/              # GDesigner backbone (copied)
│   ├── graph/
│   │   ├── graph.py       # Original Graph class
│   │   └── cache_graph.py # NEW: CacheGraph with cache fusion
│   ├── agents/
│   ├── llm/
│   └── ...
├── experiments/
│   ├── run_gsm8k.py       # Original GDesigner runner
│   └── run_gsm8k_cache.py # NEW: CacheDesigner runner
├── cache_models.py         # LatentMAS cache generation
└── cache_methods.py        # LatentMAS methods
```

## Key Changes (Minimal)

### 1. CacheGraph (extends Graph)
- Adds `CacheFuser` module for KV-cache fusion
- Stores node caches during execution
- Fuses predecessor caches before node execution

### 2. Run Script
- Uses `CacheGraph` instead of `Graph`
- Adds cache-specific arguments
- Includes cache fuser in optimizer

## Usage

```bash
# Run with cache communication
cd G-cache/experiments
python run_gsm8k_cache.py \
    --use_cache \
    --optimized_spatial \
    --agent_names MathSolver \
    --agent_nums 4 \
    --batch_size 4 \
    --num_iterations 10

# Run without cache (original GDesigner)
python run_gsm8k.py \
    --optimized_spatial \
    --agent_names MathSolver \
    --agent_nums 4
```

## Method Overview

From your paper:

1. **Task-Adaptive Topology** (from GDesigner)
   - GCN learns agent communication graph
   - Sparse, task-specific connections

2. **Cache-Level Communication** (from LatentMAS)
   - KV-cache fusion between agents
   - Layer-wise gating
   - Edge-weighted aggregation

3. **Joint Optimization**
   - Stage 1: Pretrain cache fuser
   - Stage 2: Joint topology + cache optimization
   - Policy gradient for discrete decisions

## Implementation Notes

- **Minimal changes**: Only 2 new files added to GDesigner
- **Backward compatible**: Can run with `use_cache=False`
- **Modular**: Cache fuser is separate module
- **Efficient**: Reuses GDesigner's graph execution logic
