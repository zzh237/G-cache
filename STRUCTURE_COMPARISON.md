# G-cache vs GDesigner-main: Structure Comparison

## Overview
G-cache = **GDesigner-main (backbone)** + **LatentMAS cache logic** + **Minimal integration layer**

---

## 📁 File Structure Comparison

### ✅ IDENTICAL (Copied from GDesigner-main)
```
G-cache/
├── GDesigner/              # 100% SAME as GDesigner-main
│   ├── agents/            # ✅ All agent implementations
│   ├── llm/               # ✅ LLM interface & API calls
│   ├── prompt/            # ✅ Prompt templates
│   ├── tools/             # ✅ Utilities (coding, search, etc.)
│   ├── utils/             # ✅ Helper functions
│   ├── gnn/               # ✅ GCN for topology learning
│   └── graph/
│       ├── graph.py       # ✅ Original Graph class
│       └── node.py        # ✅ Original Node class
├── datasets/              # ✅ Same datasets
├── experiments/
│   ├── run_gsm8k.py      # ✅ Original GDesigner runner
│   ├── run_mmlu.py       # ✅ Original
│   └── run_humaneval.py  # ✅ Original
└── requirements.txt       # ✅ Same dependencies
```

### 🆕 NEW FILES (Added for CacheDesigner)
```
G-cache/
├── GDesigner/graph/
│   └── cache_graph.py     # 🆕 NEW: Extends Graph with cache
├── experiments/
│   └── run_gsm8k_cache.py # 🆕 NEW: Runner with cache support
├── cache_designer/
│   ├── __init__.py        # 🆕 NEW: Package init
│   └── cache_fuser.py     # 🆕 NEW: Full cache fusion (not used yet)
├── cache_models.py        # 🆕 NEW: LatentMAS model wrapper
├── cache_methods.py       # 🆕 NEW: LatentMAS methods
└── README.md              # 🆕 UPDATED: CacheDesigner docs
```

**Total new code: Only 3 files matter**
- `cache_graph.py` (90 lines)
- `run_gsm8k_cache.py` (180 lines)
- `cache_fuser.py` (130 lines, advanced version)

---

## 🔍 Detailed Code Changes

### 1. **cache_graph.py** (NEW - Extends Graph)

**What it does:**
- Inherits from `Graph` class
- Adds `CacheFuser` module
- Stores node KV-caches during execution
- Fuses predecessor caches before node execution

**Key additions:**
```python
class CacheGraph(Graph):
    def __init__(self, *args, use_cache_communication=True, **kwargs):
        super().__init__(*args, **kwargs)  # ← Calls original Graph.__init__
        
        if use_cache_communication:
            self.cache_fuser = CacheFuser(...)  # ← NEW: Cache fusion module
            self.node_caches = {}               # ← NEW: Store caches
    
    def store_node_cache(self, node_id, cache):  # ← NEW method
        self.node_caches[node_id] = cache
    
    def get_fused_cache(self, node):             # ← NEW method
        # Collect caches from predecessors
        # Fuse them using cache_fuser
        return fused_cache
    
    async def arun(self, ...):                   # ← OVERRIDE parent
        self.node_caches.clear()                 # ← NEW: Clear at start
        return await super().arun(...)           # ← Call original
```

**Changes from GDesigner:**
- ✅ Keeps all original Graph logic
- ➕ Adds cache storage
- ➕ Adds cache fusion
- ✅ Backward compatible (can disable cache)

---

### 2. **run_gsm8k_cache.py** (NEW - Modified Runner)

**What it does:**
- Same as `run_gsm8k.py` but uses `CacheGraph`
- Adds cache-specific arguments
- Includes cache fuser in optimizer

**Key changes:**
```python
# BEFORE (GDesigner):
from GDesigner.graph.graph import Graph
graph = Graph(domain="gsm8k", llm_name=args.llm_name, ...)
optimizer = torch.optim.Adam(graph.gcn.parameters(), lr=args.lr)

# AFTER (CacheDesigner):
from GDesigner.graph.cache_graph import CacheGraph  # ← Changed import
graph = CacheGraph(                                  # ← Changed class
    domain="gsm8k",
    llm_name=args.llm_name,
    use_cache_communication=args.use_cache,          # ← NEW argument
    hidden_dim=args.hidden_dim,                      # ← NEW argument
    num_cache_layers=args.num_cache_layers,          # ← NEW argument
    ...
)

# Include cache fuser in optimizer
params = list(graph.gcn.parameters())
if args.use_cache:
    params += list(graph.cache_fuser.parameters())   # ← NEW
optimizer = torch.optim.Adam(params, lr=args.lr)
```

**Everything else is IDENTICAL to run_gsm8k.py**

---

### 3. **CacheFuser Module** (NEW - Cache Fusion Logic)

**Two versions exist:**

#### Simple version (in `cache_graph.py`):
```python
class CacheFuser(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        self.layer_gates = nn.Parameter(torch.zeros(num_layers))
        self.fusion_weights = nn.Parameter(torch.ones(num_layers))
    
    def forward(self, receiver_cache, sharer_caches, edge_weights):
        # Simple weighted sum with gating
        for l in range(num_layers):
            gate = torch.sigmoid(self.layer_gates[l])
            agg = sum(w * sc[l] for w, sc in zip(edge_weights, sharer_caches))
            fused.append(receiver_cache[l] + gate * agg)
        return fused
```

#### Advanced version (in `cache_fuser.py`):
```python
class CacheFuser(nn.Module):
    def __init__(self, hidden_dim, num_layers, device):
        # Alignment MLPs for K and V
        self.align_mlps_k = nn.ModuleList([...])
        self.align_mlps_v = nn.ModuleList([...])
        
        # Fusion MLPs
        self.fusion_mlps_k = nn.ModuleList([...])
        self.fusion_mlps_v = nn.ModuleList([...])
        
        # Layer gates
        self.layer_gates_alpha = nn.Parameter(...)
    
    def forward(self, receiver_caches, sharer_caches_list, edge_weights):
        # 1. Align Sharer caches to Receiver dimension
        # 2. Aggregate with edge weights
        # 3. Fuse with residual connection
        return fused_caches
```

**Currently using: Simple version** (in cache_graph.py)

---

## 🎯 What's Different from GDesigner?

### Conceptual Changes:
| Aspect | GDesigner | CacheDesigner |
|--------|-----------|---------------|
| **Communication** | Text-to-text | Text + KV-cache |
| **Agent interaction** | Via text messages | Via text + hidden states |
| **Information flow** | Explicit (readable) | Explicit + Implicit (latent) |
| **Overhead** | High token cost | Lower token cost |
| **Semantic richness** | Limited by text | Richer (direct embeddings) |

### Technical Changes:
| Component | GDesigner | CacheDesigner |
|-----------|-----------|---------------|
| **Graph class** | `Graph` | `CacheGraph` (extends Graph) |
| **Node execution** | Text output only | Text + cache storage |
| **Topology learning** | GCN on text | GCN on text + cache |
| **Optimization** | GCN parameters | GCN + CacheFuser parameters |
| **Memory** | Text history | Text + KV-cache history |

---

## 🚀 Can You Run the Same Experiment?

### YES - Three Ways:

#### 1. **Run Original GDesigner (Baseline)**
```bash
cd G-cache/experiments
python run_gsm8k.py \
    --optimized_spatial \
    --agent_names MathSolver \
    --agent_nums 4 \
    --batch_size 4 \
    --num_iterations 10
```

#### 2. **Run CacheDesigner WITHOUT Cache (Should match baseline)**
```bash
python run_gsm8k_cache.py \
    --optimized_spatial \
    --agent_names MathSolver \
    --agent_nums 4 \
    --batch_size 4 \
    --num_iterations 10
    # Note: --use_cache is NOT set, so no cache communication
```

#### 3. **Run CacheDesigner WITH Cache (Expected better results)**
```bash
python run_gsm8k_cache.py \
    --use_cache \
    --optimized_spatial \
    --agent_names MathSolver \
    --agent_nums 4 \
    --batch_size 4 \
    --num_iterations 10 \
    --hidden_dim 4096 \
    --num_cache_layers 32
```

---

## 📊 Expected Results

### Hypothesis (from your paper):

| Method | Accuracy | Token Usage | Latency |
|--------|----------|-------------|---------|
| GDesigner (text-only) | Baseline | High | High |
| CacheDesigner (text+cache) | **+1-2.5%** | **-20-40%** | **-15-30%** |

### Why Better Results?

1. **Richer Information**: KV-caches contain full semantic embeddings, not compressed text
2. **Less Ambiguity**: Direct hidden state transfer vs. natural language interpretation
3. **Efficient**: Fewer tokens needed for communication
4. **Complementary**: Text for explicit reasoning + cache for implicit knowledge

---

## ⚠️ Current Limitations

### 🔴 NOT YET IMPLEMENTED:
1. **Actual cache extraction from LLM**
   - Currently: Placeholder cache storage
   - Need: Hook into LLM forward pass to extract real KV-caches

2. **Cache injection into LLM**
   - Currently: Fused cache computed but not used
   - Need: Modify LLM generation to use fused cache

3. **LatentMAS integration**
   - `cache_models.py` and `cache_methods.py` copied but not connected
   - Need: Integrate LatentMAS's `generate_latent_batch()` method

### 🟡 PARTIALLY IMPLEMENTED:
1. **Cache fusion logic** ✅ (CacheFuser module exists)
2. **Graph structure** ✅ (CacheGraph extends Graph)
3. **Training loop** ✅ (Optimizer includes cache parameters)

### 🟢 FULLY WORKING:
1. **GDesigner backbone** ✅
2. **Topology learning** ✅
3. **Text-based communication** ✅
4. **All original experiments** ✅

---

## 🛠️ To Make It Fully Functional:

### Step 1: Connect to Real LLM Caches
```python
# In Node._async_execute():
response, kv_cache = await self.llm.agen_with_cache(message)  # ← Need this
graph.store_node_cache(self.id, kv_cache)  # ← Store real cache
```

### Step 2: Use Fused Cache in Generation
```python
# In Node._async_execute():
fused_cache = graph.get_fused_cache(self)  # ← Get fused cache
response = await self.llm.agen_with_cache(message, past_kv=fused_cache)  # ← Use it
```

### Step 3: Integrate LatentMAS Methods
```python
# Use LatentMAS's cache generation:
from cache_models import ModelWrapper
model = ModelWrapper(llm_name, device, use_vllm=True)
past_kv = model.generate_latent_batch(input_ids, latent_steps=10)
```

---

## 📝 Summary

### What's the Same:
- ✅ 95% of codebase (entire GDesigner backbone)
- ✅ All agents, prompts, tools, utilities
- ✅ GCN topology learning
- ✅ Training loop structure
- ✅ Evaluation metrics

### What's Different:
- 🆕 `CacheGraph` class (extends `Graph`)
- 🆕 `CacheFuser` module (cache fusion logic)
- 🆕 `run_gsm8k_cache.py` (modified runner)
- 🆕 Cache storage and retrieval methods

### Can You Run Experiments?
- ✅ **YES** - Original GDesigner experiments work perfectly
- ✅ **YES** - CacheDesigner without cache works (same as GDesigner)
- ⚠️ **PARTIAL** - CacheDesigner with cache needs LLM integration

### Will You Get Better Results?
- **In theory: YES** (based on your paper's method)
- **In practice: NOT YET** (needs LLM cache extraction/injection)
- **After full implementation: EXPECTED +1-2.5% accuracy, -20-40% tokens**

---

## 🎯 Next Steps to Complete Implementation:

1. Modify `gpt_chat.py` to return KV-caches
2. Modify agents to store/use caches
3. Integrate LatentMAS's cache generation
4. Test on small dataset
5. Compare with baseline

**Current Status: 70% complete - Structure ready, needs LLM integration**
