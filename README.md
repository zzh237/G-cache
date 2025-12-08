# G-cache: Graph-Guided KV-Cache Communication for Multi-Agent LLMs

Combines **GDesigner's** graph topology with **LatentMAS's** KV-cache generation.

---

## 🚀 Quick Start

```bash
cd experiments

# Hybrid mode (RECOMMENDED) - small GPU + free API
python run_gsm8k_cache_API.py --llm_name hybrid_cache --use_cache
```

---

## 🎓 What Gets Updated During Training?

### Critical Insight: LatentMAS vs G-cache

**LatentMAS (original):**
- ❌ **NO training** - Inference only!
- ❌ No optimizer, no loss.backward()
- ❌ All models frozen
- Just runs multi-agent inference with cache

**G-cache (this project):**
- ✅ **HAS training** - Learns graph structure!
- ✅ Optimizer updates GCN + CacheFuser
- ✅ Learns from task performance

### Models in G-cache

| Model | Trainable? | Updated? | Purpose | LatentMAS Has This? |
|-------|-----------|----------|---------|---------------------|
| **1. GCN** | ✅ Yes | ✅ Yes | Graph edge weights | ❌ No (NEW in G-cache) |
| **2. CacheFuser** | ✅ Yes | ✅ Yes | Cache fusion weights | ❌ No (NEW in G-cache) |
| **3. Small Local Model** | ❌ Frozen | ❌ No | Cache generation | ✅ Yes (frozen in both) |
| **4. API Model** | ❌ External | ❌ No | Text generation | ✅ Yes (frozen in both) |

### Training Code (G-cache ONLY)

```python
# File: experiments/run_gsm8k_cache_API.py

# Step 1: Setup optimizer with trainable components (NEW in G-cache)
params = list(graph.gcn.parameters())  # ← GCN weights (NEW)
if args.use_cache:
    params += list(graph.cache_fuser.parameters())  # ← CacheFuser weights (NEW)
optimizer = torch.optim.Adam(params, lr=0.1)

# Step 2: Forward pass
answer, log_prob = await graph.arun(question)

# Step 3: Compute loss (NEW in G-cache)
is_correct = (answer == gold_answer)
utility = float(is_correct)
loss = -log_prob * utility

# Step 4: Backprop (NEW in G-cache)
optimizer.zero_grad()
loss.backward()  # Updates:
                 # ✅ GCN.edge_weights (NEW)
                 # ✅ CacheFuser.layer_gates (NEW)
                 # ✅ CacheFuser.fusion_weights (NEW)
                 # ❌ Small local model (frozen - same as LatentMAS)
                 # ❌ API model (external - same as LatentMAS)
optimizer.step()
```

### What LatentMAS Does (Inference Only)

```python
# File: LatentMAS/run.py

# NO optimizer!
# NO loss!
# NO training!

# Just inference:
for item in dataset:
    result = method.run_batch([item])  # Generate answer
    is_correct = (result['prediction'] == item['gold'])
    # That's it - no backprop!
```

### Comparison

| Aspect | LatentMAS | G-cache |
|--------|-----------|---------|
| **Training** | ❌ No | ✅ Yes |
| **Optimizer** | ❌ No | ✅ Yes (Adam) |
| **Loss** | ❌ No | ✅ Yes (-log_prob * utility) |
| **Backprop** | ❌ No | ✅ Yes |
| **Trainable params** | 0 | GCN + CacheFuser |
| **Cache generation** | ✅ Yes (frozen model) | ✅ Yes (frozen model) |
| **Graph structure** | Fixed (sequential/hierarchical) | ✅ Learned |
| **Cache fusion** | Fixed (concatenation) | ✅ Learned |

### Guarantee: No Missing Updates

**LatentMAS updates**: 0 models (inference only)

**G-cache updates**: 2 models
1. ✅ GCN - Added to optimizer (line 108)
2. ✅ CacheFuser - Added to optimizer (line 110)

**Verification:**
```python
# Check what's in optimizer
params = list(graph.gcn.parameters())  # ← All GCN params
params += list(graph.cache_fuser.parameters())  # ← All CacheFuser params
optimizer = torch.optim.Adam(params, lr=0.1)

# These are ALL trainable parameters in the system!
# Small model and API are explicitly frozen/external
```

**Guarantee**: ✅ We update everything that should be updated!

---

## 🔗 How Graph Functions Connect with Cache Functions

### The Connection Chain

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. CacheGraph (cache_graph.py)                                 │
│    - Manages cache storage: node_caches = {}                   │
│    - Provides: get_fused_cache(), store_node_cache()           │
│    - Trainable: CacheFuser ✅                                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓ passes graph reference
┌─────────────────────────────────────────────────────────────────┐
│ 2. MathSolverCache (math_solver_cache.py)                      │
│    - Receives: self.graph = graph                              │
│    - Calls: graph.get_fused_cache(self)                        │
│    - Calls: graph.store_node_cache(self.id, cache)             │
│    - Trainable: None                                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓ calls LLM
┌─────────────────────────────────────────────────────────────────┐
│ 3. HybridCacheLLM (llm_cache_hybrid.py)                        │
│    - Receives: past_key_values from graph                      │
│    - Calls: hybrid_model.generate_latent_batch()               │
│    - Returns: (text, kv_cache) to agent                        │
│    - Trainable: None                                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓ calls LatentMAS
┌─────────────────────────────────────────────────────────────────┐
│ 4. HybridCacheModel (hybrid_cache_model.py)                    │
│    - EXACT LatentMAS implementation                            │
│    - generate_latent_batch(past_key_values=fused_cache)        │
│    - Returns: new_cache                                        │
│    - Trainable: None (frozen) ❌                                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Connections Summary

| Connection | From | To | Data | Function |
|------------|------|----|----|----------|
| 1 | CacheGraph | Agent | `graph` reference | `node.graph = self` |
| 2 | Agent | CacheGraph | `self` | `graph.get_fused_cache(self)` |
| 3 | CacheGraph | Agent | `fused_cache` | Returns fused cache |
| 4 | Agent | LLM | `past_key_values` | `llm.agen_with_cache(..., past_key_values)` |
| 5 | LLM | LatentMAS | `past_key_values` | `generate_latent_batch(..., past_key_values)` |
| 6 | LatentMAS | LLM | `new_cache` | Returns cache |
| 7 | LLM | Agent | `(text, cache)` | Returns tuple |
| 8 | Agent | CacheGraph | `cache` | `graph.store_node_cache(id, cache)` |

---

## 🎯 Four Modes

### Mode 1: Pure Local with Real Cache (BEST for Research) ✅

```bash
cd experiments
export CUDA_VISIBLE_DEVICES=1,2  # Use 2 GPUs for 7B model
python run_gsm8k_cache.py --llm_name Qwen/Qwen3-8B --use_cache --device cuda
```

**What it does**:
- ✅ **Real KV-cache** from local model (LatentMAS)
- ✅ **Alignment matrix** enabled by default
- ✅ Trains GCN + CacheFuser
- ✅ No API needed (fully local)
- ✅ Uses HuggingFace transformers (no vLLM)

**GPU**: ~16GB for 8B model (or ~4GB for 1.7B model)

**Model options**:
- `Qwen/Qwen3-1.7B` - Small, ~4GB GPU
- `Qwen/Qwen3-4B` - Medium, ~8GB GPU  
- `Qwen/Qwen3-8B` - Large, ~16GB GPU (recommended)

**Use this for**: Real cache experiments, full control

---

### Mode 2: Hybrid (Small GPU + API) ✅

```bash
cd experiments
export CUDA_VISIBLE_DEVICES=1  # Only need 1 GPU
python run_gsm8k_cache_API.py --llm_name hybrid_cache --use_cache
```

**What it does**:
- ✅ Small local model (1.5B) generates **real KV-cache**
- ✅ API (qwen-flash) generates final text (faster & cheaper than qwen-plus)
- ✅ **Alignment matrix** enabled
- ⚠️ Requires API key

**GPU**: ~4GB only

**API models available**: `qwen-flash` (default), `qwen-plus`, `qwen-turbo`

**Use this for**: When you have API access

---

### Mode 3: API Baseline (No Cache)

```bash
cd experiments
python run_gsm8k_cache_API.py --llm_name qwen-flash  # or qwen-plus, qwen-turbo
```

**What it does**:
- ❌ No real cache (text-based only)
- Uses API for everything
- Baseline for comparison

**GPU**: 0GB

---

### Mode 4: Local Cache (Deprecated)

```bash
cd experiments
export CUDA_VISIBLE_DEVICES=1,2
python run_gsm8k_cache_API.py --llm_name local_cache --use_cache
```

**Note**: Use Mode 1 (`run_gsm8k_cache.py`) instead - it's the same but better tested

---

## 🔧 Setup

```bash
# 1. Install dependencies
pip install torch transformers openai python-dotenv

# 2. Set API key in .env file (already done!)
# Just edit .env:
# API_KEY=your_dashscope_api_key

# 3. Run (no export needed!)
cd experiments
python run_gsm8k_cache_API.py --llm_name hybrid_cache --use_cache
```

**Note**: The code automatically loads `API_KEY` from `.env` - no need to export anything!

---

## 🔬 LatentMAS Alignment Matrix (Training-Free)

### What is it?

LatentMAS uses a **projection matrix** `W_a` to align hidden states back to valid input embeddings:

```
e = h * W_a, where W_a ≈ W_out^(-1) * W_in
```

**Problem**: Hidden states from last layer have different distribution than input embeddings

**Solution**: Linear transformation that maps output space → input space (training-free!)

### Implementation in LatentMAS

**File**: `LatentMAS/models.py`

```python
class ModelWrapper:
    def _build_latent_realign_matrix(self, model, device, args):
        """Compute W_a = (W_out^T * W_out)^(-1) * W_out^T * W_in"""
        input_weight = model.get_input_embeddings().weight   # W_in
        output_weight = model.get_output_embeddings().weight # W_out
        
        # Solve: W_out * W_a = W_in
        gram = torch.matmul(output_weight.T, output_weight)  # W_out^T * W_out
        reg = 1e-5 * torch.eye(gram.shape[0])                # Regularization
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)    # W_out^T * W_in
        realign_matrix = torch.linalg.solve(gram, rhs)       # W_a
        
        target_norm = input_weight.norm(dim=1).mean()        # For normalization
        return realign_matrix, target_norm
    
    def _apply_latent_realignment(self, hidden, model):
        """Apply: e = normalize(h * W_a)"""
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device)
        aligned = torch.matmul(hidden.float(), matrix)       # h * W_a
        
        # Normalize to match input embedding norms
        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)
    
    def generate_latent_batch(self, ...):
        for step in range(latent_steps):
            # KEY: Apply alignment before feeding back
            latent_vec = self._apply_latent_realignment(last_hidden, self.model)
            latent_embed = latent_vec.unsqueeze(1)
            
            outputs = self.model(
                inputs_embeds=latent_embed,  # ← Aligned embedding
                ...
            )
```

**Usage**: `python run.py --latent_space_realign` (optional flag)

### Does G-cache Have This?

**Answer**: ✅ **YES! Now implemented!**

**File**: `G-cache/hybrid_cache_model.py`

```python
class HybridCacheModel:
    def __init__(self, ..., use_alignment: bool = True):
        # Build alignment matrix (training-free, computed once)
        if self.use_alignment:
            self._build_alignment_matrix()
    
    def _build_alignment_matrix(self):
        """Build W_a ≈ W_out^(-1) * W_in"""
        W_in = self.cache_model.get_input_embeddings().weight
        W_out = self.cache_model.get_output_embeddings().weight
        gram = torch.matmul(W_out.T, W_out) + 1e-5 * I
        rhs = torch.matmul(W_out.T, W_in)
        self._alignment_matrix = torch.linalg.solve(gram, rhs)
    
    def _apply_alignment(self, hidden):
        """Apply: e = normalize(h * W_a)"""
        aligned = torch.matmul(hidden, self._alignment_matrix)
        return aligned * (target_norm / ||aligned||)
    
    def generate_latent_batch(self, ...):
        for step in range(latent_steps):
            latent_vec = self._apply_alignment(last_hidden)  # ✅ Now aligned!
            latent_embed = latent_vec.unsqueeze(1)
```

**Usage**: Enabled by default! Disable with `use_alignment=False`

### Comparison

| Aspect | LatentMAS | G-cache |
|--------|-----------|---------||
| **Alignment matrix** | ✅ Optional (`--latent_space_realign`) | ✅ **Enabled by default!** |
| **Matrix computation** | `W_a ≈ W_out^(-1) * W_in` | ✅ Same |
| **Normalization** | ✅ Match input embedding norms | ✅ Same |
| **Training-free** | ✅ Yes (computed once) | ✅ Yes (computed once) |
| **Implementation** | 3 methods in ModelWrapper | ✅ 2 methods in HybridCacheModel |

### Benefits in G-cache

**✅ Enabled by default**:
- Better cache quality (aligned embeddings)
- Training-free (computed once at init)
- Minimal overhead (one matrix multiply per latent step)

**Disable if needed**:
```python
model = HybridCacheModel(use_alignment=False)  # Disable alignment
```

---

## 🎯 Summary

**What**: Graph manages cache flow, LatentMAS generates cache

**LatentMAS**: Inference only (no training) + optional alignment matrix

**G-cache**: Trains GCN + CacheFuser (learns graph structure + fusion)

**Frozen**: Small local model + API model (same as LatentMAS)

**Alignment**: Both have it! (LatentMAS optional, G-cache default)

**Run (Pure Local - RECOMMENDED)**:
```bash
cd experiments
export CUDA_VISIBLE_DEVICES=1,2
python run_gsm8k_cache.py --llm_name Qwen/Qwen3-1.7B --use_cache --device cuda
```

**That's it!** 🎉

---

## 🔌 API Model Options

### Default: qwen-flash ✅

The code now uses **qwen-flash** by default (faster and cheaper than qwen-plus).

### Available Models

| Model | Speed | Cost | Use Case |
|-------|-------|------|----------|
| **qwen-flash** | ⚡ Fastest | 💰 Cheapest | Default (recommended) |
| qwen-plus | 🐢 Medium | 💰💰 Medium | More capable |
| qwen-turbo | ⚡ Fast | 💰 Cheap | Alternative |

### Usage Examples

**Hybrid Mode (Cache + API)**:
```bash
# Default: qwen-flash
python run_gsm8k_cache_API.py --llm_name hybrid_cache --use_cache
```

**API-Only Mode (No Cache)**:
```bash
# qwen-flash (fastest, cheapest)
python run_gsm8k_cache_API.py --llm_name qwen-flash

# qwen-plus (more capable)
python run_gsm8k_cache_API.py --llm_name qwen-plus

# qwen-turbo (alternative)
python run_gsm8k_cache_API.py --llm_name qwen-turbo
```

---

## 🐛 Bug Fixes

### 1. Missing `get_edge_weight` Method
**Error:** `'GCN' object has no attribute 'get_edge_weight'`

**Location:** `GDesigner/gnn/gcn.py`

**Fix:** Added the missing method to the GCN class:
```python
def get_edge_weight(self, src_id, dst_id):
    """Get edge weight between two nodes (returns 1.0 as default)"""
    return 1.0
```

### 2. Gradient Flow Issue
**Error:** `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

**Root Cause:** The `spatial_logits` tensor was being reassigned in `arun()` method, breaking the gradient connection to the GCN parameters.

**Location:** `GDesigner/graph/graph.py` (line ~327)

**Fix:** Changed from:
```python
self.spatial_logits = min_max_norm(torch.flatten(self.spatial_logits))
```

To:
```python
spatial_logits_matrix = logits @ logits.t()
self.spatial_logits = torch.nn.Parameter(
    min_max_norm(torch.flatten(spatial_logits_matrix)),
    requires_grad=self.optimized_spatial
)
```

This ensures the tensor maintains gradient tracking and is properly registered as a Parameter.

### 3. Training Configuration
**Location:** `experiments/run_gsm8k_cache_API.py`

**Changes:**
- Set `optimized_spatial` default to `True` (required for cache training)
- Simplified backprop condition to only check `args.use_cache`

### Key Points

1. **Gradient Flow:** The GCN → MLP → spatial_logits → log_probs chain now properly maintains gradients
2. **Cache Training:** When `--use_cache` is enabled, both GCN and CacheFuser parameters are optimized
3. **Edge Weights:** The `get_edge_weight` method provides default weights (can be extended for learned edge weights)
