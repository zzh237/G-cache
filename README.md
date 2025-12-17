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

## 🔄 Cache Flow Explained

### What Are Agent IDs?

Agent IDs like `3KsM`, `QXan`, `5Hp5` are randomly generated 4-character identifiers for each agent node in the graph.

### Agent Graph Structure

```
     3KsM (MathSolver) → cache: 2377 tokens
     QXan (MathSolver) → cache: 474 tokens  
     5Hp5 (MathSolver) → cache: 181 tokens
       ↓ ↓ ↓ (all feed into)
     5xcm (Decision Maker)
       ↓ receives: 3KsM's cache (first one)
       ↓ generates: final answer
```

### What Comes From Cache vs LLM?

| Component | What It Does | Visible? | Example |
|-----------|-------------|----------|----------|
| **Cache (GPU)** | Stores reasoning as tensors | ❌ No | `torch.Size([1, 2, 2377, 128])` |
| **API (qwen-flash)** | Generates text responses | ✅ Yes | "Let's solve step by step..." |
| **Cache Fusion** | Combines multiple caches | ❌ No | Internal process |
| **Graph Structure** | Routes cache between agents | ✅ Yes | Shown in logs |

**Key Insight**: Cache = "compressed thoughts" that agents pass to each other. API converts these thoughts into readable text.

### ⚠️ Important: How Cache is Used

**Cache is NOT directly used by API!** Here's the actual flow:

1. **Small local model (GPU)** generates KV-cache tensors
2. **Cache is converted to text context**: `"[Using 28 layers of cached reasoning from previous agents, 2377 tokens]"`
3. **This text is prepended to the API prompt**
4. **API (qwen-flash) generates response** based on the text context (NOT the raw cache)

**Why?** APIs don't support KV-cache input - they only accept text. The cache serves as a "hint" to the API about what previous agents computed.

### Cache Flow Example

**Without Cache (Cold Start)**:
```
Agent 3KsM:
  🆕 No input cache
  🧠 Generates from scratch
  📝 Output: "Let's solve... Total = $64"
  💾 Stores cache: 2377 tokens
```

**With Cache (Warm Start)**:
```
Agent 5xcm:
  🔗 Receives cache from 3KsM (2377 tokens)
  🧠 Continues from 3KsM's reasoning
  📝 Output: "After reviewing... The answer is 64"
  💾 Stores cache: 3724 tokens (grew from 2377)
```

### Sequence Length Mismatch

**Problem**: Can't fuse caches of different lengths (2377 ≠ 474 ≠ 181)

**Solution (LatentMAS-proven)**: Use first cache only
- Takes 3KsM's cache (2377 tokens)
- Ignores QXan and 5Hp5 caches
- Sequential passing (proven by LatentMAS paper)

### Debug Markers

Look for these in output:
```
🔗 [CACHE] Using input cache with 28 layers     ← Cache is being used
🆕 [CACHE] No input cache - generating from scratch  ← No cache (first agent)
✅ [CACHE] Generated cache with 28 layers, seq_len=2377  ← Cache created
🤖 [API] Calling qwen-flash to generate text...  ← API generates text
📝 [API] Generated 1234 characters of text       ← Text output
```

### Cache Structure Details

**What is `past_key_values`?**

Type: Tuple of tuples (NOT text, NOT string)

```python
past_key_values = (
    (key_tensor_0, value_tensor_0),    # Layer 0: [1, 2, 2377, 128]
    (key_tensor_1, value_tensor_1),    # Layer 1: [1, 2, 2377, 128]
    # ... 26 more layers ...
    (key_tensor_27, value_tensor_27)   # Layer 27: [1, 2, 2377, 128]
)
```

**Total**: 28 layers × 2 tensors (key + value) = 56 tensors

**Tensor shape**: `[batch=1, heads=2, seq_len=2377, hidden_dim=128]`
- Data type: `torch.bfloat16` (16-bit floating point)
- Device: `cuda:0` (GPU memory)

**Why Can't API Use This?**

API endpoints only accept TEXT, not binary tensors.

**Solution**: Convert to text hint
```python
# Cache (tensor): 56 tensors × millions of numbers
past_key_values = ((tensor1, tensor2), ...)

# Converted to text:
cache_info = "[Using 28 layers of cached reasoning, 2377 tokens]"

# Sent to API:
prompt = cache_info + "\n\n" + original_prompt
```

**Analogy**: Cache = full book (tensor data), API receives = "This person read a 300-page book" (text hint)

### Two Ways to Generate Text

**Method 1: Local Model with Cache Tensors (REAL cache usage)**
```python
# Function: generate_text_batch()
# Uses: Local GPU model (Qwen2.5-1.5B)
# Cache: Passed DIRECTLY as tensors to model.generate()

outputs = model.generate(
    input_ids=input_ids,
    past_key_values=cache_tensors,  # ← REAL cache usage!
    ...
)
```

**Method 2: API with Text Hint (Simulated cache)**
```python
# Function: generate_text_batch_api()
# Uses: API (qwen-flash)
# Cache: Converted to text hint, prepended to prompt

prompt = "[Using 28 layers of cache, 2377 tokens]\n\n" + original_prompt
response = api.chat.completions.create(
    messages=[{"content": prompt}],  # ← Text hint only
    ...
)
```

**Comparison**:

| Aspect | Local (Real Cache) | API (Text Hint) |
|--------|-------------------|------------------|
| **Cache Format** | Tensor objects | Text string |
| **Cache Usage** | Direct (model.generate) | Indirect (text context) |
| **GPU Required** | Yes (~4GB) | No |
| **Speed** | Faster (cache reuse) | Slower (no cache reuse) |
| **Quality** | Good (small model) | Better (large API model) |
| **Cost** | GPU cost | API cost |
| **Best For** | Pure local | API only | **Production** ⭐ |

**Key Difference**:
- **Local**: `model.generate(past_key_values=cache_tensors)` ← Tensors used directly
- **API**: `api.create(messages=[text_hint])` ← Only text hint sent

**Why API Can't Use Cache Tensors**:
- API endpoints only accept JSON/text
- Can't send binary tensor data over HTTP
- Solution: Convert cache metadata to text hint

**Method 3: TRUE Hybrid (BEST - combines both!)**
```python
# Function: generate_text_batch_hybrid()
# Step 1: Local model uses cache tensors (fast)
# Step 2: API refines output (high quality)

local_text, new_cache = model.generate_text_batch(
    input_ids, past_key_values=cache_tensors  # ← Real cache!
)
api_text, _ = await api.generate_text_batch_api(
    messages_with_local_context  # ← API refines
)
return api_text, new_cache  # Best of both!
```

**Trade-off**:
- Local only: Real cache ✅ but small model ❌
- API only: Large model ✅ but no cache ❌
- **Hybrid**: Real cache ✅ + Large model ✅ = BEST ⭐

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

## 🎮 Generation Modes

### Three Ways to Generate Text

#### 1. API_HINT (Default)

**Method**: `generate_text_batch_api`

```bash
python run_gsm8k_cache_API.py --llm_name hybrid_cache --use_cache --generation_mode api_hint
```

**Flow**: Cache → Text hint → API

**Pros**: Simple | **Cons**: No real cache usage

#### 2. HYBRID (Best Quality) ⭐

**Method**: `generate_text_batch_hybrid`

```bash
python run_gsm8k_cache_API.py --llm_name hybrid_cache --use_cache --generation_mode hybrid
```

**Flow**: Cache → Local model (uses cache) → API refines

**Pros**: Real cache + API quality | **Cons**: Slower

#### 3. LOCAL (Pure Local)

**Method**: `generate_text_batch`

```bash
python run_gsm8k_cache_API.py --llm_name hybrid_cache --use_cache --generation_mode local
```

**Flow**: Cache → Local model (uses cache) → Output

**Pros**: Real cache, no API cost | **Cons**: Lower quality

### Comparison

| Mode | Cache Usage | Quality | Speed | API Cost |
|------|-------------|---------|-------|----------|
| **api_hint** | Text hint | Good | Medium | Yes |
| **hybrid** | Real tensors | **Best** ⭐ | Slow | Yes |
| **local** | Real tensors | OK | Fast | No |

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


The API call flow 
FinalRefer → HybridCacheLLM.agen() → hybrid_model.generate_text_batch_api() 
→ AsyncOpenAI.chat.completions.create(model="qwen-flash", max_tokens=4096)
→ Your .env BASE_URL + API_KEY


Step-by-Step Call Chain:
1. run_gpqa_cache_API.py (line ~200)
   answer_log_probs.append(asyncio.create_task(graph.arun(input_dict, args.num_rounds)))
   ↓

2. graph.arun() (graph.py line 374)
   - Constructs graph connections
   - Executes nodes in topological order
   - Line 402: await self.nodes[current_node_id].async_execute(input)
   ↓

3. node.async_execute() (node.py line 157)
   - Gets spatial_info and temporal_info
   - Line 162: self._async_execute(input, spatial_info, temporal_info)
   ↓

4. MathSolverCache._async_execute() (math_solver_cache.py line 101)
   - Your STEP 3-11 logs
   - Line 175: return response
   ↓

5. Back to node.async_execute() (node.py line 168)
   - Stores result in self.outputs
   - Returns self.outputs
   ↓

6. Back to graph.arun() (graph.py line 418-420)
   - After all nodes execute, calls decision_node
   - Returns: final_answers, log_probs
   ↓

7. Back to run_gpqa_cache_API.py (line ~210)
   raw_results = await asyncio.gather(*answer_log_probs)
   raw_answers, log_probs = zip(*raw_results)




# CacheDesigner Workflow Tracking Guide

## Complete Execution Flow with Step Numbers

### High-Level Overview
```
User Task → Graph Execution → Multi-Agent Processing → Cache Communication → Final Response
```

### Detailed Step-by-Step Flow

#### **STEP 1-2: Graph Initialization**
- **Location:** `cache_graph.py:169`
- **Print:** `🚀 [STEP 1] CacheGraph.arun() - Starting graph execution`
- **What happens:** 
  - Clears node caches
  - Sets graph reference on all nodes
  - Calls parent Graph.arun()

#### **STEP 3: Agent Execution Starts**
- **Location:** `math_solver_cache.py:113`
- **Print:** `🎯 [STEP 3] MathSolverCache._async_execute() - Executing node {node_id}`
- **What happens:** Agent begins processing its task

#### **STEP 4: Cache Retrieval**
- **Location:** `cache_graph.py:127`
- **Print:** `🔄 [STEP 4] CacheGraph.get_fused_cache() - Getting fused cache for node {node_id}`
- **What happens:** 
  - Checks for spatial predecessors
  - Fuses caches from predecessor nodes
  - Returns None for root nodes (no predecessors)

#### **STEP 5: Cache Status**
- **Location:** `math_solver_cache.py:118`
- **Print:** `📥 [STEP 5] MathSolverCache - Received fused cache: {True/False}`
- **What happens:** Agent receives (or doesn't receive) cache from predecessors

#### **STEP 5a: Prompt Building**
- **Location:** `math_solver_cache.py:121`
- **Print:** `📝 [STEP 5a] MathSolverCache._process_inputs() - Building prompt with context`
- **What happens:**
  - Combines task + spatial context + temporal context
  - Builds system and user prompts
  - Creates messages array

#### **STEP 6: LLM Call**
- **Location:** `math_solver_cache.py:126`
- **Print:** `🧠 [STEP 6] MathSolverCache - Calling llm.agen_with_cache() (Goal: Generate reasoning cache + text response)`
- **What happens:** Agent calls LLM to generate response with cache

#### **STEP 7: Cache Generation Starts**
- **Location:** `llm_cache_hybrid.py:81`
- **Print:** `📦 [STEP 7] HybridCacheLLM.agen_with_cache() - Starting cache generation`
- **What happens:** LLM wrapper begins processing

#### **STEP 7a: Message Conversion**
- **Location:** `llm_cache_hybrid.py:84`
- **Print:** `💬 [STEP 7a] Converting messages to text prompt...`
- **What happens:** Converts chat messages to text format

#### **STEP 7b: Tokenization**
- **Location:** `llm_cache_hybrid.py:88`
- **Print:** `🔤 [STEP 7b] Tokenizing prompt...`
- **What happens:**
  - Clears CUDA cache
  - Tokenizes prompt
  - Validates token IDs
  - Moves tensors to device

#### **STEP 8: Latent Cache Generation**
- **Location:** `llm_cache_hybrid.py:135`
- **Print:** `🔗 [STEP 8] HybridCacheLLM - Calling hybrid_model.generate_latent_batch()`
- **What happens:** Calls model to generate KV-cache

#### **STEP 8a: Cache Model Forward Pass**
- **Location:** `hybrid_cache_model.py:102`
- **Print:** `🧠 [STEP 8a] HybridCacheModel.generate_latent_batch() - Generating cache using small local model`
- **What happens:**
  - Forward pass through local model
  - 10 latent reasoning steps
  - Builds KV-cache (no text yet!)

#### **STEP 9: Text Generation**
- **Location:** `llm_cache_hybrid.py:145`
- **Print:** `📝 [STEP 9] HybridCacheLLM - Generating text with mode: {mode}`
- **What happens:** Chooses generation mode (hybrid/local/api_hint)

#### **STEP 9a: Local Model Generation** (if mode=local or hybrid)
- **Location:** `hybrid_cache_model.py:54`
- **Print:** `🎯 [STEP 9a] HybridCacheModel.generate_text_batch() - Generating text using cache TENSORS directly`
- **What happens:**
  - Uses KV-cache from STEP 8
  - Calls HuggingFace model.generate()
  - Generates text tokens autoregressively

#### **STEP 9b: API Generation** (if mode=api_hint)
- **Location:** `hybrid_cache_model.py:130`
- **Print:** `🔄 [STEP 9b] HybridCacheModel.generate_text_batch_api() - Converting cache to text context`
- **What happens:**
  - Converts cache to text hint
  - Calls API (qwen-plus/qwen-flash)
  - Returns API response

#### **STEP 10: Cache Storage**
- **Location:** `math_solver_cache.py:136`
- **Print:** `💾 [STEP 10] MathSolverCache - Calling store_node_cache() for {n} successors`
- **What happens:** Stores generated cache for successor nodes to use

#### **STEP 11: Response Return**
- **Location:** `math_solver_cache.py:141`
- **Print:** `🎯 [STEP 11] MathSolverCache - Returning final response from node {node_id}`
- **What happens:** Agent returns its final text response

#### **STEP 12: Graph Aggregation**
- **Location:** `run_gsm8k_cache_API.py:175`
- **Print:** `⏳ [STEP 12] Waiting for graph.arun() to complete for {n} tasks...`
- **What happens:** Waits for all agents to finish

#### **STEP 13: Results Collection**
- **Location:** `run_gsm8k_cache_API.py:178`
- **Print:** `🏁 [STEP 13] Graph execution complete - received {n} responses`
- **What happens:** 
  - Collects all agent responses
  - Shows preview of each response
  - **THIS IS WHERE YOUR OUTPUT COMES FROM!**

#### **STEP 14: Metrics Computation**
- **Location:** `run_gsm8k_cache_API.py:183`
- **Print:** `📊 [STEP 14] Processing results and computing metrics...`
- **What happens:**
  - Extracts predicted answers
  - Computes accuracy
  - Saves results to JSON

---

## Understanding Your Output

The output you see:
```
The task is: Kylar went to the store...
At the same time, the output of other agents is as follows:
7LdC: Let's solve this step by step...
5qxQ: Let's analyze the problem...
...
################response:After reviewing the analysis...
```

**Comes from STEP 13** - This is the **final aggregated response** from the graph after all agents have completed.

### Where Each Part Comes From:

1. **"The task is:"** - From the prompt template in `_process_inputs()` (STEP 5a)
2. **"output of other agents"** - From spatial context (other agents in same round)
3. **"7LdC:", "5qxQ:", etc.** - Individual agent responses (STEP 11 from each agent)
4. **"################response:"** - Final decision/aggregation (from decision agent or final node)

### Is it from Cache or Direct Model?

The text is generated in **STEP 9**:
- **If mode=local:** Direct from local model using cache tensors (STEP 9a)
- **If mode=api_hint:** From API with cache converted to text hint (STEP 9b)
- **If mode=hybrid:** Local model first, then API refinement

The cache itself (KV-cache tensors) is generated in **STEP 8a** and contains the "reasoning" in latent space, but the actual text you see is generated in **STEP 9**.

---

## Quick Reference

| Step | Component | What You See |
|------|-----------|--------------|
| 1-2 | Graph Init | Graph setup messages |
| 3-6 | Agent Setup | Agent execution start |
| 7-7b | Tokenization | Prompt processing |
| 8-8a | Cache Gen | KV-cache creation (latent reasoning) |
| 9-9b | Text Gen | **Actual text generation** |
| 10-11 | Cache Store | Cache saved for next nodes |
| 12-13 | Aggregation | **Final output collection** |
| 14 | Metrics | Accuracy computation |

**Your output appears at STEP 13** - it's the final aggregated response after all agents complete their processing.


我来详细解释Transformer模型中的forward pass和generation过程，包括所有维度信息。

1. Hidden State H(l) 的计算过程
H(l) ∈ R^(1×690×2560) 是这样得到的：

完整的Transformer Layer计算：

输入: H(l-1) ∈ R^(B×T×D)  (B=1, T=690, D=2560)

步骤1: Self-Attention
------
Q = H(l-1) @ W_Q  →  Q ∈ R^(1×690×2560)
K = H(l-1) @ W_K  →  K ∈ R^(1×690×2560)  
V = H(l-1) @ W_V  →  V ∈ R^(1×690×2560)

步骤2: Multi-Head Attention (假设32个heads)
------
将Q,K,V分成32个heads:
Q_heads ∈ R^(1×32×690×80)  (2560/32=80)
K_heads ∈ R^(1×32×690×80)
V_heads ∈ R^(1×32×690×80)

步骤3: Scaled Dot-Product Attention
------
scores = (Q_heads @ K_heads^T) / sqrt(80)  →  scores ∈ R^(1×32×690×690)

步骤4: Apply Attention Mask (这里回答你的mask问题)
------
# attention_mask ∈ R^(1×690), 值为0或1
# 扩展为 R^(1×1×690×690) 的causal mask
mask = (1 - attention_mask) * (-10000)  # 0变0, 1变-10000
scores = scores + mask  # Element-wise加法!

步骤5: Softmax (在sequence维度上)
------
attn_weights = softmax(scores, dim=-1)  # 在最后一维(690)上做softmax
# attn_weights ∈ R^(1×32×690×690)

步骤6: Apply to Values
------
attn_output = attn_weights @ V_heads  →  R^(1×32×690×80)

步骤7: Concat heads
------
attn_output = concat(attn_output)  →  R^(1×690×2560)

步骤8: Output projection
------
O = attn_output @ W_O  →  O ∈ R^(1×690×2560)

步骤9: Add & Norm
------
H_attn = LayerNorm(H(l-1) + O)  →  R^(1×690×2560)

步骤10: FFN
------
H(l) = LayerNorm(H_attn + FFN(H_attn))  →  R^(1×690×2560)


Copy
所以是的，需要QKV！ 每一层都要计算QKV。

2. Attention Mask 的作用机制
# attention_mask 示例: [1, 1, 1, 0, 0]  (前3个token有效，后2个padding)

# 步骤1: 转换为additive mask
mask = (1 - attention_mask) * (-10000)
# 结果: [0, 0, 0, -10000, -10000]

# 步骤2: 扩展到attention scores维度
# scores ∈ R^(B×heads×T×T)
# mask扩展为 R^(B×1×1×T) 然后broadcast

# 步骤3: Element-wise 加法 (不是乘法!)
scores = scores + mask
# 例如某个位置的score是5.2，如果mask是-10000，则变成-9994.8

# 步骤4: Softmax
attn_weights = softmax(scores, dim=-1)
# exp(-10000) ≈ 0，所以被mask的位置权重≈0

关键点：

Mask是 加法 ，不是乘法

Softmax是在 sequence level （最后一维）上做的

被mask的位置经过softmax后权重≈0，相当于"knock out"

3. Generation过程详解
Prefill阶段（第一次forward）

输入: input_ids ∈ R^(1×T)  例如 T=10
past_kv = None

步骤1: Embedding
------
H(0) = Embedding(input_ids)  →  H(0) ∈ R^(1×10×2560)

步骤2: 通过所有Transformer layers
------
for layer in layers:
    H(l) = TransformerLayer(H(l-1))  # 如上面详细过程
    # 同时生成并保存 K(l), V(l) ∈ R^(1×32×10×80)

最终: H(L) ∈ R^(1×10×2560)  (L是最后一层)

步骤3: 取最后一个token的hidden state
------
h_last = H(L)[:, -1, :]  →  h_last ∈ R^(1×2560)

步骤4: 生成logits
------
logits = h_last @ W_lm_head  →  logits ∈ R^(1×vocab_size)

步骤5: 采样next token
------
next_token_id = sample(logits)  →  scalar

步骤6: 保存KV cache
------
past_kv = [(K(1), V(1)), (K(2), V(2)), ..., (K(L), V(L))]
# 每个 K(l), V(l) ∈ R^(1×32×10×80)

Autoregressive阶段（后续生成）

输入: next_token_id (scalar)
past_kv = [(K, V) for each layer]  # K,V ∈ R^(1×32×T_past×80)

步骤1: Embedding
------
h_new = Embedding(next_token_id)  →  h_new ∈ R^(1×1×2560)

步骤2: 通过所有layers (使用KV cache!)
------
for layer_idx, layer in enumerate(layers):
    K_past, V_past = past_kv[layer_idx]  # R^(1×32×T_past×80)
    
    # 只计算新token的Q,K,V
    Q_new = h_new @ W_Q  →  R^(1×32×1×80)
    K_new = h_new @ W_K  →  R^(1×32×1×80)
    V_new = h_new @ W_V  →  R^(1×32×1×80)
    
    # Concat with past
    K_full = concat([K_past, K_new], dim=2)  →  R^(1×32×(T_past+1)×80)
    V_full = concat([V_past, V_new], dim=2)  →  R^(1×32×(T_past+1)×80)
    
    # Attention (Q只有1个token，但attend to所有past tokens)
    scores = Q_new @ K_full^T  →  R^(1×32×1×(T_past+1))
    attn = softmax(scores, dim=-1)
    output = attn @ V_full  →  R^(1×32×1×80)
    
    # 继续FFN等...
    h_new = TransformerLayer(h_new, K_full, V_full)  →  R^(1×1×2560)
    
    # 更新cache
    past_kv[layer_idx] = (K_full, V_full)

步骤3: 生成下一个token
------
h_last = h_new[:, -1, :]  →  R^(1×2560)
logits = h_last @ W_lm_head  →  R^(1×vocab_size)
next_token = sample(logits)

4. 关于你的代码中的latent generation
在你的代码中：

# generate_latent_batch() - 只生成cache，不生成token
outputs = self.cache_model(
    inputs_embeds=latent_embed,  # R^(1×1×2560)
    past_key_values=past,
    use_cache=True,
    output_hidden_states=True,
    return_dict=True,
)
# 注意: 这里没有调用 .generate()
# 所以没有 outputs.sequences 属性!
# 只有 outputs.hidden_states 和 outputs.past_key_values

你的错误原因：

output_seq_len = outputs.sequences.shape[1]  # ❌ 错误!
# CausalLMOutputWithPast 没有 sequences 属性
# sequences 只在 .generate() 的返回值中存在