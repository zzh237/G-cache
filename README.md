# CacheDesigner: Graph-Guided Cache Communication for Multi-Agent LLMs

Combines **GDesigner's** graph topology learning with **LatentMAS's** KV-cache communication.

---

## 🚀 Quick Start (Use Your Free API)

```bash
cd G-cache/experiments

# Run with cache (recommended)
python run_gsm8k_cache_API.py --use_cache --optimized_spatial --batch_size 2

# Run without cache (baseline)
python run_gsm8k_cache_API.py --optimized_spatial --batch_size 2
```

**Uses your free Qwen API - no GPU needed!**

---

## 📁 Project Structure

```
G-cache/
├── GDesigner/                      # GDesigner backbone
│   ├── graph/
│   │   ├── graph.py               # Original Graph
│   │   └── cache_graph.py         # NEW: Cache-enabled Graph
│   ├── llm/
│   │   ├── gpt_chat.py            # Original API LLM
│   │   └── gpt_chat_cache_api.py  # NEW: Cache-enabled API LLM
│   └── agents/
│       ├── math_solver.py         # Original agent
│       └── math_solver_cache.py   # NEW: Cache-enabled agent
├── experiments/
│   ├── run_gsm8k.py               # Original runner
│   ├── run_gsm8k_cache_API.py     # NEW: API + cache (use this!)
│   └── run_gsm8k_cache_WORKING.py # NEW: vLLM + cache (needs GPU)
└── .env                            # Your API credentials
```

---

## 🎯 Two Versions

### Option 1: API Version (FREE - Recommended) ✅

**What it uses:**
- Your free Qwen API for text generation
- Simulated cache for testing structure

**Pros:**
- ✅ FREE (uses your API)
- ✅ No GPU needed
- ✅ Tests full structure
- ✅ Easy to run

**Cons:**
- ⚠️ Cache is simulated (not real KV-cache)
- ⚠️ ~0-5% improvement

**Run:**
```bash
python run_gsm8k_cache_API.py --use_cache --optimized_spatial
```

### Option 2: vLLM Version (Needs GPU) 

**What it uses:**
- Local model with vLLM
- Real KV-cache extraction

**Pros:**
- ✅ Real cache benefits
- ✅ ~10-25% improvement

**Cons:**
- ❌ Needs GPU (16GB+ VRAM)
- ❌ Costs money ($1-2/hour)

**Run:**
```bash
python run_gsm8k_cache_WORKING.py --use_cache --device cuda
```

---

## 🔍 Debug Output

When running, you'll see:

```
🤖 [AGENT abc1] Executing...
   🔍 Checking for predecessor caches...
   ✅ Found fused cache from predecessors

🌐 [API CALL] Calling Qwen API...
   Status: 200
   ✅ SUCCESS: Received 245 characters

📦 [CACHE] agen_with_cache called
   🔄 Using cached context from 32 layers
   ✅ Cache generated: 32 layers

💾 [GRAPH] Storing cache for node abc1
   Cache layers: 32

🔄 [GRAPH] Getting fused cache for node xyz2
   ✅ Found cache from abc1
   🧪 Fusing 2 caches
```

**Success indicators:**
1. ✅ API calls succeed
2. ✅ Cache generated
3. ✅ Cache stored
4. ✅ Cache retrieved
5. ✅ Cache fused
6. ✅ Cache used

---

## 🔧 Setup

### 1. Check .env file exists:
```bash
cat .env
# Should show:
# BASE_URL=https://idealab-external.alibaba-inc.com/api/openai/v1
# API_KEY=c3a588a3e15983ab2dc8facefecc5bd9
```

### 2. Install dependencies:
```bash
pip install torch transformers aiohttp python-dotenv tenacity
```

### 3. Run:
```bash
cd experiments
python run_gsm8k_cache_API.py --use_cache --optimized_spatial
```

---

## 📊 What's Different from GDesigner?

| Aspect | GDesigner | CacheDesigner |
|--------|-----------|---------------|
| **Communication** | Text only | Text + Cache |
| **Agent class** | `MathSolver` | `MathSolverCache` |
| **LLM class** | `GPTChat` | `GPTChatCacheAPI` |
| **Graph class** | `Graph` | `CacheGraph` |
| **Cache extraction** | ❌ No | ✅ Yes |
| **Cache fusion** | ❌ No | ✅ Yes |
| **Cache injection** | ❌ No | ✅ Yes |

---

## 🐛 Troubleshooting

### Error: "403 Client Error"
```bash
# Check .env file
cat .env
# Make sure BASE_URL and API_KEY are set correctly
```

### Error: "No module named 'GDesigner'"
```bash
# Make sure you're in the right directory
cd /Users/bleachvex/Downloads/projects/G-cache/experiments
```

### No cache operations shown
```bash
# Make sure --use_cache flag is set
python run_gsm8k_cache_API.py --use_cache  # ← Must have this!
```

---

## 📝 Key Files

**To run experiments:**
- `experiments/run_gsm8k_cache_API.py` - Main runner (use this!)

**Core implementation:**
- `GDesigner/llm/gpt_chat_cache_api.py` - Cache-enabled LLM
- `GDesigner/agents/math_solver_cache.py` - Cache-enabled agent
- `GDesigner/graph/cache_graph.py` - Cache-enabled graph

**Configuration:**
- `.env` - Your API credentials

---

## 🎯 Summary

**What is CacheDesigner?**
- GDesigner (graph topology) + LatentMAS (cache communication)

**Which version should I use?**
- API version (free, simulated cache)

**Do I need GPU?**
- No (for API version)

**Does it use my API?**
- Yes (same as G-Designer)

**How do I run it?**
```bash
cd experiments
python run_gsm8k_cache_API.py --use_cache --optimized_spatial
```

**That's it!** 🎉
