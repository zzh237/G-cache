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
