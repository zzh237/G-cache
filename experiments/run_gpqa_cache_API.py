"""
CacheDesigner with API for GPQA Diamond (Graduate-Level Science Q&A)
Tests graph-guided KV-cache on graduate-level science multiple choice questions
"""
import sys
import os

# Use only GPUs 1 and 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Add parent directory to path FIRST (before any imports)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
print(f"[DEBUG] Added to sys.path: {project_root}")
print(f"[DEBUG] Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

import argparse
import json
import time
import asyncio
from pathlib import Path
import torch
from typing import List

from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.graph.cache_graph import CacheGraph
from GDesigner.utils.globals import Time, Cost, PromptTokens, CompletionTokens
from run_gsm8k import load_result, dataloader, get_kwargs

# Import from local gcache_data folder
from gcache_data.data import load_gpqa_diamond
from gcache_data.utils import normalize_answer, extract_gpqa_answer


def load_gpqa(split: str = "test") -> List[dict]:
    """Load GPQA dataset"""
    dataset = list(load_gpqa_diamond(split=split))
    processed = []
    for item in dataset:
        processed.append({
            'task': item['question'],
            'answer': item['gold'],
        })
    return processed


def parse_args():
    parser = argparse.ArgumentParser(description="CacheDesigner for GPQA Diamond")
    parser.add_argument("--llm_name", type=str, default="hybrid_cache_v2",
                        help="LLM mode: hybrid_cache_v2 (small GPU+API v2), qwen-plus (API only), local_cache (local only)")
    parser.add_argument('--mode', type=str, default='FullConnected')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--domain', type=str, default="gpqa")
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4])
    parser.add_argument('--decision_method', type=str, default='FinalRefer')
    parser.add_argument('--optimized_spatial', type=bool, default=True, help='Enable spatial optimization (required for cache training)')
    
    # Cache arguments
    parser.add_argument('--use_cache', action='store_true', help='Enable cache')
    parser.add_argument('--fuse_method', type=str, default='concatenation',
                        choices=['weighted_sum', 'concatenation'],
                        help='Cache fusion method: weighted_sum (blend) or concatenation (stack)')
    parser.add_argument('--generation_mode', type=str, default='api_hint',
                        choices=['api_hint', 'hybrid', 'local'],
                        help='Generation mode: api_hint (API with text hint), hybrid (local+API), local (local only)')
    parser.add_argument('--hidden_dim', type=int, default=4096)
    parser.add_argument('--num_cache_layers', type=int, default=32)
    parser.add_argument('--add_role', action='store_true', help='Keep latent + role context (discard only input)')
    parser.add_argument('--latent_only', action='store_true', help='Keep only latent tokens (LatentMAS-style)')
    parser.add_argument('--latent_steps', type=int, default=10, help='Number of latent reasoning steps')
    parser.add_argument('--question_id', type=int, default=None, help='Run specific question by index (0-based)')
    
    args = parser.parse_args()
    return args


async def main():
    args = parse_args()
    
    print("="*80)
    print("🎓 CacheDesigner for GPQA Diamond")
    print("="*80)
    print(f"Mode: {args.llm_name}")
    print(f"Cache enabled: {args.use_cache}")
    if args.llm_name == "hybrid_cache_v2":
        print(f"Backend: Small local model (cache) + API (text) [V2]")
    elif args.llm_name == "hybrid_cache":
        print(f"Backend: Small local model (cache) + API (text) [V1 - deprecated]")
    elif args.llm_name in ["qwen-plus", "qwen-turbo"]:
        print(f"Backend: API only (text-based cache)")
    elif args.llm_name == "local_cache":
        print(f"Backend: Local model (real cache)")
    print("="*80)
    
    # Load dataset from HuggingFace cache
    print("📥 Loading GPQA Diamond from cache...")
    dataset = load_gpqa(split='test')
    print(f"✅ Loaded {len(dataset)} GPQA questions")
    
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    
    result_dir = Path(f"{GDesigner_ROOT}/result/gpqa")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"cache_API_{args.domain}_{current_time}.json"
    
    log_dir = Path(f"{GDesigner_ROOT}/log/{args.domain}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"cache_API_{args.domain}_{current_time}.log"
    
    # Setup dual logger (console + file) with custom path
    from GDesigner.utils.log import DualLogger
    logger = DualLogger.__new__(DualLogger)
    logger.log_path = log_file
    logger.log_file = open(log_file, 'w', encoding='utf-8')
    logger.original_stdout = sys.stdout
    logger.original_stderr = sys.stderr
    sys.stdout = logger
    sys.stderr = logger
    print(f"📝 Logging to: {log_file}")
    
    # Setup agents
    if args.use_cache:
        # Option 1: Diverse science agents (recommended for GPQA)
        # agent_names = ['ScienceExpertCacheV2', 'ScientificAnalystCacheV2', 'CodeAgentCacheV2', 'ResearcherCacheV2']
        # decision_method = 'FinalReferCacheV2'
        # print(f"✅ Using diverse cache-enabled agents: Science Expert, Scientific Analyst, Code, Researcher")
        
        # Option 2: Uniform agents (uncomment to use)
        agent_names = ['MathSolverCacheV2'] * sum(args.agent_nums)
        decision_method = 'FinalReferCacheV2'
        print(f"✅ Using cache-enabled agents (all Science Expert)")
    else:
        agent_names = ['MathSolver'] * sum(args.agent_nums)
        decision_method = args.decision_method  # Use standard decision method
        print(f"📝 Using text-only agents (baseline)")
    
    kwargs = get_kwargs(args.mode, len(agent_names))
    
    # Set generation mode and max_new_tokens
    if args.use_cache:
        if args.generation_mode == 'hybrid':
            print(f"⭐ Generation mode: HYBRID (local model + API refinement)")
        elif args.generation_mode == 'local':
            print(f"🖥️  Generation mode: LOCAL (local model only, real cache)")
        else:  # api_hint
            print(f"🌐 Generation mode: API_HINT (API with text hint)")
        
        # Different max_new_tokens for intermediate vs final agents
        node_kwargs = []
        for i, _ in enumerate(agent_names):
            is_final = (i == len(agent_names) - 1)  # Last agent before FinalRefer
            max_tokens = 2048 if is_final else 512  # Final: 2048, Intermediate: 512
            node_kwargs.append({
                "generation_mode": args.generation_mode,
                "max_new_tokens": max_tokens,
                "latent_only": args.latent_only,
                "add_role": args.add_role,
                "latent_steps": args.latent_steps
            })
        print(f"📏 Token limits: Intermediate agents=512, Final agent=2048")
        if args.add_role:
            print(f"✂️  ADD_ROLE mode: Keep {args.latent_steps} latent + role context tokens per agent")
        elif args.latent_only:
            print(f"✂️  LATENT_ONLY mode: Keep only {args.latent_steps} latent tokens per agent")
        else:
            print(f"💾 FULL mode: Keep all tokens (input + context + latent)")
        
        # Decision agent kwargs (judger gets more tokens)
        decision_kwargs = {
            "generation_mode": args.generation_mode,
            "max_new_tokens": 2048  # Increased for longer reasoning
        }
    else:
        node_kwargs = [{} for _ in agent_names]
        decision_kwargs = {}
    
    # Remove node_kwargs from kwargs if it exists to avoid duplicate
    kwargs.pop('node_kwargs', None)
    
    # Create CacheGraph
    graph = CacheGraph(
        domain="gpqa",
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=False,
        use_cache_communication=args.use_cache,
        hidden_dim=args.hidden_dim,
        num_cache_layers=args.num_cache_layers,
        fuse_method=args.fuse_method,  # NEW: Pass fuse_method
        node_kwargs=node_kwargs,
        decision_kwargs=decision_kwargs,
        **kwargs
    )
    
    print(f"\n🔗 Cache fusion method: {args.fuse_method.upper()}")
    if args.fuse_method == 'weighted_sum':
        print(f"   ⚖️  Weighted sum: Blend caches with learned weights (sequence length unchanged)")
    else:
        print(f"   🔗 Concatenation: Stack caches sequentially (sequence length increases)")
    
    # Print role information for each agent
    print("\n" + "="*80)
    print("🎭 AGENT ROLES ASSIGNMENT")
    print("="*80)
    for node_id, node in graph.nodes.items():
        print(f"Agent ID: {node_id}")
        print(f"  - Agent Class: {node.agent_name}")
        print(f"  - Role: {node.role}")
        print(f"  - Constraint (first 100 chars): {node.constraint[:100]}...")
        print()
    
    print("\n📊 ROLE CONNECTIONS (from ROLE_CONNECTION):")
    role_connections = graph.prompt_set.get_role_connection()
    
    # Create role to node mapping
    role_to_nodes = {}
    for node_id, node in graph.nodes.items():
        role = node.role
        if role not in role_to_nodes:
            role_to_nodes[role] = []
        role_to_nodes[role].append(node_id)
    
    for i, (from_role, to_role) in enumerate(role_connections, 1):
        from_nodes = role_to_nodes.get(from_role, [])
        to_nodes = role_to_nodes.get(to_role, [])
        from_ids = ", ".join(from_nodes) if from_nodes else "N/A"
        to_ids = ", ".join(to_nodes) if to_nodes else "N/A"
        print(f"  {i}. {from_role} ({from_ids}) → {to_role} ({to_ids})")
    
    print("\n🔗 ROLE ADJACENCY MATRIX:")
    print(f"  Shape: {graph.role_adj_matrix.shape}")
    print(f"  Edge index: {graph.role_adj_matrix}")
    
    # Print in readable format with node IDs
    print("\n  📊 Edge List (with Node IDs):")
    edge_index = graph.role_adj_matrix
    node_list = list(graph.nodes.keys())
    for i in range(edge_index.shape[1]):
        src_idx, dst_idx = edge_index[0, i].item(), edge_index[1, i].item()
        src_node_id = node_list[src_idx]
        dst_node_id = node_list[dst_idx]
        src_role = graph.nodes[src_node_id].role
        dst_role = graph.nodes[dst_node_id].role
        print(f"    Edge {i+1}: Node {src_idx} ({src_node_id}, {src_role}) → Node {dst_idx} ({dst_node_id}, {dst_role})")
    print("="*80 + "\n")
    graph.gcn.train()
    
    # Optimizer
    params = list(graph.gcn.parameters())
    if args.use_cache:
        params += list(graph.cache_fuser.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    # Handle single question mode
    if args.question_id is not None:
        print(f"\n🎯 Running single question mode: Question ID {args.question_id}")
        if args.question_id >= len(dataset):
            print(f"❌ Error: Question ID {args.question_id} out of range (dataset has {len(dataset)} questions)")
            return
        dataset = [dataset[args.question_id]]
        num_batches = 1
    else:
        num_batches = min(int(len(dataset) / args.batch_size), args.num_iterations)
        print(f"\n The num_batches: {num_batches}, the length of dataset: {len(dataset)}, the args.batch_size: {args.batch_size}, the args.num_iterations: {args.num_iterations}")
    
    total_solved, total_executed = 0, 0
    
    for i_batch in range(num_batches):
        # Clear CUDA cache to prevent memory corruption
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"\n{'='*80}")
        print(f"Batch {i_batch+1}/{num_batches}")
        print(f"{'='*80}")
        start_ts = time.time()
        
        current_batch = dataloader(dataset, args.batch_size, i_batch)
        if current_batch is None:
            break
        
        # Process batch
        answer_log_probs = []
        answers = []
        
        print(f"\n📦 [STEP 0] Loading batch tasks and preparing for execution...")
        for idx, record in enumerate(current_batch):
            task = record["task"]
            answer = record["answer"]
            answers.append(answer)
            input_dict = {"task": task}
            global_idx = i_batch * args.batch_size + idx
            print(f"\n📝 [STEP 0.{idx+1}] Question ID: {global_idx}")
            print(f"   Task {idx+1}/{len(current_batch)}: {task[:100]}...")
            print(f"   🎯 Expected answer: {answer}")
            # Reuse same graph for all tasks (models can't be deepcopied)
            answer_log_probs.append(asyncio.create_task(graph.arun(input_dict, args.num_rounds)))
        
        print(f"\n⏳ [STEP 0.1] Waiting for graph.arun() to complete for {len(answer_log_probs)} tasks...")
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        
        print(f"\n🏁 [STEP 13] Graph execution complete - received {len(raw_answers)} responses")
        # Compute metrics
        loss_list = []
        utilities = []
        data = load_result(result_file)
        
        print(f"\n📊 [STEP 14] Processing results and computing metrics...")
        for idx, (task, answer, log_prob, true_answer) in enumerate(zip(current_batch, raw_answers, log_probs, answers)):
            # print(f"\n🔍 [DEBUG] Extracting answer from Task {idx+1} response {str(answer[0])}...")
            extracted = extract_gpqa_answer(answer[0])
            predict_answer = normalize_answer(extracted) if extracted else None
            print(f"   Extracted: '{predict_answer}', Expected: '{true_answer}'")
            # Exact match - both already normalized
            is_solved = (predict_answer == true_answer) if predict_answer and true_answer else False
            total_solved += is_solved
            total_executed += 1
            accuracy = total_solved / total_executed
            
            utility = is_solved
            utilities.append(utility)
            single_loss = -log_prob * utility
            loss_list.append(single_loss)
            
            global_idx = i_batch * args.batch_size + idx
            print(f"   🔢 Question ID {global_idx}: Predicted={predict_answer}, Expected={true_answer}, Solved={is_solved}, -log_prob={-log_prob}")
            
            data.append({
                "Question_ID": global_idx,
                "Question": task["task"],
                "Answer": true_answer,
                "Response": answer,
                "Attempt answer": predict_answer,
                "Solved": is_solved,
                "Accuracy": accuracy,
                "Use Cache": args.use_cache,
                "Cache Method": args.llm_name if args.use_cache else "None"
            })
        
        with open(result_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        
        # Backprop
        if loss_list:
            total_loss = torch.mean(torch.stack(loss_list))
            print(f"\n 📉 [STEP 15] Total loss for batch: {total_loss.item():.4f}")
            if args.use_cache and total_loss.requires_grad:
                print(f"args.use_cache: {args.use_cache}, total_loss.requires_grad: {total_loss.requires_grad}")
                try:
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    optimizer.step()
                except RuntimeError as e:
                    print(f"   ⚠️ Backprop error: {e}")
            else:
                print(f"   ⚠️ Skipping backprop: Cache not used or loss does not require grad")
        else:
            print(f"   ⚠️ No losses computed for this batch, skipping backprop, setting total_loss to 0")
            total_loss = torch.tensor(0.0)
        
        print(f"\n📊 [BATCH/UPDATE {i_batch}] Summary:")
        print(f"   ⏱️  Batch time: {time.time() - start_ts:.3f}s")
        print(f"   📊 Cumulative Accuracy: {accuracy:.4f} ({total_solved}/{total_executed})")
        print(f"   📉 Batch Loss: {total_loss.item():.4f}")
        print(f"   ✅ Batch Solved: {sum(utilities)}/{len(utilities)}")
        print(f"   💰 Cost: {Cost.instance().value}")
        print(f"   📝 PromptTokens: {PromptTokens.instance().value}")
        print(f"   📝 CompletionTokens: {CompletionTokens.instance().value}")
    
    print(f"\n{'='*80}")
    print(f"✅ FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Solved: {total_solved}/{total_executed}")
    print(f"Results saved to: {result_file}")
    print(f"💰 Total Cost: {Cost.instance().value}")
    print(f"📝 Total PromptTokens: {PromptTokens.instance().value}")
    print(f"📝 Total CompletionTokens: {CompletionTokens.instance().value}")
    print(f"{'='*80}")
    
    # Close logger
    logger.close()
    print(f"📝 Log saved to: {logger.log_path}")


if __name__ == '__main__':
    asyncio.run(main())
