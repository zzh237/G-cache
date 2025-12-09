"""
CacheDesigner with API (uses your free Qwen API)
Simulates cache for testing structure without GPU
"""
import sys
import os

# Use only GPUs 1 and 2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
import copy
from typing import List

from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.graph.cache_graph import CacheGraph
from GDesigner.tools.reader.readers import JSONLReader
from GDesigner.utils.globals import Time, Cost, PromptTokens, CompletionTokens

# Import from local datasets folder (not HuggingFace datasets)
from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict
from run_gsm8k import load_result, dataloader, get_kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="CacheDesigner (supports hybrid/API/local modes)")
    # Use absolute path for dataset (relative to project root)
    default_dataset = os.path.join(project_root, "datasets/gsm8k/gsm8k.jsonl")
    parser.add_argument("--dataset_json", type=str, default=default_dataset)
    parser.add_argument("--llm_name", type=str, default="hybrid_cache",
                        help="LLM mode: hybrid_cache (small GPU+API), qwen-plus (API only), local_cache (local only)")
    parser.add_argument('--mode', type=str, default='FullConnected')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=5)
    parser.add_argument('--domain', type=str, default="gsm8k")
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4])
    parser.add_argument('--decision_method', type=str, default='FinalRefer')
    parser.add_argument('--optimized_spatial', type=bool, default=True, help='Enable spatial optimization (required for cache training)')
    
    # Cache arguments
    parser.add_argument('--use_cache', action='store_true', help='Enable cache')
    parser.add_argument('--generation_mode', type=str, default='api_hint', 
                        choices=['api_hint', 'hybrid', 'local'],
                        help='Generation mode: api_hint (API with text hint), hybrid (local+API), local (local only)')
    parser.add_argument('--hidden_dim', type=int, default=4096)
    parser.add_argument('--num_cache_layers', type=int, default=32)
    
    args = parser.parse_args()
    return args


async def main():
    args = parse_args()
    
    print("="*80)
    print("🚀 CacheDesigner")
    print("="*80)
    print(f"Mode: {args.llm_name}")
    print(f"Cache enabled: {args.use_cache}")
    if args.llm_name == "hybrid_cache":
        print(f"Backend: Small local model (cache) + API (text)")
    elif args.llm_name in ["qwen-plus", "qwen-turbo"]:
        print(f"Backend: API only (text-based cache)")
    elif args.llm_name == "local_cache":
        print(f"Backend: Local model (real cache)")
    print("="*80)
    
    # Load dataset
    dataset = JSONLReader.parse_file(args.dataset_json)
    dataset = gsm_data_process(dataset)
    
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    
    result_dir = Path(f"{GDesigner_ROOT}/result/gsm8k")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"cache_API_{args.domain}_{current_time}.json"
    
    # Setup agents
    if args.use_cache:
        agent_names = ['MathSolverCache'] * sum(args.agent_nums)
        print(f"✅ Using cache-enabled agents")
    else:
        agent_names = ['MathSolver'] * sum(args.agent_nums)
        print(f"📝 Using text-only agents (baseline)")
    
    kwargs = get_kwargs(args.mode, len(agent_names))
    
    # Set generation mode
    if args.use_cache:
        if args.generation_mode == 'hybrid':
            print(f"⭐ Generation mode: HYBRID (local model + API refinement)")
        elif args.generation_mode == 'local':
            print(f"🖥️  Generation mode: LOCAL (local model only, real cache)")
        else:  # api_hint
            print(f"🌐 Generation mode: API_HINT (API with text hint)")
        node_kwargs = [{"generation_mode": args.generation_mode} for _ in agent_names]
    else:
        node_kwargs = [{} for _ in agent_names]
    
    # Remove node_kwargs from kwargs if it exists to avoid duplicate
    kwargs.pop('node_kwargs', None)
    
    # Create CacheGraph
    graph = CacheGraph(
        domain="gsm8k",
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=False,
        use_cache_communication=args.use_cache,
        hidden_dim=args.hidden_dim,
        num_cache_layers=args.num_cache_layers,
        node_kwargs=node_kwargs,
        **kwargs
    )
    
    graph.gcn.train()
    
    # Optimizer
    params = list(graph.gcn.parameters())
    if args.use_cache:
        params += list(graph.cache_fuser.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    num_batches = min(int(len(dataset) / args.batch_size), args.num_iterations)
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
        
        for record in current_batch:
            task = record["task"]
            answer = record["answer"]
            answers.append(answer)
            input_dict = {"task": task}
            print(f"\n📝 Task: {task[:100]}...")  # Debug: show task
            print(f"🎯 Expected answer: {answer}")
            # Reuse same graph for all tasks (models can't be deepcopied)
            answer_log_probs.append(asyncio.create_task(graph.arun(input_dict, args.num_rounds)))
        
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        
        # Compute metrics
        loss_list = []
        utilities = []
        data = load_result(result_file)
        
        for task, answer, log_prob, true_answer in zip(current_batch, raw_answers, log_probs, answers):
            predict_answer = gsm_get_predict(answer[0])
            is_solved = float(predict_answer) == float(true_answer)
            total_solved += is_solved
            total_executed += 1
            accuracy = total_solved / total_executed
            
            utility = is_solved
            utilities.append(utility)
            single_loss = -log_prob * utility
            loss_list.append(single_loss)
            
            data.append({
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
            if args.use_cache and total_loss.requires_grad:
                try:
                    optimizer.zero_grad()
                    total_loss.backward()
                    # Clip gradients to prevent NaN/Inf
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    optimizer.step()
                except RuntimeError as e:
                    print(f"⚠️ Backprop error: {e}")
                    print(f"Skipping optimization step")
        else:
            total_loss = torch.tensor(0.0)
        
        print(f"⏱️  Batch time: {time.time() - start_ts:.3f}s")
        print(f"📊 Accuracy: {accuracy:.4f}")
        print(f"📉 Loss: {total_loss.item():.4f}")
        print(f"✅ Solved: {total_solved}/{total_executed}")
    
    print(f"\n{'='*80}")
    print(f"✅ FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Solved: {total_solved}/{total_executed}")
    print(f"Results saved to: {result_file}")
    print(f"{'='*80}")


if __name__ == '__main__':
    asyncio.run(main())
