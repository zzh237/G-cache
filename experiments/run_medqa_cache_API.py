"""
CacheDesigner with API for MedQA (Medical Q&A)
Tests graph-guided KV-cache on medical multiple choice questions
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
from GDesigner.utils.globals import Time

from run_gsm8k import load_result, dataloader, get_kwargs


def load_medqa(json_path: str) -> List[dict]:
    """Load MedQA dataset from JSON file - matches LatentMAS format"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to standard format matching LatentMAS
    processed = []
    for item in data:
        question = item.get('query', '')
        raw_answer = str(item.get('answer', ''))
        options = item.get('options', [])
        
        # Map answer index to letter (matching LatentMAS)
        choice_map = {"0": "A", "1": "B", "2": "C", "3": "D"}
        answer = ""
        
        for idx, op in enumerate(options):
            if raw_answer in op:
                answer = choice_map[str(idx)].lower()
                break
        
        # Normalize answer to lowercase
        answer = normalize_answer(answer)
        
        processed.append({
            'task': question,
            'answer': answer,
            'options': options
        })
    
    return processed


def normalize_answer(ans: str) -> str:
    """Normalize answer to lowercase - matches LatentMAS"""
    if ans is None:
        return ""
    return ans.strip().lower()


def extract_medqa_answer(response: str) -> str:
    """Extract answer (A/B/C/D) from response - matches LatentMAS evaluation"""
    if not response:
        return ""
    
    response_upper = response.upper()
    
    # Look for "The answer is X" pattern (most common)
    if 'THE ANSWER IS' in response_upper:
        idx = response_upper.rfind('THE ANSWER IS')
        after = response_upper[idx+13:idx+20]
        for char in ['A', 'B', 'C', 'D']:
            if char in after:
                return normalize_answer(char)
    
    # Look for "answer is X" or "Answer: X"
    for pattern in ['ANSWER IS', 'ANSWER:']:
        if pattern in response_upper:
            idx = response_upper.rfind(pattern)
            after = response_upper[idx+len(pattern):idx+len(pattern)+10]
            for char in ['A', 'B', 'C', 'D']:
                if char in after:
                    return normalize_answer(char)
    
    # Look in last 100 characters for A/B/C/D
    last_part = response_upper[-100:]
    for char in ['A', 'B', 'C', 'D']:
        if char in last_part:
            return normalize_answer(char)
    
    return ""


def parse_args():
    parser = argparse.ArgumentParser(description="CacheDesigner for MedQA")
    parser.add_argument("--dataset_json", type=str, 
                       default="../datasets/LatentMAS/medqa.json")
    parser.add_argument("--llm_name", type=str, default="qwen-plus")
    parser.add_argument('--mode', type=str, default='FullConnected')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=5)
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4])
    parser.add_argument('--decision_method', type=str, default='FinalRefer')
    parser.add_argument('--optimized_spatial', action='store_true')
    
    # Cache arguments
    parser.add_argument('--use_cache', action='store_true', help='Enable cache')
    parser.add_argument('--hidden_dim', type=int, default=4096)
    parser.add_argument('--num_cache_layers', type=int, default=32)
    
    args = parser.parse_args()
    return args


async def main():
    args = parse_args()
    
    print("="*80)
    print("🏥 CacheDesigner for MedQA (Medical Q&A)")
    print("="*80)
    print(f"Cache enabled: {args.use_cache}")
    print(f"Model: {args.llm_name}")
    print("="*80)
    
    # Load dataset
    dataset = load_medqa(args.dataset_json)
    print(f"Loaded {len(dataset)} MedQA questions")
    
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    
    result_dir = Path(f"{GDesigner_ROOT}/result/medqa")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"cache_API_medqa_{current_time}.json"
    
    # Setup agents
    if args.use_cache:
        agent_names = ['MathSolverCache'] * sum(args.agent_nums)
        print(f"✅ Using cache-enabled agents")
    else:
        agent_names = ['MathSolver'] * sum(args.agent_nums)
        print(f"📝 Using text-only agents")
    
    kwargs = get_kwargs(args.mode, len(agent_names))
    
    # Create CacheGraph
    graph = CacheGraph(
        domain="gsm8k",  # Reuse gsm8k domain for now
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=False,
        use_cache_communication=args.use_cache,
        hidden_dim=args.hidden_dim,
        num_cache_layers=args.num_cache_layers,
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
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            if args.use_cache:
                realized_graph.cache_fuser = graph.cache_fuser
            
            task = record["task"]
            answer = record["answer"]
            answers.append(answer)
            input_dict = {"task": task}
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds)))
        
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        
        # Compute metrics
        loss_list = []
        utilities = []
        data = load_result(result_file)
        
        for task_record, answer, log_prob, true_answer in zip(current_batch, raw_answers, log_probs, answers):
            predict_answer = extract_medqa_answer(answer[0])
            is_solved = predict_answer == true_answer
            total_solved += is_solved
            total_executed += 1
            accuracy = total_solved / total_executed
            
            utility = float(is_solved)
            utilities.append(utility)
            single_loss = -log_prob * utility
            loss_list.append(single_loss)
            
            data.append({
                "Question": task_record["task"],
                "Answer": true_answer,
                "Response": answer,
                "Predicted": predict_answer,
                "Solved": is_solved,
                "Accuracy": accuracy,
                "Use Cache": args.use_cache,
            })
        
        with open(result_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        
        # Backprop
        total_loss = torch.mean(torch.stack(loss_list))
        if args.optimized_spatial or args.use_cache:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
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
