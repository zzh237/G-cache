"""
CacheDesigner experiment runner for GSM8K
Minimal modification of GDesigner's run_gsm8k.py to add cache communication
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
from GDesigner.graph.cache_graph import CacheGraph  # Use CacheGraph instead of Graph
from GDesigner.tools.reader.readers import JSONLReader
from GDesigner.utils.globals import Time, Cost, PromptTokens, CompletionTokens
from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict

# Import helper functions from original
from run_gsm8k import load_result, dataloader, get_kwargs

def parse_args():
    parser = argparse.ArgumentParser(description="CacheDesigner Experiments on gsm8k")
    parser.add_argument("--dataset_json", type=str, default="datasets/gsm8k/gsm8k.jsonl")
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument('--mode', type=str, default='FullConnected')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--domain', type=str, default="gsm8k")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'])
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4])
    parser.add_argument('--decision_method', type=str, default='FinalRefer')
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--optimized_temporal', action='store_true')
    
    # Cache-specific arguments
    parser.add_argument('--use_cache', action='store_true', help='Enable cache-to-cache communication')
    parser.add_argument('--hidden_dim', type=int, default=4096, help='Hidden dimension for cache')
    parser.add_argument('--num_cache_layers', type=int, default=32, help='Number of cache layers')
    parser.add_argument('--cache_lr', type=float, default=0.01, help='Learning rate for cache fuser')
    
    args = parser.parse_args()
    return args

async def main():
    args = parse_args()
    
    # Load dataset
    dataset = JSONLReader.parse_file(args.dataset_json)
    dataset = gsm_data_process(dataset)
    
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    
    result_dir = Path(f"{GDesigner_ROOT}/result/gsm8k")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"cache_{args.domain}_{args.llm_name}_{current_time}.json"
    
    # Setup agents
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(args.mode, len(agent_names))
    
    # Create CacheGraph instead of Graph
    graph = CacheGraph(
        domain="gsm8k",
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        use_cache_communication=args.use_cache,
        hidden_dim=args.hidden_dim,
        num_cache_layers=args.num_cache_layers,
        **kwargs
    )
    
    graph.gcn.train()
    
    # Optimizer includes cache fuser parameters if using cache
    params = list(graph.gcn.parameters())
    if args.use_cache:
        params += list(graph.cache_fuser.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    num_batches = int(len(dataset) / args.batch_size)
    total_solved, total_executed = 0, 0
    
    for i_batch in range(num_batches):
        print(f"Batch {i_batch}", 80*'-')
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
        
        # Compute loss
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
                "Use Cache": args.use_cache
            })
        
        with open(result_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        
        # Backprop
        total_loss = torch.mean(torch.stack(loss_list))
        if args.optimized_spatial or args.optimized_temporal or args.use_cache:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Loss: {total_loss.item():.4f}")
        
        # Switch to eval after training iterations
        if i_batch + 1 == args.num_iterations:
            args.optimized_spatial = False
            args.optimized_temporal = False
            total_solved = 0
            total_executed = 0
            graph.gcn.eval()
            if args.use_cache:
                graph.cache_fuser.eval()
            print("Start Eval")

if __name__ == '__main__':
    asyncio.run(main())
