import sys
import os
import argparse
import json
import time
import asyncio
from pathlib import Path
import torch
import copy
from typing import List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.graph.graph import Graph
from GDesigner.utils.globals import Time, Cost, PromptTokens, CompletionTokens

# Import from LatentMAS
sys.path.append('/Users/bleachvex/Downloads/projects/LatentMAS')
from data import load_gpqa_diamond
from utils import normalize_answer

def gpqa_get_predict(text):
    """Extract answer from response text (a, b, c, or d)"""
    text = text.lower().strip()
    for choice in ['a', 'b', 'c', 'd']:
        if choice in text:
            return choice
    return text[:1] if text else 'a'

def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner on GPQA")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument('--mode', type=str, default='FullConnected')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--agent_names', nargs='+', type=str, default=['Reasoner'])
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4])
    parser.add_argument('--decision_method', type=str, default='FinalRefer')
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--optimized_temporal', action='store_true')
    return parser.parse_args()

async def main():
    args = parse_args()
    dataset = list(load_gpqa_diamond(split='test'))
    
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{GDesigner_ROOT}/result/gpqa")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"gpqa_{args.llm_name}_{current_time}.json"
    
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    graph = Graph(domain="gpqa", llm_name=args.llm_name, agent_names=agent_names,
                  decision_method=args.decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal)
    graph.gcn.train()
    optimizer = torch.optim.Adam(graph.gcn.parameters(), lr=args.lr)
    
    num_batches = len(dataset) // args.batch_size
    total_solved, total_executed = 0, 0
    
    for i_batch in range(num_batches):
        print(f"Batch {i_batch}", 80*'-')
        current_batch = dataset[i_batch*args.batch_size:(i_batch+1)*args.batch_size]
        
        tasks = []
        for record in current_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            tasks.append(asyncio.create_task(realized_graph.arun({"task": record["question"]}, args.num_rounds)))
        
        raw_results = await asyncio.gather(*tasks)
        raw_answers, log_probs = zip(*raw_results)
        
        loss_list, utilities = [], []
        data = load_result(result_file)
        
        for record, answer, log_prob in zip(current_batch, raw_answers, log_probs):
            predict = normalize_answer(gpqa_get_predict(answer[0]))
            gold = normalize_answer(record["gold"])
            is_solved = float(predict == gold)
            total_solved += is_solved
            total_executed += 1
            
            utilities.append(is_solved)
            loss_list.append(-log_prob * is_solved)
            
            data.append({
                "Question": record["question"],
                "Gold": gold,
                "Response": answer,
                "Predict": predict,
                "Solved": is_solved,
                "Accuracy": total_solved / total_executed
            })
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        total_loss = torch.mean(torch.stack(loss_list))
        if args.optimized_spatial or args.optimized_temporal:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        print(f"Accuracy: {total_solved/total_executed:.3f}, Loss: {total_loss.item():.3f}")
        
        if i_batch + 1 == args.num_iterations:
            args.optimized_spatial = False
            args.optimized_temporal = False
            total_solved = total_executed = 0
            graph.gcn.eval()

if __name__ == "__main__":
    asyncio.run(main())
