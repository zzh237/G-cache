"""
Dataset loader using HuggingFace datasets library
"""
from typing import List, Dict
from datasets import load_dataset


def load_humaneval(split: str = "test", cache_dir: str = None) -> List[Dict]:
    """Load HumanEval dataset from HuggingFace"""
    ds = load_dataset("openai_humaneval", split=split, cache_dir=cache_dir)
    
    dataset = []
    for item in ds:
        dataset.append({
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "test": item["test"],
            "entry_point": item["entry_point"],
            "canonical_solution": item.get("canonical_solution", "")
        })
    
    return dataset
