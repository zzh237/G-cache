"""Data loaders - copied from LatentMAS"""
from typing import Dict, Iterable, Optional
from datasets import load_dataset
from .utils import extract_gold, normalize_answer


def load_gpqa_diamond(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("fingertap/GPQA-Diamond", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        answer = item["answer"].strip()
        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_gsm8k(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        solution = item["answer"]
        gold = normalize_answer(extract_gold(solution))
        yield {
            "question": question,
            "solution": solution,
            "gold": gold,
        }
