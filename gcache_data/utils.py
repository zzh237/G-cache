"""Evaluation utilities - copied from LatentMAS"""
import re
from typing import Optional


def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


def extract_gpqa_answer(text: str) -> Optional[str]:
    """Extract letter answer (A/B/C/D) from GPQA response"""
    # Look for patterns like "The answer is D" or "answer: D" or "Answer D"
    patterns = [
        r"[Tt]he answer is ([A-Da-d])",
        r"[Aa]nswer:\s*([A-Da-d])",
        r"[Aa]nswer\s+([A-Da-d])",
        r"\b([A-Da-d])\s*$",  # Letter at end of text
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    # Fallback: look for last occurrence of A, B, C, or D
    letters = re.findall(r"\b([A-Da-d])\b", text)
    if letters:
        return letters[-1]
    
    return None
