"""
Hybrid Cache LLM: Small local model for cache + API for text
Best of both worlds: Real KV-cache + Free API
"""
import torch
from typing import List, Dict, Optional, Tuple, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hybrid_cache_model import HybridCacheModel
from GDesigner.llm.llm_registry import LLMRegistry


@LLMRegistry.register('hybrid_cache')
class HybridCacheLLM:
    """
    Hybrid approach:
    1. Small local model (Qwen-1.5B) generates real KV-cache
    2. Your free API (qwen-plus) generates final text
    
    Perfect for: Free API + small GPU (4GB)
    """
    def __init__(self, 
                 cache_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",  # Larger model for better cache
                 api_model_name: str = "qwen-flash",  # Changed to qwen-flash (faster & cheaper)
                 device: str = "cuda:0",
                 device_map: str = "auto"):  # Auto-distribute across GPUs
        self.model_name = api_model_name
        
        # Initialize hybrid model
        self.hybrid_model = HybridCacheModel(
            cache_model_name=cache_model_name,
            api_model_name=api_model_name,
            device=device,
            device_map=device_map  # Pass device_map
        )
        self.tokenizer = self.hybrid_model.tokenizer
    
    def _messages_to_text(self, messages: List[Dict]) -> str:
        """Convert messages to text prompt"""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<|{role}|>\n{content}\n"
        text += "<|assistant|>\n"
        return text
    
    async def agen(self, messages: List[Dict], **kwargs) -> str:
        """Standard generation (API only, no cache)"""
        text, _ = await self.hybrid_model.generate_text_batch_api(
            messages,
            past_key_values=None,
            max_tokens=kwargs.get("max_tokens", 256)
        )
        return text[0]
    
    async def agen_with_cache(
        self,
        messages: List[Dict],
        past_key_values: Optional[Tuple] = None,
        latent_steps: int = 10,
        **kwargs
    ) -> Tuple[str, Tuple]:
        """
        Generate with hybrid cache
        
        Flow:
        1. Small local model generates real KV-cache
        2. API generates final text (cache converted to context)
        
        Returns:
            (text_response, kv_cache)
        """
        prompt = self._messages_to_text(messages)
        encoded = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(self.hybrid_model.device)
        attention_mask = encoded["attention_mask"].to(self.hybrid_model.device)
        
        # Step 1: Generate real KV-cache with small local model
        cache_kv = self.hybrid_model.generate_latent_batch(
            input_ids,
            attention_mask,
            latent_steps=latent_steps,
            past_key_values=past_key_values,
        )
        
        # Step 2: Generate text with API (cache as context)
        text, _ = await self.hybrid_model.generate_text_batch_api(
            messages,
            past_key_values=cache_kv,
            max_tokens=kwargs.get("max_tokens", 256)
        )
        
        return text[0], cache_kv
