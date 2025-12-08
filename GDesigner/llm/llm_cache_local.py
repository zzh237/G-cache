"""
Local Cache-enabled LLM for G-cache
Uses LatentMAS-style KV-cache generation with graph-guided fusion
"""
import torch
from typing import List, Dict, Optional, Tuple, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from latent_cache_wrapper import LatentCacheModel
from GDesigner.llm.llm_registry import LLMRegistry


@LLMRegistry.register('local_cache')
class LocalCacheLLM:
    """
    Local model with LatentMAS-style cache generation
    Integrates: generate_latent_batch + generate_text_batch
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 device: str = "cuda:0"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize LatentMAS cache model
        self.cache_model = LatentCacheModel(model_name, self.device)
        self.tokenizer = self.cache_model.tokenizer
    
    def _messages_to_text(self, messages: List[Dict]) -> str:
        """Convert messages to text prompt"""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        
        # Fallback
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<|{role}|>\\n{content}\\n"
        text += "<|assistant|>\\n"
        return text
    
    async def agen(self, messages: List[Dict], **kwargs) -> str:
        """Standard text generation (no cache)"""
        prompt = self._messages_to_text(messages)
        encoded = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        texts, _ = self.cache_model.generate_text_batch(
            input_ids,
            attention_mask,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
        )
        return texts[0]
    
    async def agen_with_cache(
        self,
        messages: List[Dict],
        past_key_values: Optional[Tuple] = None,
        latent_steps: int = 10,
        **kwargs
    ) -> Tuple[str, Any]:
        """
        Generate with cache (LatentMAS-style)
        
        Returns:
            (text_response, kv_cache)
        """
        prompt = self._messages_to_text(messages)
        encoded = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Step 1: Generate latent cache (LatentMAS)
        past_kv = self.cache_model.generate_latent_batch(
            input_ids,
            attention_mask,
            latent_steps=latent_steps,
            past_key_values=past_key_values,  # Use fused cache from graph
        )
        
        # Step 2: Generate text from cache
        texts, final_kv = self.cache_model.generate_text_batch(
            input_ids,
            attention_mask,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
            past_key_values=past_kv,
        )
        
        return texts[0], past_kv  # Return text + cache for graph storage
