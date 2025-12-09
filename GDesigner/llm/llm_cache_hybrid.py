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
                 cache_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 api_model_name: str = "qwen-flash",
                 device: str = "cuda:0",
                 device_map: str = "auto"):
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
        generation_mode: str = "api_hint",  # "api_hint", "hybrid", "local"
        **kwargs
    ) -> Tuple[str, Tuple]:
        """
        Generate with cache
        
        Args:
            generation_mode: "api_hint" (API with text hint), "hybrid" (local+API), "local" (local only)
        
        Returns:
            (text_response, kv_cache)
        """
        print(f"\n📦 [STEP 7] HybridCacheLLM.agen_with_cache() - Starting cache generation")
        
        print(f"   💬 [STEP 7a] Converting messages to text prompt...")
        prompt = self._messages_to_text(messages)
        print(f"   ✅ Prompt length: {len(prompt)} characters")
        
        print(f"   🔤 [STEP 7b] Tokenizing prompt...")
        # Get model's max length (default to 2048 if not available)
        max_length = getattr(self.tokenizer, 'model_max_length', 2048)
        if max_length > 100000:  # Some tokenizers return huge default values
            max_length = 2048
        
        encoded = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,  # Truncate if too long
            max_length=max_length  # Respect model's max length
        )
        print(f"   🔤 [STEP 7b] Tokenizing finished...")
        input_ids = encoded["input_ids"].to(self.hybrid_model.device)
        attention_mask = encoded["attention_mask"].to(self.hybrid_model.device)
        
        if input_ids.shape[1] >= max_length:
            print(f"   ⚠️ Prompt truncated from {len(prompt)} chars to {input_ids.shape[1]} tokens (max: {max_length})")
        print(f"   ✅ Tokenized to {input_ids.shape[1]} tokens")
        
        # Step 1: Generate real KV-cache with small local model
        has_input_cache = past_key_values is not None
        if has_input_cache:
            print(f"\n   🔗 [CACHE] Using past_key_values from predecessors: {len(past_key_values)} layers")
        else:
            print(f"\n   🆕 [CACHE] No past_key_values - generating from scratch")
        
        print(f"\n🔗 [STEP 8] HybridCacheLLM - Calling hybrid_model.generate_latent_batch()")
        cache_kv = self.hybrid_model.generate_latent_batch(
            input_ids,
            attention_mask,
            latent_steps=latent_steps,
            past_key_values=past_key_values,
        )
        
        print(f"   ✅ [CACHE] Generated cache with {len(cache_kv)} layers, seq_len={cache_kv[0][0].shape[2]}")
        
        # Step 2: Generate text based on mode
        print(f"\n📝 [STEP 9] HybridCacheLLM - Generating text with mode: {generation_mode}")
        if generation_mode == "hybrid":
            # HYBRID: Local model + API refinement
            print(f"   ⭐ [MODE] HYBRID - Calling generate_text_batch_hybrid()")
            text, cache_kv = await self.hybrid_model.generate_text_batch_hybrid(
                input_ids,
                messages,
                attention_mask=attention_mask,
                past_key_values=cache_kv,
                max_tokens=kwargs.get("max_tokens", 256)
            )
        elif generation_mode == "local":
            # LOCAL: Local model only (real cache usage)
            print(f"   🖥️  [MODE] LOCAL - Calling generate_text_batch()")
            text, cache_kv = self.hybrid_model.generate_text_batch(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=cache_kv,
                max_new_tokens=kwargs.get("max_tokens", 256)
            )
        else:  # api_hint
            # API_HINT: API with text hint
            print(f"   🌐 [MODE] API_HINT - Calling generate_text_batch_api()")
            text, _ = await self.hybrid_model.generate_text_batch_api(
                messages,
                past_key_values=cache_kv,
                max_tokens=kwargs.get("max_tokens", 256)
            )
        
        print(f"   📝 [RESULT] Generated {len(text[0])} characters of text")
        
        return text[0], cache_kv
